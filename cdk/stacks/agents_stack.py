"""Agents stack — Lambda functions, EventBridge rules, Scheduler, Fargate.

Creates:
  - Docker-based Lambda function per agent
  - EventBridge rules routing event types → agent Lambdas
  - EventBridge Scheduler for heartbeats
  - DLQs for failed Lambda invocations
  - Fargate task definition for the evolution engine
  - ECS cluster + scheduled Fargate task
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import (
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_lambda_destinations as lambda_destinations,
    aws_lambda_event_sources as lambda_event_sources,
    aws_logs as logs,
    aws_scheduler as scheduler,
    aws_sqs as sqs,
    Duration,
    RemovalPolicy,
)
from constructs import Construct

from stacks.core_stack import CoreStack

# Agent → (handler module, subscribed event patterns, heartbeat interval minutes)
AGENT_SPECS = {
    "sentinel": {
        "handler": "claudestreet.handlers.sentinel.handler",
        "events": ["market.tick", "market.daily"],
        "heartbeat_minutes": 1,
        "timeout_seconds": 60,
        "memory_mb": 512,
        "provisioned_concurrency": 1,
    },
    "analyst": {
        "handler": "claudestreet.handlers.analyst.handler",
        "events": ["signal.detected", "market.price_anomaly", "market.volume_spike", "analysis.sentiment"],
        "heartbeat_minutes": 5,
        "timeout_seconds": 120,
        "memory_mb": 1024,
        "provisioned_concurrency": 0,
    },
    "strategist": {
        "handler": "claudestreet.handlers.strategist.handler",
        "events": ["analysis.complete"],
        "heartbeat_minutes": None,  # evolution runs on Fargate, not Lambda
        "timeout_seconds": 30,
        "memory_mb": 256,
        "provisioned_concurrency": 0,
    },
    "risk_guard": {
        "handler": "claudestreet.handlers.risk_guard.handler",
        "events": ["strategy.trade_proposed"],
        "heartbeat_minutes": 2,
        "timeout_seconds": 30,
        "memory_mb": 256,
        "provisioned_concurrency": 2,
    },
    "executor": {
        "handler": "claudestreet.handlers.executor.handler",
        "events": ["risk.approved"],
        "heartbeat_minutes": None,
        "timeout_seconds": 30,
        "memory_mb": 256,
        "provisioned_concurrency": 2,
    },
    "chronicler": {
        "handler": "claudestreet.handlers.chronicler.handler",
        "events": [
            "execution.trade_executed", "execution.trade_failed",
            "execution.position_closed", "strategy.evolved",
            "risk.circuit_breaker",
        ],
        "heartbeat_minutes": 10,
        "timeout_seconds": 30,
        "memory_mb": 256,
        "provisioned_concurrency": 0,
    },
}


class AgentsStack(cdk.Stack):

    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        prefix: str,
        core: CoreStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)
        self.prefix = prefix
        self.core = core
        self.functions: dict[str, lambda_.Function] = {}

        # ── Docker image asset directory config ──
        docker_dir = ".."
        docker_file = "Dockerfile"
        docker_exclude = ["cdk", "cdk.out", ".git", "data", ".venv", "tests"]

        # ── Dead letter queue (shared) ──
        self.dlq = sqs.Queue(
            self, "AgentDLQ",
            queue_name=f"{prefix}-agent-dlq",
            retention_period=Duration.days(14),
            encryption=sqs.QueueEncryption.KMS,
            encryption_master_key=core.kms_key,
        )

        # ── IAM role for all agent Lambdas ──
        self.agent_role = self._create_agent_role(core)

        # ── Per-agent SQS queues (EventBridge → SQS → Lambda) ──
        self.agent_queues: dict[str, sqs.Queue] = {}

        # ── Create Lambda + SQS + EventBridge rule for each agent ──
        for agent_name, spec in AGENT_SPECS.items():
            agent_image = lambda_.DockerImageCode.from_image_asset(
                directory=docker_dir,
                file=docker_file,
                exclude=docker_exclude,
                cmd=[spec["handler"]],
            )
            fn = self._create_agent_lambda(agent_name, spec, agent_image)

            # Create per-agent SQS queue (EventBridge → SQS → Lambda)
            agent_queue = sqs.Queue(
                self, f"{agent_name}Queue",
                queue_name=f"{prefix}-{agent_name}-queue",
                visibility_timeout=Duration.seconds(spec["timeout_seconds"] * 6),
                dead_letter_queue=sqs.DeadLetterQueue(
                    max_receive_count=3,
                    queue=self.dlq,
                ),
                encryption=sqs.QueueEncryption.KMS,
                encryption_master_key=core.kms_key,
            )
            self.agent_queues[agent_name] = agent_queue

            # Lambda consumes from SQS
            fn.add_event_source(lambda_event_sources.SqsEventSource(
                agent_queue,
                batch_size=1,
            ))

            # EventBridge → SQS (instead of EventBridge → Lambda directly)
            self._create_event_rules(agent_name, spec, agent_queue, core.event_bus)

            if spec.get("heartbeat_minutes"):
                self._create_heartbeat_schedule(agent_name, spec, fn)

            # Provisioned concurrency
            pc = spec.get("provisioned_concurrency", 0)
            if pc > 0:
                fn.add_alias("live",
                    provisioned_concurrent_executions=pc,
                )

            self.functions[agent_name] = fn

        # ── DLQ Replayer Lambda ──
        self._create_dlq_replayer(docker_dir, docker_file, docker_exclude, core)

        # ── DynamoDB Stream processor Lambda ──
        self._create_stream_processor(docker_dir, docker_file, docker_exclude, core)

        # ── WebSocket feeder Fargate task ──
        self._create_websocket_feeder(core)

        # ── Fargate evolution engine ──
        self._create_fargate_evolution(core)

        # ── Outputs ──
        for name, fn in self.functions.items():
            cdk.CfnOutput(self, f"{name}FnArn", value=fn.function_arn)

    def _create_agent_role(self, core: CoreStack) -> iam.Role:
        role = iam.Role(
            self, "AgentRole",
            role_name=f"{self.prefix}-agent-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
            ],
        )

        # DynamoDB access
        for table in [
            core.trades_table, core.strategies_table,
            core.snapshots_table, core.events_table,
            core.attributions_table,
        ]:
            table.grant_read_write_data(role)

        # EventBridge publish
        core.event_bus.grant_put_events_to(role)

        # S3 read/write for sessions
        core.session_bucket.grant_read_write(role)

        # Secrets Manager read
        core.broker_secret.grant_read(role)

        # KMS decrypt
        core.kms_key.grant_decrypt(role)

        # Kinesis read (for Sentinel consuming market data stream)
        core.market_data_stream.grant_read(role)

        # SSM parameter read
        role.add_to_policy(iam.PolicyStatement(
            actions=["ssm:GetParametersByPath", "ssm:GetParameter"],
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter/{self.prefix}/*"],
        ))

        return role

    def _create_agent_lambda(
        self,
        name: str,
        spec: dict,
        image: lambda_.DockerImageCode,
    ) -> lambda_.DockerImageFunction:
        log_group = logs.LogGroup(
            self, f"{name.title()}Logs",
            log_group_name=f"/aws/lambda/{self.prefix}-{name}",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )
        return lambda_.DockerImageFunction(
            self, f"{name.title()}Fn",
            function_name=f"{self.prefix}-{name}",
            code=image,
            timeout=Duration.seconds(spec["timeout_seconds"]),
            memory_size=spec["memory_mb"],
            role=self.agent_role,
            environment={
                "CLAUDESTREET_EVENT_BUS": self.core.event_bus.event_bus_name,
                "TRADES_TABLE": self.core.trades_table.table_name,
                "STRATEGIES_TABLE": self.core.strategies_table.table_name,
                "SNAPSHOTS_TABLE": self.core.snapshots_table.table_name,
                "EVENTS_TABLE": self.core.events_table.table_name,
                "SESSION_BUCKET": self.core.session_bucket.bucket_name,
                "KINESIS_STREAM": self.core.market_data_stream.stream_name,
                "LOG_LEVEL": "INFO",
            },
            dead_letter_queue=self.dlq,
            retry_attempts=2,
            log_group=log_group,
        )

    def _create_event_rules(
        self,
        name: str,
        spec: dict,
        queue: sqs.Queue,
        bus: events.EventBus,
    ) -> None:
        """Create EventBridge rules that route event types → SQS → Lambda."""
        for event_type in spec["events"]:
            rule = events.Rule(
                self, f"{name}-{event_type.replace('.', '-')}-rule",
                rule_name=f"{self.prefix}-{name}-{event_type.replace('.', '-')}",
                event_bus=bus,
                event_pattern=events.EventPattern(
                    source=events.Match.prefix("claudestreet"),
                    detail_type=[event_type],
                ),
            )
            rule.add_target(targets.SqsQueue(
                queue,
                dead_letter_queue=self.dlq,
            ))

    def _create_heartbeat_schedule(
        self,
        name: str,
        spec: dict,
        fn: lambda_.Function,
    ) -> None:
        """Create EventBridge Scheduler rule for periodic heartbeats."""
        minutes = spec["heartbeat_minutes"]

        # Scheduler requires its own role
        scheduler_role = iam.Role(
            self, f"{name}SchedulerRole",
            assumed_by=iam.ServicePrincipal("scheduler.amazonaws.com"),
        )
        fn.grant_invoke(scheduler_role)

        scheduler.CfnSchedule(
            self, f"{name}Heartbeat",
            name=f"{self.prefix}-{name}-heartbeat",
            schedule_expression=f"rate({minutes} minute{'s' if minutes > 1 else ''})",
            flexible_time_window=scheduler.CfnSchedule.FlexibleTimeWindowProperty(
                mode="OFF",
            ),
            target=scheduler.CfnSchedule.TargetProperty(
                arn=fn.function_arn,
                role_arn=scheduler_role.role_arn,
                input='{"detail-type":"system.heartbeat","source":"claudestreet.scheduler","detail":{"type":"system.heartbeat","source":"scheduler","payload":{"agent":"'
                + name
                + '"}}}',
            ),
        )

    def _create_dlq_replayer(
        self,
        docker_dir: str,
        docker_file: str,
        docker_exclude: list[str],
        core: CoreStack,
    ) -> None:
        """Create Lambda that replays messages from the DLQ."""
        replayer_image = lambda_.DockerImageCode.from_image_asset(
            directory=docker_dir,
            file=docker_file,
            exclude=docker_exclude,
            cmd=["claudestreet.handlers.dlq_replayer.handler"],
        )
        replayer_log_group = logs.LogGroup(
            self, "DlqReplayerLogs",
            log_group_name=f"/aws/lambda/{self.prefix}-dlq-replayer",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )
        replayer_fn = lambda_.DockerImageFunction(
            self, "DlqReplayerFn",
            function_name=f"{self.prefix}-dlq-replayer",
            code=replayer_image,
            timeout=Duration.seconds(60),
            memory_size=256,
            role=self.agent_role,
            environment={
                "CLAUDESTREET_EVENT_BUS": core.event_bus.event_bus_name,
                "DLQ_URL": self.dlq.queue_url,
                "LOG_LEVEL": "INFO",
            },
            log_group=replayer_log_group,
        )

        # Grant SQS read/delete on DLQ
        self.dlq.grant_consume_messages(replayer_fn)

        # Schedule every 5 minutes to check DLQ
        replayer_rule = events.Rule(
            self, "DlqReplaySchedule",
            rule_name=f"{self.prefix}-dlq-replay-schedule",
            schedule=events.Schedule.rate(Duration.minutes(5)),
        )
        replayer_rule.add_target(targets.LambdaFunction(replayer_fn))

        self.dlq_replayer = replayer_fn

    def _create_stream_processor(
        self,
        docker_dir: str,
        docker_file: str,
        docker_exclude: list[str],
        core: CoreStack,
    ) -> None:
        """Create Lambda triggered by DynamoDB Streams on the trades table."""
        stream_image = lambda_.DockerImageCode.from_image_asset(
            directory=docker_dir,
            file=docker_file,
            exclude=docker_exclude,
            cmd=["claudestreet.handlers.stream_processor.handler"],
        )
        stream_log_group = logs.LogGroup(
            self, "StreamProcessorLogs",
            log_group_name=f"/aws/lambda/{self.prefix}-stream-processor",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )
        stream_fn = lambda_.DockerImageFunction(
            self, "StreamProcessorFn",
            function_name=f"{self.prefix}-stream-processor",
            code=stream_image,
            timeout=Duration.seconds(60),
            memory_size=256,
            role=self.agent_role,
            environment={
                "CLAUDESTREET_EVENT_BUS": core.event_bus.event_bus_name,
                "TRADES_TABLE": core.trades_table.table_name,
                "STRATEGIES_TABLE": core.strategies_table.table_name,
                "LOG_LEVEL": "INFO",
            },
            log_group=stream_log_group,
        )

        # Trigger from DynamoDB Streams on trades table
        stream_fn.add_event_source(lambda_event_sources.DynamoEventSource(
            core.trades_table,
            starting_position=lambda_.StartingPosition.TRIM_HORIZON,
            batch_size=10,
            max_batching_window=Duration.seconds(5),
            retry_attempts=3,
            bisect_batch_on_error=True,
            on_failure=lambda_destinations.SqsDestination(self.dlq),
        ))

        self.stream_processor = stream_fn

    def _create_websocket_feeder(self, core: CoreStack) -> None:
        """Create Fargate task for the WebSocket market data feeder."""
        vpc = ec2.Vpc.from_lookup(self, "FeederVpc", is_default=True)

        feeder_cluster = ecs.Cluster(
            self, "FeederCluster",
            cluster_name=f"{self.prefix}-feeder",
            vpc=vpc,
        )

        feeder_image = ecs.ContainerImage.from_asset(
            directory="..",
            file="Dockerfile",
            exclude=["cdk", "cdk.out", ".git", "data", ".venv"],
        )

        # Task role for the WebSocket feeder
        feeder_task_role = iam.Role(
            self, "FeederTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )

        # Grant Kinesis write access
        core.market_data_stream.grant_write(feeder_task_role)
        core.broker_secret.grant_read(feeder_task_role)
        core.kms_key.grant_encrypt_decrypt(feeder_task_role)

        feeder_task_role.add_to_policy(iam.PolicyStatement(
            actions=["ssm:GetParametersByPath", "ssm:GetParameter"],
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter/{self.prefix}/*"],
        ))

        feeder_task_def = ecs.FargateTaskDefinition(
            self, "FeederTaskDef",
            family=f"{self.prefix}-ws-feeder",
            cpu=256,
            memory_limit_mib=512,
            task_role=feeder_task_role,
        )

        feeder_task_def.add_container(
            "ws-feeder",
            image=feeder_image,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="ws-feeder",
                log_retention=logs.RetentionDays.TWO_WEEKS,
            ),
            environment={
                "KINESIS_STREAM": core.market_data_stream.stream_name,
                "LOG_LEVEL": "INFO",
            },
            secrets={
                "ALPACA_API_KEY": ecs.Secret.from_secrets_manager(
                    core.broker_secret, "alpaca_api_key",
                ),
                "ALPACA_SECRET_KEY": ecs.Secret.from_secrets_manager(
                    core.broker_secret, "alpaca_secret_key",
                ),
            },
            command=["python", "-m", "claudestreet.connectors.websocket_feeder"],
        )

        # Run as a long-lived Fargate service (1 task always running)
        ecs.FargateService(
            self, "FeederService",
            service_name=f"{self.prefix}-ws-feeder",
            cluster=feeder_cluster,
            task_definition=feeder_task_def,
            desired_count=1,
            assign_public_ip=True,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
        )

        self.feeder_cluster = feeder_cluster
        self.feeder_task_def = feeder_task_def

    def _create_fargate_evolution(self, core: CoreStack) -> None:
        """Create Fargate task for the LLM-powered evolution engine."""
        # VPC for Fargate (uses default VPC to minimize cost)
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        cluster = ecs.Cluster(
            self, "EvoCluster",
            cluster_name=f"{self.prefix}-evolution",
            vpc=vpc,
        )

        # Build the evolution engine Docker image
        evo_image = ecs.ContainerImage.from_asset(
            directory="..",
            file="Dockerfile",
            exclude=["cdk", "cdk.out", ".git", "data", ".venv"],
        )

        # Task execution role
        task_role = iam.Role(
            self, "EvoTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )

        # Grant permissions
        for table in [
            core.trades_table, core.strategies_table,
            core.snapshots_table, core.events_table,
        ]:
            table.grant_read_write_data(task_role)

        core.event_bus.grant_put_events_to(task_role)
        core.session_bucket.grant_read_write(task_role)
        core.broker_secret.grant_read(task_role)
        core.anthropic_secret.grant_read(task_role)
        core.kms_key.grant_decrypt(task_role)

        task_role.add_to_policy(iam.PolicyStatement(
            actions=["ssm:GetParametersByPath", "ssm:GetParameter"],
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter/{self.prefix}/*"],
        ))

        # Task definition
        task_def = ecs.FargateTaskDefinition(
            self, "EvoTaskDef",
            family=f"{self.prefix}-evolution",
            cpu=512,
            memory_limit_mib=1024,
            task_role=task_role,
        )

        task_def.add_container(
            "evolution-engine",
            image=evo_image,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="evolution",
                log_retention=logs.RetentionDays.TWO_WEEKS,
            ),
            environment={
                "CLAUDESTREET_EVENT_BUS": core.event_bus.event_bus_name,
                "TRADES_TABLE": core.trades_table.table_name,
                "STRATEGIES_TABLE": core.strategies_table.table_name,
                "SNAPSHOTS_TABLE": core.snapshots_table.table_name,
                "EVENTS_TABLE": core.events_table.table_name,
                "SESSION_BUCKET": core.session_bucket.bucket_name,
                "LOG_LEVEL": "INFO",
            },
            secrets={
                "ANTHROPIC_API_KEY": ecs.Secret.from_secrets_manager(
                    core.anthropic_secret
                ),
            },
        )

        # Schedule evolution every 15 minutes via EventBridge
        evo_rule = events.Rule(
            self, "EvoScheduleRule",
            rule_name=f"{self.prefix}-evolution-schedule",
            schedule=events.Schedule.rate(Duration.minutes(15)),
        )
        evo_rule.add_target(targets.EcsTask(
            cluster=cluster,
            task_definition=task_def,
            subnet_selection=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            assign_public_ip=True,
        ))

        # Emergency evolution on REGIME_CHANGE event
        regime_change_rule = events.Rule(
            self, "RegimeChangeEvoRule",
            rule_name=f"{self.prefix}-regime-change-evolution",
            event_bus=core.event_bus,
            event_pattern=events.EventPattern(
                source=events.Match.prefix("claudestreet"),
                detail_type=["strategy.regime_change"],
            ),
        )
        regime_change_rule.add_target(targets.EcsTask(
            cluster=cluster,
            task_definition=task_def,
            subnet_selection=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            assign_public_ip=True,
        ))

        self.evo_cluster = cluster
        self.evo_task_def = task_def
