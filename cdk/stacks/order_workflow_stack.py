"""Order workflow stack — Step Functions Express Workflow.

Creates:
  - Express Workflow: Validate (RiskGuard) → Submit (Executor) → Record → Notify
  - Built-in retry and timeout handling
  - Visual execution history for debugging
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import (
    aws_logs as logs,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
    Duration,
)
from constructs import Construct

from stacks.core_stack import CoreStack
from stacks.agents_stack import AgentsStack


class OrderWorkflowStack(cdk.Stack):

    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        prefix: str,
        core: CoreStack,
        agents: AgentsStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # Reference existing Lambda functions from agents stack
        risk_guard_fn = agents.functions["risk_guard"]
        executor_fn = agents.functions["executor"]
        chronicler_fn = agents.functions["chronicler"]

        # ── Step 1: Validate (RiskGuard) ──
        validate_step = sfn_tasks.LambdaInvoke(
            self, "ValidateRisk",
            lambda_function=risk_guard_fn,
            payload=sfn.TaskInput.from_json_path_at("$"),
            result_path="$.riskResult",
            retry_on_service_exceptions=True,
        )
        validate_step.add_retry(
            errors=["States.TaskFailed"],
            interval=Duration.seconds(2),
            max_attempts=2,
            backoff_rate=2.0,
        )

        # ── Step 2: Check approval ──
        check_approval = sfn.Choice(self, "RiskApproved?")

        # ── Step 3: Submit Order (Executor) ──
        submit_order = sfn_tasks.LambdaInvoke(
            self, "SubmitOrder",
            lambda_function=executor_fn,
            payload=sfn.TaskInput.from_json_path_at("$"),
            result_path="$.executionResult",
            retry_on_service_exceptions=True,
        )
        submit_order.add_retry(
            errors=["States.TaskFailed"],
            interval=Duration.seconds(5),
            max_attempts=3,
            backoff_rate=2.0,
        )

        # ── Step 4: Record trade (Chronicler) ──
        record_trade = sfn_tasks.LambdaInvoke(
            self, "RecordTrade",
            lambda_function=chronicler_fn,
            payload=sfn.TaskInput.from_json_path_at("$"),
            result_path="$.recordResult",
            retry_on_service_exceptions=True,
        )

        # ── Rejected path ──
        rejected = sfn.Pass(self, "TradeRejected")

        # ── Failed path ──
        failed = sfn.Fail(
            self, "OrderFailed",
            cause="Order execution failed after retries",
            error="OrderExecutionError",
        )

        # ── Success ──
        success = sfn.Succeed(self, "OrderComplete")

        # ── Wire the workflow ──
        definition = (
            validate_step
            .next(
                check_approval
                .when(
                    sfn.Condition.number_equals("$.riskResult.Payload.statusCode", 200),
                    submit_order
                    .next(record_trade)
                    .next(success)
                )
                .otherwise(rejected)
            )
        )

        # Add catch for submit failures
        submit_order.add_catch(failed, errors=["States.ALL"])

        # ── Create Express State Machine ──
        log_group = logs.LogGroup(
            self, "OrderWorkflowLogs",
            log_group_name=f"/aws/vendedlogs/states/{prefix}-order-workflow",
            retention=logs.RetentionDays.TWO_WEEKS,
        )

        self.state_machine = sfn.StateMachine(
            self, "OrderWorkflow",
            state_machine_name=f"{prefix}-order-workflow",
            state_machine_type=sfn.StateMachineType.EXPRESS,
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.minutes(5),
            logs=sfn.LogOptions(
                destination=log_group,
                level=sfn.LogLevel.ALL,
            ),
        )

        # ── Outputs ──
        cdk.CfnOutput(
            self, "OrderWorkflowArn",
            value=self.state_machine.state_machine_arn,
        )
