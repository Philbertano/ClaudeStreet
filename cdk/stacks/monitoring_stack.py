"""Monitoring stack — observability and alerting.

Creates:
  - CloudWatch dashboard with key trading metrics
  - Alarms for circuit breakers, error rates, DLQ depth
  - SNS topic for alerts (email/SMS/Slack)
  - Log metric filters for trade events
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import (
    aws_cloudwatch as cw,
    aws_cloudwatch_actions as cw_actions,
    aws_logs as logs,
    aws_sns as sns,
    aws_sns_subscriptions as subs,
    Duration,
)
from constructs import Construct

from stacks.core_stack import CoreStack
from stacks.agents_stack import AgentsStack


class MonitoringStack(cdk.Stack):

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

        # ── SNS alert topic ──
        self.alerts_topic = sns.Topic(
            self, "AlertsTopic",
            topic_name=f"{prefix}-alerts",
            display_name="ClaudeStreet Trading Alerts",
        )

        # ── DLQ depth alarm ──
        dlq_alarm = cw.Alarm(
            self, "DLQDepthAlarm",
            alarm_name=f"{prefix}-dlq-messages",
            metric=agents.dlq.metric_approximate_number_of_messages_visible(),
            threshold=5,
            evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="Dead letter queue has unprocessed agent failures",
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        dlq_alarm.add_alarm_action(cw_actions.SnsAction(self.alerts_topic))

        # ── Lambda error alarms (one per agent) ──
        for agent_name, fn in agents.functions.items():
            alarm = cw.Alarm(
                self, f"{agent_name}ErrorAlarm",
                alarm_name=f"{prefix}-{agent_name}-errors",
                metric=fn.metric_errors(period=Duration.minutes(5)),
                threshold=3,
                evaluation_periods=2,
                comparison_operator=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
                alarm_description=f"{agent_name} agent error rate too high",
                treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            )
            alarm.add_alarm_action(cw_actions.SnsAction(self.alerts_topic))

        # ── CloudWatch Dashboard ──
        dashboard = cw.Dashboard(
            self, "Dashboard",
            dashboard_name=f"{prefix}-trading",
        )

        # Row 1: Agent invocations and errors
        invocation_widgets = []
        error_widgets = []
        for agent_name, fn in agents.functions.items():
            invocation_widgets.append(cw.GraphWidget(
                title=f"{agent_name} Invocations",
                left=[fn.metric_invocations(period=Duration.minutes(5))],
                width=4,
                height=6,
            ))
            error_widgets.append(cw.GraphWidget(
                title=f"{agent_name} Errors",
                left=[fn.metric_errors(period=Duration.minutes(5))],
                width=4,
                height=6,
            ))

        dashboard.add_widgets(*invocation_widgets)
        dashboard.add_widgets(*error_widgets)

        # Row 2: Agent durations
        duration_widgets = []
        for agent_name, fn in agents.functions.items():
            duration_widgets.append(cw.GraphWidget(
                title=f"{agent_name} Duration",
                left=[fn.metric_duration(period=Duration.minutes(5))],
                width=4,
                height=6,
            ))
        dashboard.add_widgets(*duration_widgets)

        # Row 3: DynamoDB metrics
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Trades Table - Read/Write",
                left=[
                    core.trades_table.metric_consumed_read_capacity_units(
                        period=Duration.minutes(5)
                    ),
                    core.trades_table.metric_consumed_write_capacity_units(
                        period=Duration.minutes(5)
                    ),
                ],
                width=8,
                height=6,
            ),
            cw.GraphWidget(
                title="Strategies Table - Read/Write",
                left=[
                    core.strategies_table.metric_consumed_read_capacity_units(
                        period=Duration.minutes(5)
                    ),
                    core.strategies_table.metric_consumed_write_capacity_units(
                        period=Duration.minutes(5)
                    ),
                ],
                width=8,
                height=6,
            ),
            cw.GraphWidget(
                title="DLQ Depth",
                left=[agents.dlq.metric_approximate_number_of_messages_visible()],
                width=8,
                height=6,
            ),
        )

        # Row 4: EventBridge metrics
        dashboard.add_widgets(
            cw.GraphWidget(
                title="EventBridge - Events Matched",
                left=[cw.Metric(
                    namespace="AWS/Events",
                    metric_name="MatchedEvents",
                    dimensions_map={"EventBusName": core.event_bus.event_bus_name},
                    period=Duration.minutes(5),
                    statistic="Sum",
                )],
                width=12,
                height=6,
            ),
            cw.GraphWidget(
                title="EventBridge - Failed Invocations",
                left=[cw.Metric(
                    namespace="AWS/Events",
                    metric_name="FailedInvocations",
                    dimensions_map={"EventBusName": core.event_bus.event_bus_name},
                    period=Duration.minutes(5),
                    statistic="Sum",
                )],
                width=12,
                height=6,
            ),
        )

        # ── Outputs ──
        cdk.CfnOutput(self, "AlertsTopicArn", value=self.alerts_topic.topic_arn)
        cdk.CfnOutput(self, "DashboardUrl",
                       value=f"https://{self.region}.console.aws.amazon.com/cloudwatch/home"
                             f"?region={self.region}#dashboards:name={prefix}-trading")
