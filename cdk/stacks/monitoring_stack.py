"""Monitoring stack — observability and alerting.

Creates:
  - CloudWatch dashboard with key trading metrics
  - Alarms for circuit breakers, error rates, DLQ depth
  - Trading-specific alarms: PnL loss, drawdown, heartbeat gaps, stuck orders
  - SNS topic for alerts (email/SMS/Slack)
  - Log metric filters for trade events
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import (
    aws_cloudwatch as cw,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
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

        # ── Trading-specific CloudWatch alarms ──

        # Daily PnL loss > 3%
        daily_pnl_alarm = cw.Alarm(
            self, "DailyPnLAlarm",
            alarm_name=f"{prefix}-daily-pnl-loss",
            metric=cw.Metric(
                namespace="ClaudeStreet",
                metric_name="DailyPnL",
                period=Duration.hours(1),
                statistic="Minimum",
            ),
            threshold=-3.0,
            evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
            alarm_description="Daily PnL loss exceeds 3% — circuit breaker may be needed",
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        daily_pnl_alarm.add_alarm_action(cw_actions.SnsAction(self.alerts_topic))

        # Drawdown > 10%
        drawdown_alarm = cw.Alarm(
            self, "DrawdownAlarm",
            alarm_name=f"{prefix}-max-drawdown",
            metric=cw.Metric(
                namespace="ClaudeStreet",
                metric_name="MaxDrawdown",
                period=Duration.hours(1),
                statistic="Maximum",
            ),
            threshold=10.0,
            evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="Portfolio drawdown exceeds 10%",
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        drawdown_alarm.add_alarm_action(cw_actions.SnsAction(self.alerts_topic))

        # No sentinel heartbeat > 5 minutes
        sentinel_heartbeat_alarm = cw.Alarm(
            self, "SentinelHeartbeatAlarm",
            alarm_name=f"{prefix}-sentinel-no-heartbeat",
            metric=agents.functions["sentinel"].metric_invocations(period=Duration.minutes(5)),
            threshold=0,
            evaluation_periods=2,
            comparison_operator=cw.ComparisonOperator.LESS_THAN_OR_EQUAL_TO_THRESHOLD,
            alarm_description="Sentinel agent has not executed in 10 minutes",
            treat_missing_data=cw.TreatMissingData.BREACHING,
        )
        sentinel_heartbeat_alarm.add_alarm_action(cw_actions.SnsAction(self.alerts_topic))

        # Orders stuck > 5 min (executor handler duration)
        executor_duration_alarm = cw.Alarm(
            self, "ExecutorDurationAlarm",
            alarm_name=f"{prefix}-executor-slow",
            metric=cw.Metric(
                namespace="ClaudeStreet",
                metric_name="HandlerDurationMs",
                dimensions_map={"Agent": "executor"},
                period=Duration.minutes(5),
                statistic="Maximum",
            ),
            threshold=300000,  # 5 minutes in ms
            evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="Executor agent taking too long — orders may be stuck",
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        executor_duration_alarm.add_alarm_action(cw_actions.SnsAction(self.alerts_topic))

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

        # Row 3: Trading-specific EMF metrics
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Trades Executed",
                left=[cw.Metric(
                    namespace="ClaudeStreet",
                    metric_name="TradesExecuted",
                    dimensions_map={"Agent": "executor"},
                    period=Duration.minutes(5),
                    statistic="Sum",
                )],
                width=8,
                height=6,
            ),
            cw.GraphWidget(
                title="Handler Duration by Agent",
                left=[
                    cw.Metric(
                        namespace="ClaudeStreet",
                        metric_name="HandlerDurationMs",
                        dimensions_map={"Agent": agent_name},
                        period=Duration.minutes(5),
                        statistic="Average",
                    )
                    for agent_name in agents.functions
                ],
                width=8,
                height=6,
            ),
            cw.GraphWidget(
                title="Events Published",
                left=[
                    cw.Metric(
                        namespace="ClaudeStreet",
                        metric_name="EventsPublished",
                        dimensions_map={"Agent": agent_name},
                        period=Duration.minutes(5),
                        statistic="Sum",
                    )
                    for agent_name in agents.functions
                ],
                width=8,
                height=6,
            ),
        )

        # Row 4: DynamoDB metrics
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

        # Row 5: EventBridge metrics
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
