"""Core infrastructure stack — the foundation layer.

Creates:
  - EventBridge custom event bus with archive
  - DynamoDB tables (trades, strategies, snapshots, events)
  - S3 bucket for session logs and evolution archives
  - Secrets Manager secret for broker credentials
  - SSM parameters for runtime configuration
  - KMS key for encryption at rest
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import (
    aws_dynamodb as dynamodb,
    aws_events as events,
    aws_kinesis as kinesis,
    aws_kms as kms,
    aws_s3 as s3,
    aws_secretsmanager as secretsmanager,
    aws_ssm as ssm,
    RemovalPolicy,
    Duration,
)
from constructs import Construct


class CoreStack(cdk.Stack):

    def __init__(
        self, scope: Construct, id: str, *, prefix: str, **kwargs
    ) -> None:
        super().__init__(scope, id, **kwargs)
        self.prefix = prefix

        # ── KMS encryption key ──
        self.kms_key = kms.Key(
            self, "EncryptionKey",
            alias=f"alias/{prefix}",
            description="ClaudeStreet encryption key",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ── EventBridge custom bus ──
        self.event_bus = events.EventBus(
            self, "EventBus",
            event_bus_name=prefix,
        )

        # Archive all events for replay/debugging (14 day retention)
        events.CfnArchive(
            self, "EventArchive",
            source_arn=self.event_bus.event_bus_arn,
            archive_name=f"{prefix}-archive",
            retention_days=14,
        )

        # ── DynamoDB tables ──
        self.trades_table = self._create_trades_table()
        self.strategies_table = self._create_strategies_table()
        self.snapshots_table = self._create_snapshots_table()
        self.events_table = self._create_events_table()
        self.attributions_table = self._create_attributions_table()
        self.decisions_table = self._create_decisions_table()
        self.patterns_table = self._create_patterns_table()

        # ── S3 bucket for sessions and evolution archives ──
        self.session_bucket = s3.Bucket(
            self, "SessionBucket",
            bucket_name=f"{prefix}-sessions-{self.account}",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.kms_key,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="archive-old-sessions",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30),
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90),
                        ),
                    ],
                ),
            ],
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ── Secrets Manager (broker credentials) ──
        self.broker_secret = secretsmanager.Secret(
            self, "BrokerSecret",
            secret_name=f"{prefix}/broker",
            description="IG Markets broker API credentials",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"ig_api_key":"","ig_username":"","ig_password":"","ig_acc_number":"","ig_acc_type":"LIVE"}',
                generate_string_key="placeholder",
            ),
        )

        # ── Secrets Manager (Anthropic API key for evolution engine) ──
        self.anthropic_secret = secretsmanager.Secret(
            self, "AnthropicSecret",
            secret_name=f"{prefix}/anthropic",
            description="Anthropic API key for Claude Agent SDK evolution engine",
        )

        # ── Kinesis Data Stream for real-time market data ──
        self.market_data_stream = kinesis.Stream(
            self, "MarketDataStream",
            stream_name=f"{prefix}-market-data",
            shard_count=2,
            retention_period=Duration.hours(24),
            encryption=kinesis.StreamEncryption.KMS,
            encryption_key=self.kms_key,
        )

        # ── SSM parameters for runtime config ──
        ssm.StringParameter(
            self, "ConfigWatchlist",
            parameter_name=f"/{prefix}/config/watchlist",
            string_value='["AAPL","MSFT","GOOG","AMZN","NVDA","TSLA","META","JPM","V","SPY"]',
        )
        ssm.StringParameter(
            self, "ConfigTrading",
            parameter_name=f"/{prefix}/config/paper_trading",
            string_value="true",
        )

        # ── Outputs ──
        cdk.CfnOutput(self, "EventBusName", value=self.event_bus.event_bus_name)
        cdk.CfnOutput(self, "TradesTableOutput", value=self.trades_table.table_name)
        cdk.CfnOutput(self, "StrategiesTableOutput", value=self.strategies_table.table_name)
        cdk.CfnOutput(self, "SessionBucketOutput", value=self.session_bucket.bucket_name)
        cdk.CfnOutput(self, "AttributionsTableOutput", value=self.attributions_table.table_name)
        cdk.CfnOutput(self, "DecisionsTableOutput", value=self.decisions_table.table_name)
        cdk.CfnOutput(self, "PatternsTableOutput", value=self.patterns_table.table_name)
        cdk.CfnOutput(self, "MarketDataStreamOutput", value=self.market_data_stream.stream_name)

    # ── Table factories ──

    def _create_trades_table(self) -> dynamodb.Table:
        table = dynamodb.Table(
            self, "TradesTable",
            table_name=f"{self.prefix}-trades",
            partition_key=dynamodb.Attribute(
                name="trade_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.kms_key,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(point_in_time_recovery_enabled=True),
            removal_policy=RemovalPolicy.RETAIN,
            stream=dynamodb.StreamViewType.NEW_AND_OLD_IMAGES,
        )
        table.add_global_secondary_index(
            index_name="symbol-status-index",
            partition_key=dynamodb.Attribute(
                name="symbol", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="status", type=dynamodb.AttributeType.STRING
            ),
        )
        table.add_global_secondary_index(
            index_name="strategy-index",
            partition_key=dynamodb.Attribute(
                name="strategy_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="opened_at", type=dynamodb.AttributeType.STRING
            ),
        )
        return table

    def _create_strategies_table(self) -> dynamodb.Table:
        table = dynamodb.Table(
            self, "StrategiesTable",
            table_name=f"{self.prefix}-strategies",
            partition_key=dynamodb.Attribute(
                name="strategy_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.kms_key,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(point_in_time_recovery_enabled=True),
            removal_policy=RemovalPolicy.RETAIN,
        )
        table.add_global_secondary_index(
            index_name="active-index",
            partition_key=dynamodb.Attribute(
                name="is_active", type=dynamodb.AttributeType.NUMBER
            ),
            sort_key=dynamodb.Attribute(
                name="total_pnl", type=dynamodb.AttributeType.NUMBER
            ),
        )
        return table

    def _create_snapshots_table(self) -> dynamodb.Table:
        return dynamodb.Table(
            self, "SnapshotsTable",
            table_name=f"{self.prefix}-snapshots",
            partition_key=dynamodb.Attribute(
                name="date", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.kms_key,
            removal_policy=RemovalPolicy.RETAIN,
        )

    def _create_attributions_table(self) -> dynamodb.Table:
        """Signal-to-outcome attribution records."""
        table = dynamodb.Table(
            self, "AttributionsTable",
            table_name=f"{self.prefix}-attributions",
            partition_key=dynamodb.Attribute(
                name="correlation_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.kms_key,
            removal_policy=RemovalPolicy.RETAIN,
        )
        table.add_global_secondary_index(
            index_name="strategy-index",
            partition_key=dynamodb.Attribute(
                name="strategy_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
        )
        table.add_global_secondary_index(
            index_name="signal-type-index",
            partition_key=dynamodb.Attribute(
                name="signal_type", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
        )
        return table

    def _create_events_table(self) -> dynamodb.Table:
        table = dynamodb.Table(
            self, "EventsTable",
            table_name=f"{self.prefix}-events",
            partition_key=dynamodb.Attribute(
                name="event_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.kms_key,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY,
        )
        table.add_global_secondary_index(
            index_name="correlation-index",
            partition_key=dynamodb.Attribute(
                name="correlation_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
        )
        return table

    def _create_decisions_table(self) -> dynamodb.Table:
        """Decision ledger — structured reasoning record per trade pipeline step."""
        table = dynamodb.Table(
            self, "DecisionsTable",
            table_name=f"{self.prefix}-decisions",
            partition_key=dynamodb.Attribute(
                name="correlation_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="step_key", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.kms_key,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.RETAIN,
        )
        table.add_global_secondary_index(
            index_name="symbol-index",
            partition_key=dynamodb.Attribute(
                name="symbol", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
        )
        table.add_global_secondary_index(
            index_name="strategy-index",
            partition_key=dynamodb.Attribute(
                name="strategy_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
        )
        return table

    def _create_patterns_table(self) -> dynamodb.Table:
        """Pattern library — conditional performance tracking."""
        table = dynamodb.Table(
            self, "PatternsTable",
            table_name=f"{self.prefix}-patterns",
            partition_key=dynamodb.Attribute(
                name="pattern_key", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.kms_key,
            removal_policy=RemovalPolicy.RETAIN,
        )
        table.add_global_secondary_index(
            index_name="symbol-index",
            partition_key=dynamodb.Attribute(
                name="symbol", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="occurrences", type=dynamodb.AttributeType.NUMBER
            ),
        )
        return table
