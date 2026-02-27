#!/usr/bin/env python3
"""CDK app entry point for ClaudeStreet infrastructure."""

import os

import aws_cdk as cdk

from stacks.core_stack import CoreStack
from stacks.agents_stack import AgentsStack
from stacks.order_workflow_stack import OrderWorkflowStack
from stacks.monitoring_stack import MonitoringStack

app = cdk.App()

env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
)

prefix = app.node.try_get_context("stack_prefix") or "claudestreet"

# Core: EventBridge, DynamoDB, S3, Secrets, IAM
core = CoreStack(app, f"{prefix}-core", env=env, prefix=prefix)

# Agents: Lambda functions, EventBridge rules, Scheduler, Fargate
agents = AgentsStack(
    app, f"{prefix}-agents", env=env, prefix=prefix, core=core,
)
agents.add_dependency(core)

# Order Workflow: Step Functions Express Workflow
order_workflow = OrderWorkflowStack(
    app, f"{prefix}-order-workflow", env=env, prefix=prefix,
    core=core, agents=agents,
)
order_workflow.add_dependency(agents)

# Monitoring: CloudWatch dashboard, alarms, SNS
monitoring = MonitoringStack(
    app, f"{prefix}-monitoring", env=env, prefix=prefix,
    core=core, agents=agents,
)
monitoring.add_dependency(agents)

app.synth()
