"""Lambda handler for the Executor agent."""

from claudestreet.handlers.base import create_handler
from claudestreet.agents.executor import ExecutorAgent

handler = create_handler(ExecutorAgent)
