"""Lambda handler for the Sentinel agent."""

from claudestreet.handlers.base import create_handler
from claudestreet.agents.sentinel import SentinelAgent

handler = create_handler(SentinelAgent)
