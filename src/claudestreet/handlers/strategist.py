"""Lambda handler for the Strategist agent."""

from claudestreet.handlers.base import create_handler
from claudestreet.agents.strategist import StrategistAgent

handler = create_handler(StrategistAgent)
