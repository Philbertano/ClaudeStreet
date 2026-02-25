"""Lambda handler for the Analyst agent."""

from claudestreet.handlers.base import create_handler
from claudestreet.agents.analyst import AnalystAgent

handler = create_handler(AnalystAgent)
