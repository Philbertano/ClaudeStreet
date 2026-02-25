"""Lambda handler for the Chronicler agent."""

from claudestreet.handlers.base import create_handler
from claudestreet.agents.chronicler import ChroniclerAgent

handler = create_handler(ChroniclerAgent)
