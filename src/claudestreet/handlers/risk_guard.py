"""Lambda handler for the RiskGuard agent."""

from claudestreet.handlers.base import create_handler
from claudestreet.agents.risk_guard import RiskGuardAgent

handler = create_handler(RiskGuardAgent)
