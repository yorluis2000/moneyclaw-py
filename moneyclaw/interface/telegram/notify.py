"""Telegram notification sender — pushes alerts and reports to the user."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from aiogram import Bot

    from moneyclaw.plugins.base import Opportunity, Result

log = structlog.get_logger()


class Notifier:
    """Sends notifications to a Telegram chat."""

    def __init__(self, bot: Bot, chat_id: str) -> None:
        self._bot = bot
        self._chat_id = chat_id

    async def send(self, text: str) -> None:
        """Send a plain text message."""
        try:
            await self._bot.send_message(chat_id=self._chat_id, text=text[:4000])
        except Exception:
            log.exception("notify.send_error")

    async def request_approval(self, opp: Opportunity) -> None:
        """Send an approval request for a high-value opportunity."""
        text = (
            f"APPROVAL NEEDED\n\n"
            f"Strategy: {opp.strategy_name}\n"
            f"{opp.title}\n"
            f"Amount: ${opp.money_involved:.2f}\n\n"
            f"Reply /approve {opp.id} to execute\n"
            f"Reply /reject {opp.id} to skip"
        )
        await self.send(text)

    async def opportunity_found(self, opp: Opportunity, score_value: float) -> None:
        """Notify about a new opportunity detected."""
        text = (
            f"OPPORTUNITY\n\n"
            f"Strategy: {opp.strategy_name}\n"
            f"{opp.title}\n"
            f"Amount: ${opp.money_involved:.2f}\n"
            f"Score: {score_value:.0%}"
        )
        await self.send(text)

    async def trade_executed(self, opp: Opportunity, result: Result) -> None:
        """Notify about a completed trade."""
        pnl = result.profit_loss
        sign = "+" if pnl >= 0 else ""
        dry = " [DRY RUN]" if result.details.get("dry_run") else ""
        text = (
            f"TRADE{dry}\n\n"
            f"Strategy: {opp.strategy_name}\n"
            f"{opp.title}\n"
            f"P&L: {sign}${pnl:.2f}\n"
            f"Status: {'Success' if result.success else 'Failed'}"
        )
        if result.details.get("message"):
            text += f"\n{result.details['message']}"
        if "order_id" in result.details:
            text += f"\nOrder: {result.details['order_id']}"
        await self.send(text)

    async def daily_report(self, report: str) -> None:
        """Send the daily summary report."""
        now = datetime.now(UTC).strftime("%Y-%m-%d")
        await self.send(f"DAILY REPORT ({now})\n\n{report}")

    async def alert(self, title: str, message: str) -> None:
        """Send an urgent alert."""
        await self.send(f"ALERT: {title}\n\n{message}")

    async def risk_alert(self, reason: str, details: dict) -> None:
        """Notify about risk limit triggers."""
        text = f"RISK ALERT: {reason}\n"
        for k, v in details.items():
            text += f"  {k}: {v}\n"
        await self.send(text)

    async def strategy_alert(self, strategy_name: str, title: str, data: dict) -> None:
        """Notify about a strategy-specific event (advisory, signal, etc.)."""
        text = f"SIGNAL: {strategy_name}\n\n{title}"
        if data.get("message"):
            text += f"\n{data['message']}"
        await self.send(text)
