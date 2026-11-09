"""Tests for Telegram command logic (/budget, /maxtrade, /maxloss, /positions, ...)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from moneyclaw.config.settings import AlpacaSettings, RiskSettings
from moneyclaw.execution.alpaca import AlpacaExecutor
from moneyclaw.execution.risk import RiskManager
from moneyclaw.interface.telegram.bot import TelegramBot, parse_usd

TOKEN = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"  # syntactically valid, never used


def _bot(chat_id: str = "42", with_executor: bool = True):
    settings = RiskSettings()
    risk = RiskManager(settings)
    ex = AlpacaExecutor(AlpacaSettings(), settings, dry_run=True) if with_executor else None
    if ex:
        risk.set_exposure_provider(lambda: ex.deployed_capital)
    bot = TelegramBot(
        token=TOKEN,
        chat_id=chat_id,
        brain=AsyncMock(),
        memory=AsyncMock(),
        llm=MagicMock(),
        strategies=MagicMock(),
        risk=risk,
        stock_executor=ex,
    )
    return bot, risk, ex


class TestParseUsd:
    @pytest.mark.parametrize(("text", "value"), [("25", 25), ("$25.50", 25.5), (" 1,000 ", 1000)])
    def test_valid(self, text: str, value: float) -> None:
        assert parse_usd(text) == value

    @pytest.mark.parametrize("text", ["abc", "0", "-5", "nan", "inf", "1000000"])
    def test_invalid(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_usd(text)


class TestLimitCommands:
    def test_budget(self) -> None:
        bot, risk, _ = _bot()
        reply = bot.budget_text("$75")
        assert "Portfolio budget set to $75.00" in reply
        assert risk.status()["max_portfolio_budget"] == 75

    def test_maxtrade(self) -> None:
        bot, risk, _ = _bot()
        assert "Max trade set to $12.00" in bot.maxtrade_text("12")
        assert risk.status()["max_trade_amount"] == 12

    def test_maxtrade_above_budget_warns(self) -> None:
        bot, _, _ = _bot()
        assert "capped by the $50.00 budget" in bot.maxtrade_text("80")

    def test_maxloss(self) -> None:
        bot, risk, _ = _bot()
        bot.maxloss_text("8")
        assert risk.status()["daily_loss_limit"] == 8

    def test_invalid_amount_changes_nothing(self) -> None:
        bot, risk, _ = _bot()
        assert "Invalid amount" in bot.budget_text("-10")
        assert risk.status()["max_portfolio_budget"] == 50

    def test_no_argument_shows_usage_and_limits(self) -> None:
        bot, _, _ = _bot()
        reply = bot.maxloss_text("")
        assert reply.startswith("Usage: /maxloss") and "Daily loss" in reply

    async def test_budget_change_applies_to_executor(self) -> None:
        bot, _, ex = _bot()
        bot.budget_text("5")
        assert not (await ex.buy("NVDA", 10, price=100)).ok

    def test_limits_text(self) -> None:
        bot, risk, _ = _bot()
        risk.pause()
        text = bot.limits_text()
        assert "[PAUSED]" in text and "Max trade: $20.00" in text and "DRY RUN" in text


class TestPositions:
    async def test_empty(self) -> None:
        bot, _, _ = _bot()
        text = await bot.positions_text()
        assert "No open positions" in text and "Budget used: $0.00 / $50.00" in text

    async def test_with_holdings(self) -> None:
        bot, _, ex = _bot()
        await ex.buy("NVDA", 10, price=100)
        ex.mark_price("NVDA", 110)
        text = await bot.positions_text()
        assert "NVDA: 0.1000 sh, $11.00 (+1.00, +10.0%)" in text
        assert "Budget used: $10.00 / $50.00 (room $40.00)" in text

    async def test_not_configured(self) -> None:
        bot, _, _ = _bot(with_executor=False)
        assert "not configured" in await bot.positions_text()


class TestAuthorization:
    def test_owner_chat_id_parsed(self) -> None:
        assert _bot("42")[0]._authorized_chat_id() == 42

    def test_missing_chat_id_disables_commands(self) -> None:
        bot, _, _ = _bot("")
        assert bot._authorized_chat_id() is None


class TestPollingResilience:
    async def test_retries_after_network_error(self, monkeypatch) -> None:
        import asyncio

        bot, _, _ = _bot()
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("telegram unreachable")

        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(bot._dp, "start_polling", flaky)
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        await bot.start()
        assert calls["n"] == 2 and sleeps == [5.0]
