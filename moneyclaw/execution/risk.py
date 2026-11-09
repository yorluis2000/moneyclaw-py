"""Risk manager — prevents the agent from losing too much money."""

from __future__ import annotations

from collections.abc import Callable

import structlog

from moneyclaw.config.settings import RiskSettings
from moneyclaw.plugins.base import Opportunity

log = structlog.get_logger()


class RiskManager:
    """Enforces risk limits: per-trade caps, daily loss limits, cooldown periods,
    per-strategy limits, max position ratios, and dry_run mode.
    """

    def __init__(self, settings: RiskSettings) -> None:
        self._settings = settings
        self._daily_loss = 0.0
        self._consecutive_losses = 0
        self._paused = False
        # Per-strategy daily loss tracking
        self._strategy_daily_loss: dict[str, float] = {}
        # Returns USD currently deployed in open positions (set by the stock executor)
        self._exposure_provider: Callable[[], float] | None = None

    @staticmethod
    def is_exit(opp: Opportunity) -> bool:
        """Sells reduce exposure; they skip size/budget/approval limits but honor pause."""
        return opp.data.get("side") == "sell"

    def set_exposure_provider(self, provider: Callable[[], float]) -> None:
        self._exposure_provider = provider

    @property
    def deployed(self) -> float:
        if not self._exposure_provider:
            return 0.0
        try:
            return float(self._exposure_provider())
        except Exception:
            log.exception("risk.exposure_error")
            return float("inf")  # Fail closed: block new buys if exposure is unknown

    def allow(self, opp: Opportunity) -> bool:
        """Check if this opportunity passes risk controls."""
        if self._paused:
            log.info("risk.paused")
            return False

        if self.is_exit(opp):
            return True

        # Total portfolio budget across open positions
        if self.deployed + opp.money_involved > self._settings.max_portfolio_budget + 1e-9:
            log.warning(
                "risk.budget_exceeded",
                amount=opp.money_involved,
                deployed=self.deployed,
                budget=self._settings.max_portfolio_budget,
            )
            return False

        # Daily loss limit (global)
        if self._daily_loss >= self._settings.max_daily_loss:
            log.warning("risk.daily_loss_limit", loss=self._daily_loss)
            return False

        # Per-trade limit
        if opp.money_involved > self._settings.max_trade_amount:
            log.warning(
                "risk.trade_too_large",
                amount=opp.money_involved,
                limit=self._settings.max_trade_amount,
            )
            return False

        # Cooldown after consecutive losses
        if self._consecutive_losses >= self._settings.cooldown_after_losses:
            log.warning("risk.cooldown", losses=self._consecutive_losses)
            return False

        # Per-strategy daily loss limit
        if self._settings.per_strategy_daily_loss > 0 and opp.strategy_name:
            strat_loss = self._strategy_daily_loss.get(opp.strategy_name, 0.0)
            if strat_loss >= self._settings.per_strategy_daily_loss:
                log.warning(
                    "risk.strategy_daily_limit",
                    strategy=opp.strategy_name,
                    loss=strat_loss,
                )
                return False

        # Max position ratio check (advisory — caller passes current ratio)
        position_ratio = opp.data.get("position_ratio", 0.0)
        if position_ratio > self._settings.max_position_ratio:
            log.warning(
                "risk.position_too_large",
                ratio=position_ratio,
                limit=self._settings.max_position_ratio,
            )
            return False

        return True

    @property
    def is_dry_run(self) -> bool:
        """Whether all trades should be simulated."""
        return self._settings.dry_run

    def needs_approval(self, opp: Opportunity) -> bool:
        """Does this opportunity need human approval?"""
        if self.is_exit(opp):
            return False
        return opp.money_involved >= self._settings.approval_threshold

    def update_limits(
        self,
        *,
        max_portfolio_budget: float | None = None,
        max_trade_amount: float | None = None,
        max_daily_loss: float | None = None,
        approval_threshold: float | None = None,
    ) -> None:
        """Change limits at runtime (e.g. from Telegram). Values must be positive.

        Changes are in-memory; set the RISK_* variables in .env to make them permanent.
        """
        updates = {
            "max_portfolio_budget": max_portfolio_budget,
            "max_trade_amount": max_trade_amount,
            "max_daily_loss": max_daily_loss,
            "approval_threshold": approval_threshold,
        }
        for key, value in updates.items():
            if value is None:
                continue
            if not (value > 0) or value != value or value == float("inf"):
                raise ValueError(f"{key} must be a positive number")
            setattr(self._settings, key, float(value))
            log.info("risk.limit_updated", limit=key, value=value)

    @property
    def settings(self):
        return self._settings

    def record_outcome(self, profit_loss: float, strategy_name: str = "") -> None:
        """Update risk state based on trade outcome."""
        if profit_loss < 0:
            self._daily_loss += abs(profit_loss)
            self._consecutive_losses += 1
            if strategy_name:
                prev = self._strategy_daily_loss.get(strategy_name, 0.0)
                self._strategy_daily_loss[strategy_name] = prev + abs(profit_loss)
        else:
            self._consecutive_losses = 0

    def reset_daily(self) -> None:
        """Called at start of each day."""
        self._daily_loss = 0.0
        self._consecutive_losses = 0
        self._strategy_daily_loss.clear()

    def pause(self) -> None:
        self._paused = True
        log.info("risk.paused_manually")

    def resume(self) -> None:
        self._paused = False
        log.info("risk.resumed")

    @property
    def is_paused(self) -> bool:
        return self._paused

    def status(self) -> dict:
        deployed = self.deployed
        return {
            "paused": self._paused,
            "dry_run": self._settings.dry_run,
            "daily_loss": self._daily_loss,
            "daily_loss_limit": self._settings.max_daily_loss,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_threshold": self._settings.cooldown_after_losses,
            "strategy_daily_losses": dict(self._strategy_daily_loss),
            "max_position_ratio": self._settings.max_position_ratio,
            "max_trade_amount": self._settings.max_trade_amount,
            "approval_threshold": self._settings.approval_threshold,
            "max_portfolio_budget": self._settings.max_portfolio_budget,
            "deployed": deployed if deployed != float("inf") else None,
        }
