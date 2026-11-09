"""Tests for risk management."""

import pytest

from moneyclaw.config.settings import RiskSettings
from moneyclaw.execution.risk import RiskManager
from moneyclaw.plugins.base import Opportunity


def make_risk(
    max_trade: float = 50,
    max_daily_loss: float = 100,
    approval_threshold: float = 50,
    cooldown: int = 3,
) -> RiskManager:
    return RiskManager(
        RiskSettings(
            max_trade_amount=max_trade,
            max_daily_loss=max_daily_loss,
            approval_threshold=approval_threshold,
            cooldown_after_losses=cooldown,
        )
    )


class TestRiskManager:
    def test_allows_small_trade(self) -> None:
        risk = make_risk()
        opp = Opportunity(money_involved=10)
        assert risk.allow(opp) is True

    def test_blocks_trade_over_limit(self) -> None:
        risk = make_risk(max_trade=50)
        opp = Opportunity(money_involved=100)
        assert risk.allow(opp) is False

    def test_blocks_after_daily_loss_limit(self) -> None:
        risk = make_risk(max_daily_loss=20)
        # Simulate losses
        risk.record_outcome(-10)
        risk.record_outcome(-10)
        opp = Opportunity(money_involved=5)
        assert risk.allow(opp) is False

    def test_cooldown_after_consecutive_losses(self) -> None:
        risk = make_risk(cooldown=2)
        risk.record_outcome(-5)
        risk.record_outcome(-5)
        opp = Opportunity(money_involved=5)
        assert risk.allow(opp) is False

    def test_cooldown_resets_on_profit(self) -> None:
        risk = make_risk(cooldown=3)
        risk.record_outcome(-5)
        risk.record_outcome(-5)
        risk.record_outcome(10)  # Win resets counter
        opp = Opportunity(money_involved=5)
        assert risk.allow(opp) is True

    def test_needs_approval(self) -> None:
        risk = make_risk(approval_threshold=50)
        small = Opportunity(money_involved=30)
        big = Opportunity(money_involved=100)
        assert risk.needs_approval(small) is False
        assert risk.needs_approval(big) is True

    def test_pause_resume(self) -> None:
        risk = make_risk()
        opp = Opportunity(money_involved=5)
        assert risk.allow(opp) is True

        risk.pause()
        assert risk.allow(opp) is False

        risk.resume()
        assert risk.allow(opp) is True

    def test_daily_reset(self) -> None:
        risk = make_risk(max_daily_loss=20)
        risk.record_outcome(-20)
        opp = Opportunity(money_involved=5)
        assert risk.allow(opp) is False

        risk.reset_daily()
        assert risk.allow(opp) is True


class TestPortfolioBudget:
    def _risk(self, deployed: float, **kw) -> RiskManager:
        r = RiskManager(RiskSettings(max_trade_amount=20, max_portfolio_budget=50, **kw))
        r.set_exposure_provider(lambda: deployed)
        return r

    def test_defaults_sized_for_50_budget(self) -> None:
        s = RiskSettings()
        assert (s.max_trade_amount, s.max_daily_loss, s.approval_threshold) == (20, 20, 15)
        assert s.max_portfolio_budget == 50

    def test_blocks_buy_over_budget(self) -> None:
        assert self._risk(40).allow(Opportunity(money_involved=10)) is True
        assert self._risk(45).allow(Opportunity(money_involved=10)) is False

    def test_sells_bypass_size_budget_and_approval(self) -> None:
        r = self._risk(60, max_daily_loss=1)
        r.record_outcome(-5)
        sell = Opportunity(money_involved=30, data={"side": "sell"})
        assert r.allow(sell) is True
        assert r.needs_approval(sell) is False
        r.pause()
        assert r.allow(sell) is False

    def test_unknown_exposure_fails_closed(self) -> None:
        r = RiskManager(RiskSettings())

        def boom() -> float:
            raise RuntimeError

        r.set_exposure_provider(boom)
        assert r.allow(Opportunity(money_involved=1)) is False
        assert r.status()["deployed"] is None

    def test_update_limits(self) -> None:
        r = self._risk(0)
        r.update_limits(max_portfolio_budget=100, max_trade_amount=30, max_daily_loss=10)
        st = r.status()
        assert (st["max_portfolio_budget"], st["max_trade_amount"], st["daily_loss_limit"]) == (
            100,
            30,
            10,
        )

    @pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf")])
    def test_update_limits_rejects_bad_values(self, bad: float) -> None:
        with pytest.raises(ValueError):
            self._risk(0).update_limits(max_trade_amount=bad)
