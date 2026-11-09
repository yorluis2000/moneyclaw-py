"""Tests for AgentBrain integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from moneyclaw.agent.brain import AgentBrain
from moneyclaw.plugins.base import Opportunity, Result, Score


def _make_brain(*, strategies=None, notifier=None):
    """Build a brain with mocked dependencies."""
    llm = MagicMock()
    llm.cost_tracker = MagicMock()
    llm.cost_tracker.format_status.return_value = "No LLM usage today."

    memory = AsyncMock()
    memory.today_pnl = AsyncMock(return_value=0.0)
    memory.pending_count = AsyncMock(return_value=0)
    memory.get_pending = AsyncMock(return_value=[])
    memory.get_approved = AsyncMock(return_value=[])

    from moneyclaw.config.settings import RiskSettings
    from moneyclaw.execution.risk import RiskManager

    risk = RiskManager(RiskSettings())
    evaluator = MagicMock()
    planner = MagicMock()
    scheduler = MagicMock()
    scheduler.start = MagicMock()
    scheduler.stop = MagicMock()

    registry = MagicMock()
    registry.active = strategies or []
    registry.status.return_value = []

    return AgentBrain(
        llm=llm,
        memory=memory,
        planner=planner,
        evaluator=evaluator,
        strategies=registry,
        risk=risk,
        scheduler=scheduler,
        notifier=notifier,
    )


class TestBrainTick:
    async def test_tick_with_no_strategies(self) -> None:
        brain = _make_brain()
        await brain._tick()
        assert brain.tick_count == 1

    async def test_tick_scans_strategies(self) -> None:
        strategy = AsyncMock()
        strategy.name = "test"
        strategy.scan = AsyncMock(return_value=[])
        brain = _make_brain(strategies=[strategy])
        await brain._tick()
        strategy.scan.assert_called_once()

    async def test_tick_executes_scored_opportunity(self) -> None:
        opp = Opportunity(strategy_name="test", title="Test opp", money_involved=10, pre_score=0.9)
        strategy = AsyncMock()
        strategy.name = "test"
        strategy.scan = AsyncMock(return_value=[opp])
        strategy.execute = AsyncMock(return_value=Result(success=True, profit_loss=5.0))

        brain = _make_brain(strategies=[strategy])
        brain._evaluator.score = AsyncMock(return_value=Score(value=0.9, threshold=0.5))
        brain._strategies.get = MagicMock(return_value=strategy)

        await brain._tick()
        strategy.execute.assert_called_once_with(opp)
        brain._memory.record_result.assert_called_once()

    async def test_tick_blocks_risky_trade(self) -> None:
        opp = Opportunity(strategy_name="test", title="Big trade", money_involved=999)
        strategy = AsyncMock()
        strategy.name = "test"
        strategy.scan = AsyncMock(return_value=[opp])

        brain = _make_brain(strategies=[strategy])
        brain._evaluator.score = AsyncMock(return_value=Score(value=0.9, threshold=0.5))

        await brain._tick()
        # Should not execute — exceeds max_trade_amount (50)
        brain._memory.record_result.assert_not_called()

    async def test_tick_sends_notification_on_execute(self) -> None:
        opp = Opportunity(strategy_name="test", title="Test", money_involved=10, pre_score=0.9)
        strategy = AsyncMock()
        strategy.name = "test"
        strategy.scan = AsyncMock(return_value=[opp])
        strategy.execute = AsyncMock(
            return_value=Result(success=True, profit_loss=1.0, details={"dry_run": True})
        )

        notifier = AsyncMock()
        brain = _make_brain(strategies=[strategy], notifier=notifier)
        brain._evaluator.score = AsyncMock(return_value=Score(value=0.9, threshold=0.5))
        brain._strategies.get = MagicMock(return_value=strategy)

        await brain._tick()
        notifier.trade_executed.assert_called_once()

    async def test_records_outcome_with_strategy_name(self) -> None:
        opp = Opportunity(strategy_name="my_strat", title="T", money_involved=5, pre_score=0.9)
        strategy = AsyncMock()
        strategy.name = "my_strat"
        strategy.scan = AsyncMock(return_value=[opp])
        strategy.execute = AsyncMock(return_value=Result(success=True, profit_loss=-2.0))

        brain = _make_brain(strategies=[strategy])
        brain._evaluator.score = AsyncMock(return_value=Score(value=0.9, threshold=0.5))
        brain._strategies.get = MagicMock(return_value=strategy)

        await brain._tick()
        # Risk manager should track per-strategy loss
        assert brain._risk._strategy_daily_loss.get("my_strat", 0) == 2.0


class TestBrainStatus:
    async def test_get_status(self) -> None:
        brain = _make_brain()
        status = await brain.get_status()
        assert "running" in status
        assert "strategies_active" in status
        assert "today_pnl" in status
        assert "risk" in status
        assert "dry_run" in status
        assert "tick_count" in status


class TestApprovals:
    def _approved(self, money: float) -> dict:
        return {
            "id": "opp1",
            "strategy": "s",
            "title": "BUY NVDA",
            "money_involved": money,
            "data": {"symbol": "NVDA", "side": "buy"},
        }

    async def test_approved_trade_executes_with_amount(self) -> None:
        strategy = AsyncMock()
        strategy.name = "s"
        strategy.scan = AsyncMock(return_value=[])
        strategy.execute = AsyncMock(return_value=Result(success=True))
        brain = _make_brain(strategies=[strategy])
        brain._strategies.get = MagicMock(return_value=strategy)
        brain._memory.get_approved = AsyncMock(return_value=[self._approved(18.0)])

        await brain._tick()
        executed = strategy.execute.call_args.args[0]
        assert executed.id == "opp1" and executed.money_involved == 18.0

    async def test_approved_trade_rechecks_limits(self) -> None:
        strategy = AsyncMock()
        strategy.name = "s"
        strategy.scan = AsyncMock(return_value=[])
        notifier = AsyncMock()
        brain = _make_brain(strategies=[strategy], notifier=notifier)
        brain._strategies.get = MagicMock(return_value=strategy)
        brain._memory.get_approved = AsyncMock(return_value=[self._approved(18.0)])
        brain._risk.update_limits(max_trade_amount=10)  # lowered after approval was requested

        await brain._tick()
        strategy.execute.assert_not_called()
        brain._memory.set_status.assert_awaited_once_with("opp1", "blocked")
        notifier.alert.assert_awaited_once()

    async def test_risk_blocked_alert_sent_once(self) -> None:
        opp = Opportunity(
            strategy_name="s", title="BUY NVDA", money_involved=999, data={"symbol": "NVDA"}
        )
        strategy = AsyncMock()
        strategy.name = "s"
        strategy.scan = AsyncMock(return_value=[opp])
        notifier = AsyncMock()
        brain = _make_brain(strategies=[strategy], notifier=notifier)
        brain._evaluator.score = AsyncMock(return_value=Score(value=0.9, threshold=0.5))
        await brain._tick()
        await brain._tick()
        assert notifier.alert.await_count == 1

    async def test_tick_refreshes_stock_positions(self) -> None:
        brain = _make_brain()
        brain._stock_executor = AsyncMock()
        await brain._tick()
        brain._stock_executor.refresh.assert_awaited_once()


class TestScoringFallback:
    async def test_falls_back_to_strategy_evaluate_when_llm_unavailable(self) -> None:
        opp = Opportunity(strategy_name="s", title="T", money_involved=5)
        strategy = AsyncMock()
        strategy.name = "s"
        strategy.scan = AsyncMock(return_value=[opp])
        strategy.evaluate = AsyncMock(return_value=Score(value=0.9, threshold=0.5))
        strategy.execute = AsyncMock(return_value=Result(success=True))
        brain = _make_brain(strategies=[strategy])
        brain._strategies.get = MagicMock(return_value=strategy)
        brain._evaluator.score = AsyncMock(side_effect=RuntimeError("No suitable model"))

        await brain._tick()
        strategy.execute.assert_awaited_once_with(opp)

    async def test_bad_opportunity_does_not_abort_tick(self) -> None:
        bad = Opportunity(strategy_name="s", title="bad", money_involved=5)
        good = Opportunity(strategy_name="s", title="good", money_involved=5, pre_score=0.9)
        strategy = AsyncMock()
        strategy.name = "s"
        strategy.scan = AsyncMock(return_value=[bad, good])
        strategy.evaluate = AsyncMock(side_effect=ValueError("boom"))
        strategy.execute = AsyncMock(return_value=Result(success=True))
        brain = _make_brain(strategies=[strategy])
        brain._strategies.get = MagicMock(return_value=strategy)

        async def score(o):
            if o is bad:
                raise RuntimeError("llm down")
            return Score(value=0.9, threshold=0.5)

        brain._evaluator.score = score
        await brain._tick()
        strategy.execute.assert_awaited_once_with(good)
