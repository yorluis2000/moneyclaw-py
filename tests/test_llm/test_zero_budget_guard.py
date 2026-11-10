"""$0 token guard: with no LLM budget, paid models are never discovered or selected."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from moneyclaw.config.settings import LLMSettings
from moneyclaw.llm.budget_manager import BudgetManager, BudgetPolicy
from moneyclaw.llm.cost_tracker import CostTracker
from moneyclaw.llm.model_discovery import LOCAL_PROVIDERS, ModelDiscoveryService
from moneyclaw.llm.model_intelligence import create_profile_from_model_id
from moneyclaw.llm.model_profile import TaskType
from moneyclaw.llm.performance_tracker import PerformanceTracker
from moneyclaw.llm.smart_router import SmartRouter
from moneyclaw.llm.types import TaskRequest


def _router(daily_budget: float, model_ids: list[str]) -> SmartRouter:
    tracker = CostTracker(daily_budget=daily_budget)
    router = SmartRouter(
        registry=MagicMock(),
        cost_tracker=tracker,
        budget_manager=BudgetManager(
            cost_tracker=tracker, policy=BudgetPolicy(daily_budget=daily_budget)
        ),
        performance_tracker=PerformanceTracker(),
    )
    for mid in model_ids:
        profile = create_profile_from_model_id(mid, mid.split("/")[0])
        router._providers[mid] = SimpleNamespace(model_profile=profile, model_id=mid)
    return router


def _candidates(router: SmartRouter) -> list[str]:
    found = router._filter_candidates(
        task_type=TaskType.CONVERSATION,
        complexity=0.0,
        require_tools=False,
        require_vision=False,
        require_json_mode=False,
        max_cost=router._budget_manager.get_routing_strategy()["max_cost_per_call"],
    )
    return [p.model_id for p in found]


def test_default_llm_budget_is_zero() -> None:
    assert LLMSettings().daily_llm_budget == 0.0


def test_zero_budget_blocks_paid_models_even_if_priced_free() -> None:
    # "openai/some-new-model" has no pricing entry → estimated $0, but it's still paid
    router = _router(0.0, ["openai/gpt-4o", "openai/some-new-model", "ollama/qwen2.5:7b"])
    assert router._budget_manager.get_routing_strategy()["max_cost_per_call"] == 0.0
    assert _candidates(router) == ["ollama/qwen2.5:7b"]


async def test_zero_budget_preferred_paid_model_refused() -> None:
    router = _router(0.0, ["openai/gpt-4o"])
    picked = await router._select_model(
        TaskRequest(prompt="hi", preferred_model="openai/gpt-4o"), TaskType.CONVERSATION, 0.0
    )
    assert picked is None


def test_positive_budget_allows_paid_models() -> None:
    router = _router(1.0, ["openai/gpt-4o-mini"])
    assert _candidates(router) == ["openai/gpt-4o-mini"]


def test_exhausted_budget_falls_back_to_local_only() -> None:
    router = _router(0.01, ["openai/gpt-4o-mini", "ollama/qwen2.5:7b"])
    assert "openai/gpt-4o-mini" in _candidates(router)
    router._cost_tracker.record(
        layer=2, model="openai/gpt-4o-mini", input_tokens=1, output_tokens=1, cost=0.02, latency=0.1
    )  # spend more than the $0.01 budget
    assert router._budget_manager.budget_remaining == 0
    assert _candidates(router) == ["ollama/qwen2.5:7b"]


async def test_local_only_discovery_never_contacts_cloud(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")
    svc = ModelDiscoveryService(api_keys={"ANTHROPIC_API_KEY": "x"}, local_only=True)
    seen: list[str] = []

    async def fake_discover(provider, config):
        seen.append(provider)
        return SimpleNamespace(provider=provider, models=[], success=True)

    svc._discover_provider = fake_discover
    await svc.discover_all()
    assert seen and set(seen) <= LOCAL_PROVIDERS


async def test_rule_based_opportunity_never_calls_llm() -> None:
    from moneyclaw.agent.evaluator import Evaluator
    from moneyclaw.plugins.base import Opportunity

    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=AssertionError("LLM must not be called"))
    score = await Evaluator(llm).score(Opportunity(title="BUY NVDA", pre_score=0.8))
    assert score.value == 0.8
    llm.complete.assert_not_called()
