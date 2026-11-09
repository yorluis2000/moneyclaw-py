"""Tests for the Alpaca stock executor (no network: httpx.MockTransport)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from moneyclaw.config.settings import AlpacaSettings, RiskSettings
from moneyclaw.execution.alpaca import (
    LIVE_URL,
    PAPER_URL,
    AlpacaExecutor,
    ForbiddenEndpointError,
    _is_regular_session,
)


def _risk(**kw) -> RiskSettings:
    base = {"max_trade_amount": 20.0, "max_portfolio_budget": 50.0}
    return RiskSettings(**{**base, **kw})


def _creds(paper: bool = True) -> AlpacaSettings:
    return AlpacaSettings(api_key="key", secret_key="secret", paper=paper)


class FakeAlpaca:
    """Minimal in-memory Alpaca Trading API."""

    def __init__(self, market_open: bool = True) -> None:
        self.market_open = market_open
        self.requests: list[httpx.Request] = []
        self.positions: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, method = request.url.path, request.method
        if path == "/v2/clock":
            return httpx.Response(200, json={"is_open": self.market_open})
        if path == "/v2/orders" and method == "POST":
            body = json.loads(request.content)
            return httpx.Response(
                200, json={"id": "ord-1", "status": "accepted", "filled_qty": "0", **body}
            )
        if path == "/v2/positions" and method == "GET":
            return httpx.Response(200, json=self.positions)
        if path.startswith("/v2/positions/") and method == "GET":
            return httpx.Response(200, json={"symbol": path.rsplit("/", 1)[1]})
        if path.startswith("/v2/positions/") and method == "DELETE":
            return httpx.Response(200, json={"id": "ord-2", "status": "accepted"})
        if path == "/v2/account":
            return httpx.Response(
                200, json={"equity": "1000", "cash": "900", "buying_power": "900"}
            )
        return httpx.Response(404, json={"message": "not found"})


def _live(fake: FakeAlpaca, **risk) -> AlpacaExecutor:
    return AlpacaExecutor(
        _creds(), _risk(**risk), dry_run=False, transport=httpx.MockTransport(fake)
    )


class TestDryRun:
    async def test_buy_simulates_fill(self) -> None:
        ex = AlpacaExecutor(AlpacaSettings(), _risk(), dry_run=True)
        order = await ex.buy("NVDA", 10, price=100.0)
        assert order.ok and order.dry_run and order.status == "filled"
        assert order.qty == pytest.approx(0.1)
        assert ex.deployed_capital == pytest.approx(10.0)
        assert ex.get_position("NVDA").avg_entry_price == 100.0

    async def test_close_realizes_pl(self) -> None:
        ex = AlpacaExecutor(AlpacaSettings(), _risk(), dry_run=True)
        await ex.buy("AMD", 10, price=100.0)
        order = await ex.close_position("AMD", price=110.0)
        assert order.ok and order.realized_pl == pytest.approx(1.0)
        assert ex.deployed_capital == 0
        assert ex.get_position("AMD") is None

    async def test_mark_price_updates_value(self) -> None:
        ex = AlpacaExecutor(AlpacaSettings(), _risk(), dry_run=True)
        await ex.buy("LLY", 10, price=100.0)
        ex.mark_price("LLY", 90.0)
        pos = (await ex.get_positions())[0]
        assert pos.market_value == pytest.approx(9.0)
        assert pos.unrealized_pl == pytest.approx(-1.0)

    async def test_buy_needs_reference_price(self) -> None:
        ex = AlpacaExecutor(AlpacaSettings(), _risk(), dry_run=True)
        assert not (await ex.buy("NVDA", 10)).ok

    async def test_close_without_position(self) -> None:
        ex = AlpacaExecutor(AlpacaSettings(), _risk(), dry_run=True)
        assert (await ex.close_position("NVDA")).status == "rejected"


class TestBudgetCaps:
    async def test_rejects_over_max_trade(self) -> None:
        ex = AlpacaExecutor(AlpacaSettings(), _risk(max_trade_amount=20), dry_run=True)
        order = await ex.buy("NVDA", 25, price=100.0)
        assert order.status == "rejected" and "max trade" in order.reason

    async def test_never_exceeds_portfolio_budget(self) -> None:
        ex = AlpacaExecutor(AlpacaSettings(), _risk(max_portfolio_budget=50), dry_run=True)
        for sym in ("NVDA", "AMD"):
            assert (await ex.buy(sym, 20, price=100.0)).ok
        assert ex.max_buy_amount() == pytest.approx(10.0)
        blocked = await ex.buy("PLTR", 20, price=10.0)
        assert blocked.status == "rejected" and "budget" in blocked.reason
        assert (await ex.buy("PLTR", 10, price=10.0)).ok
        assert ex.deployed_capital == pytest.approx(50.0)
        assert ex.max_buy_amount() == 0

    async def test_selling_frees_budget(self) -> None:
        ex = AlpacaExecutor(AlpacaSettings(), _risk(max_portfolio_budget=20), dry_run=True)
        await ex.buy("NVDA", 20, price=100.0)
        assert not (await ex.buy("AMD", 5, price=100.0)).ok
        await ex.close_position("NVDA", price=100.0)
        assert (await ex.buy("AMD", 5, price=100.0)).ok

    async def test_limits_read_live_from_settings(self) -> None:
        risk = _risk(max_portfolio_budget=50)
        ex = AlpacaExecutor(AlpacaSettings(), risk, dry_run=True)
        risk.max_portfolio_budget = 5.0  # e.g. changed via Telegram /budget
        assert not (await ex.buy("NVDA", 10, price=100.0)).ok

    async def test_minimum_notional(self) -> None:
        ex = AlpacaExecutor(AlpacaSettings(), _risk(), dry_run=True)
        assert "minimum" in (await ex.buy("NVDA", 0.5, price=100.0)).reason

    async def test_invalid_symbol(self) -> None:
        ex = AlpacaExecutor(AlpacaSettings(), _risk(), dry_run=True)
        assert (await ex.buy("not a ticker", 5, price=1.0)).reason == "invalid symbol"


class TestLive:
    async def test_buy_sends_notional_market_order(self) -> None:
        fake = FakeAlpaca()
        ex = _live(fake)
        order = await ex.buy("nvda", 12.345)
        assert order.ok and order.id == "ord-1" and not order.dry_run
        post = next(r for r in fake.requests if r.method == "POST")
        assert str(post.url).startswith(PAPER_URL)
        assert post.headers["APCA-API-KEY-ID"] == "key"
        assert json.loads(post.content) == {
            "symbol": "NVDA",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "notional": "12.35",
        }
        # Counts against the budget until it shows up as a position
        assert ex.deployed_capital == pytest.approx(12.35)

    async def test_live_url_when_not_paper(self) -> None:
        fake = FakeAlpaca()
        ex = AlpacaExecutor(
            _creds(paper=False), _risk(), dry_run=False, transport=httpx.MockTransport(fake)
        )
        await ex.buy("NVDA", 5)
        assert all(str(r.url).startswith(LIVE_URL) for r in fake.requests)

    async def test_market_closed_skips_order(self) -> None:
        fake = FakeAlpaca(market_open=False)
        order = await _live(fake).buy("NVDA", 5)
        assert order.status == "skipped" and order.reason == "market closed"
        assert not any(r.method == "POST" for r in fake.requests)

    async def test_refresh_replaces_in_flight_with_position(self) -> None:
        fake = FakeAlpaca()
        ex = _live(fake)
        await ex.buy("NVDA", 10)
        fake.positions = [
            {
                "symbol": "NVDA",
                "qty": "0.1",
                "avg_entry_price": "100",
                "cost_basis": "10",
                "market_value": "11",
                "unrealized_pl": "1",
            }
        ]
        await ex.refresh()
        assert ex.deployed_capital == pytest.approx(10.0)
        assert ex.get_position("NVDA").market_value == 11.0

    async def test_close_position_uses_positions_endpoint(self) -> None:
        fake = FakeAlpaca()
        fake.positions = [
            {
                "symbol": "AMD",
                "qty": "0.2",
                "avg_entry_price": "50",
                "cost_basis": "10",
                "market_value": "9",
                "unrealized_pl": "-1",
            }
        ]
        ex = _live(fake)
        order = await ex.close_position("AMD")
        assert order.ok and order.realized_pl == -1.0
        assert any(
            r.method == "DELETE" and r.url.path == "/v2/positions/AMD" for r in fake.requests
        )

    async def test_account(self) -> None:
        acct = await _live(FakeAlpaca()).get_account()
        assert acct["equity"] == 1000.0

    def test_live_requires_credentials(self) -> None:
        with pytest.raises(ValueError):
            AlpacaExecutor(AlpacaSettings(), _risk(), dry_run=False)


class TestEndpointAllowlist:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("POST", "/v1/accounts/abc/transfers"),
            ("GET", "/v1/accounts/abc/ach_relationships"),
            ("POST", "/v1/accounts/abc/ach_relationships"),
            ("POST", "/v1/journals"),
            ("GET", "/v2/account/activities"),
            ("PATCH", "/v2/account/configurations"),
            ("POST", "/v2/positions/NVDA"),
            ("GET", "/v1/accounts/abc/recipient_banks"),
            ("POST", "/v2/wallets/transfers"),
        ],
    )
    async def test_forbidden(self, method: str, path: str) -> None:
        fake = FakeAlpaca()
        with pytest.raises(ForbiddenEndpointError):
            await _live(fake)._request(method, path)
        assert fake.requests == []  # refused before any network call

    async def test_order_body_fields_restricted(self) -> None:
        fake = FakeAlpaca()
        with pytest.raises(ForbiddenEndpointError):
            await _live(fake)._request(
                "POST", "/v2/orders", json={"symbol": "NVDA", "side": "buy", "client_extra": 1}
            )

    async def test_ticker_containing_blocked_word_is_fine(self) -> None:
        fake = FakeAlpaca()
        ex = _live(fake)
        await ex._request("GET", "/v2/positions/ACHC")  # "ach" inside a ticker
        assert fake.requests[-1].url.path == "/v2/positions/ACHC"


def test_regular_session_hours() -> None:
    assert _is_regular_session(datetime(2026, 11, 9, 15, 0, tzinfo=UTC))  # Mon 10:00 ET
    assert not _is_regular_session(datetime(2026, 11, 9, 13, 0, tzinfo=UTC))  # 08:00 ET
    assert not _is_regular_session(datetime(2026, 11, 8, 15, 0, tzinfo=UTC))  # Sunday
