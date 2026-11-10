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
    return RiskSettings(**{"max_trade_amount": 20.0, "max_portfolio_budget": 50.0, **kw})


def _creds(paper: bool = True) -> AlpacaSettings:
    return AlpacaSettings(api_key="key", secret_key="secret", paper=paper)


def _dry(cash: float = 100.0, **risk) -> AlpacaExecutor:
    return AlpacaExecutor(AlpacaSettings(dry_run_cash=cash), _risk(**risk), dry_run=True)


def _pos(symbol: str, qty: str, cost: float, value: float, side: str = "long") -> dict:
    return {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "avg_entry_price": str(cost / abs(float(qty))),
        "cost_basis": str(cost),
        "market_value": str(value),
        "unrealized_pl": str(value - cost),
    }


class FakeAlpaca:
    """Minimal in-memory Alpaca Trading API."""

    def __init__(self, market_open: bool = True, cash: str | None = "1000") -> None:
        self.market_open = market_open
        self.account: dict = {
            "equity": "1000",
            "cash": "1000",
            "buying_power": "4000",  # margin account: buying power > cash
            "non_marginable_buying_power": cash,
        }
        self.account_status = 200
        self.requests: list[httpx.Request] = []
        self.positions: dict[str, dict] = {}

    @property
    def posts(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.method == "POST"]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, method = request.url.path, request.method
        if path == "/v2/clock":
            return httpx.Response(200, json={"is_open": self.market_open})
        if path == "/v2/account":
            acct = {k: v for k, v in self.account.items() if v is not None}
            return httpx.Response(self.account_status, json=acct)
        if path == "/v2/orders" and method == "POST":
            body = json.loads(request.content)
            return httpx.Response(200, json={"id": "ord-1", "status": "accepted", **body})
        if path == "/v2/positions" and method == "GET":
            return httpx.Response(200, json=list(self.positions.values()))
        if path.startswith("/v2/positions/") and method == "GET":
            sym = path.rsplit("/", 1)[1]
            if sym in self.positions:
                return httpx.Response(200, json=self.positions[sym])
            return httpx.Response(404, json={"message": "position does not exist"})
        return httpx.Response(404, json={"message": "not found"})


def _live(fake: FakeAlpaca, paper: bool = True, **risk) -> AlpacaExecutor:
    return AlpacaExecutor(
        _creds(paper), _risk(**risk), dry_run=False, transport=httpx.MockTransport(fake)
    )


class TestDryRun:
    async def test_buy_simulates_fill(self) -> None:
        ex = _dry()
        order = await ex.buy("NVDA", 10, price=100.0)
        assert order.ok and order.dry_run and order.status == "filled"
        assert order.qty == pytest.approx(0.1)
        assert ex.deployed_capital == pytest.approx(10.0)
        assert ex.get_position("NVDA").avg_entry_price == 100.0
        assert (await ex.get_account())["non_marginable_buying_power"] == pytest.approx(90.0)

    async def test_close_realizes_pl_and_returns_cash(self) -> None:
        ex = _dry()
        await ex.buy("AMD", 10, price=100.0)
        order = await ex.close_position("AMD", price=110.0)
        assert order.ok and order.realized_pl == pytest.approx(1.0)
        assert ex.deployed_capital == 0 and ex.get_position("AMD") is None
        assert await ex.available_cash() == pytest.approx(101.0)

    async def test_partial_sell(self) -> None:
        ex = _dry()
        await ex.buy("AMD", 10, price=100.0)  # 0.1 sh
        order = await ex.sell("AMD", qty=0.04, price=100.0)
        assert order.qty == pytest.approx(0.04)
        assert ex.get_position("AMD").qty == pytest.approx(0.06)
        assert ex.deployed_capital == pytest.approx(6.0)

    async def test_mark_price_updates_value(self) -> None:
        ex = _dry()
        await ex.buy("LLY", 10, price=100.0)
        ex.mark_price("LLY", 90.0)
        pos = (await ex.get_positions())[0]
        assert pos.market_value == pytest.approx(9.0)
        assert pos.unrealized_pl == pytest.approx(-1.0)

    async def test_buy_needs_reference_price(self) -> None:
        assert not (await _dry().buy("NVDA", 10)).ok


class TestBudgetCaps:
    async def test_rejects_over_max_trade(self) -> None:
        order = await _dry(max_trade_amount=20).buy("NVDA", 25, price=100.0)
        assert order.status == "rejected" and "max trade" in order.reason

    async def test_never_exceeds_50_portfolio_budget(self) -> None:
        ex = _dry(cash=1000, max_portfolio_budget=50)  # plenty of cash, budget still binds
        for sym in ("NVDA", "AMD"):
            assert (await ex.buy(sym, 20, price=100.0)).ok
        assert ex.max_buy_amount() == pytest.approx(10.0)
        blocked = await ex.buy("PLTR", 20, price=10.0)
        assert blocked.status == "rejected" and "budget" in blocked.reason
        assert (await ex.buy("PLTR", 10, price=10.0)).ok
        assert ex.deployed_capital == pytest.approx(50.0)
        assert ex.max_buy_amount() == 0
        assert not (await ex.buy("AVGO", 1, price=10.0)).ok

    async def test_selling_frees_budget(self) -> None:
        ex = _dry(max_portfolio_budget=20)
        await ex.buy("NVDA", 20, price=100.0)
        assert not (await ex.buy("AMD", 5, price=100.0)).ok
        await ex.close_position("NVDA", price=100.0)
        assert (await ex.buy("AMD", 5, price=100.0)).ok

    async def test_limits_read_live_from_settings(self) -> None:
        risk = _risk(max_portfolio_budget=50)
        ex = AlpacaExecutor(AlpacaSettings(), risk, dry_run=True)
        risk.max_portfolio_budget = 5.0  # e.g. changed via Telegram /budget
        assert not (await ex.buy("NVDA", 10, price=100.0)).ok

    async def test_minimum_notional_and_symbol(self) -> None:
        ex = _dry()
        assert "minimum" in (await ex.buy("NVDA", 0.5, price=100.0)).reason
        assert (await ex.buy("not a ticker", 5, price=1.0)).reason == "invalid symbol"

    async def test_live_budget_counts_in_flight_orders(self) -> None:
        fake = FakeAlpaca()
        ex = _live(fake, max_portfolio_budget=50)
        assert (await ex.buy("NVDA", 20)).ok
        assert (await ex.buy("AMD", 20)).ok
        assert not (await ex.buy("PLTR", 20)).ok  # 40 in flight + 20 > 50
        assert len(fake.posts) == 2


class TestCashOnlyGuard:
    async def test_dry_run_requires_cash_strictly_greater(self) -> None:
        ex = _dry(cash=10.0)
        order = await ex.buy("NVDA", 10, price=100.0)  # cash == amount → refused
        assert order.status == "rejected" and "settled cash" in order.reason
        assert (await ex.buy("NVDA", 9.99, price=100.0)).ok

    async def test_live_uses_non_marginable_buying_power_not_margin(self) -> None:
        fake = FakeAlpaca(cash="5.00")  # buying_power is $4000 on margin, settled cash $5
        order = await _live(fake).buy("NVDA", 10)
        assert order.status == "rejected"
        assert "non-marginable buying power $5.00" in order.reason
        assert fake.posts == []  # nothing sent

    async def test_live_equal_cash_is_refused(self) -> None:
        fake = FakeAlpaca(cash="10.00")
        assert not (await _live(fake).buy("NVDA", 10)).ok
        assert fake.posts == []

    async def test_live_enough_cash_places_order(self) -> None:
        fake = FakeAlpaca(cash="10.01")
        assert (await _live(fake).buy("NVDA", 10)).ok
        assert len(fake.posts) == 1

    async def test_cash_checked_fresh_before_every_buy(self) -> None:
        fake = FakeAlpaca(cash="100")
        ex = _live(fake)
        assert (await ex.buy("NVDA", 10)).ok
        fake.account["non_marginable_buying_power"] = "3"  # cash spent elsewhere
        assert not (await ex.buy("AMD", 10)).ok
        assert len(fake.posts) == 1

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda f: f.account.update(non_marginable_buying_power=None),  # field missing
            lambda f: f.account.update(trading_blocked=True),
            lambda f: f.account.update(account_blocked=True),
            lambda f: setattr(f, "account_status", 500),  # can't read account
        ],
    )
    async def test_fails_closed(self, mutate) -> None:
        fake = FakeAlpaca(cash="1000")
        mutate(fake)
        order = await _live(fake).buy("NVDA", 10)
        assert order.status == "rejected" and "cash check failed" in order.reason
        assert fake.posts == []


class TestNoShortSelling:
    async def test_sell_is_exact_held_qty(self) -> None:
        fake = FakeAlpaca()
        fake.positions["AMD"] = _pos("AMD", "0.123456789", 10, 9)
        order = await _live(fake).close_position("AMD")
        assert order.ok and order.realized_pl == pytest.approx(-1.0)
        assert fake.posts == [
            {
                "symbol": "AMD",
                "side": "sell",
                "type": "market",
                "time_in_force": "day",
                "qty": "0.123456789",
            }
        ]

    async def test_requested_qty_clamped_to_holdings(self) -> None:
        fake = FakeAlpaca()
        fake.positions["AMD"] = _pos("AMD", "0.5", 10, 10)
        await _live(fake).sell("AMD", qty=3.0)  # asks for more than held
        assert fake.posts[0]["qty"] == "0.5"

    async def test_partial_qty_rounds_down(self) -> None:
        fake = FakeAlpaca()
        fake.positions["AMD"] = _pos("AMD", "1", 10, 10)
        await _live(fake).sell("AMD", qty=0.1234567899)
        assert fake.posts[0]["qty"] == "0.123456789"

    async def test_no_position_no_order(self) -> None:
        fake = FakeAlpaca()
        order = await _live(fake).close_position("NVDA")
        assert order.status == "rejected" and "no long position" in order.reason
        assert fake.posts == []

    async def test_existing_short_is_not_touched(self) -> None:
        fake = FakeAlpaca()
        fake.positions["TSLA"] = _pos("TSLA", "-2", 10, 10, side="short")
        assert not (await _live(fake).close_position("TSLA")).ok
        assert fake.posts == []

    async def test_position_re_read_before_sell(self) -> None:
        fake = FakeAlpaca()
        fake.positions["AMD"] = _pos("AMD", "1", 10, 10)
        ex = _live(fake)
        await ex.refresh()
        fake.positions["AMD"] = _pos("AMD", "0.25", 2.5, 2.5)  # sold some elsewhere
        await ex.close_position("AMD")
        assert fake.posts[0]["qty"] == "0.25"

    async def test_dry_run_cannot_oversell(self) -> None:
        ex = _dry()
        await ex.buy("AMD", 10, price=100.0)
        order = await ex.sell("AMD", qty=5.0, price=100.0)
        assert order.qty == pytest.approx(0.1)
        assert not (await ex.sell("AMD", price=100.0)).ok  # nothing left

    @pytest.mark.parametrize("qty", [0, -1])
    async def test_non_positive_qty(self, qty: float) -> None:
        assert not (await _dry().sell("AMD", qty=qty)).ok


class TestLive:
    async def test_buy_sends_notional_market_order(self) -> None:
        fake = FakeAlpaca()
        ex = _live(fake)
        order = await ex.buy("nvda", 12.345)
        assert order.ok and order.id == "ord-1" and not order.dry_run
        post = next(r for r in fake.requests if r.method == "POST")
        assert str(post.url).startswith(PAPER_URL)
        assert post.headers["APCA-API-KEY-ID"] == "key"
        assert fake.posts == [
            {
                "symbol": "NVDA",
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "notional": "12.35",
            }
        ]
        assert ex.deployed_capital == pytest.approx(12.35)

    async def test_live_url_when_not_paper(self) -> None:
        fake = FakeAlpaca()
        await _live(fake, paper=False).buy("NVDA", 5)
        assert fake.requests and all(str(r.url).startswith(LIVE_URL) for r in fake.requests)

    async def test_market_closed_skips_order(self) -> None:
        fake = FakeAlpaca(market_open=False)
        order = await _live(fake).buy("NVDA", 5)
        assert order.status == "skipped" and order.reason == "market closed"
        assert fake.posts == []

    async def test_refresh_replaces_in_flight_with_position(self) -> None:
        fake = FakeAlpaca()
        ex = _live(fake)
        await ex.buy("NVDA", 10)
        fake.positions["NVDA"] = _pos("NVDA", "0.1", 10, 11)
        await ex.refresh()
        assert ex.deployed_capital == pytest.approx(10.0)
        assert ex.get_position("NVDA").market_value == 11.0

    async def test_account(self) -> None:
        acct = await _live(FakeAlpaca()).get_account()
        assert acct["equity"] == 1000.0 and acct["non_marginable_buying_power"] == 1000.0

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
            ("DELETE", "/v2/positions/NVDA"),  # liquidation endpoint not used
            ("DELETE", "/v2/positions"),
            ("GET", "/v1/accounts/abc/recipient_banks"),
            ("POST", "/v2/wallets/transfers"),
        ],
    )
    async def test_forbidden(self, method: str, path: str) -> None:
        fake = FakeAlpaca()
        with pytest.raises(ForbiddenEndpointError):
            await _live(fake)._request(method, path)
        assert fake.requests == []  # refused before any network call

    @pytest.mark.parametrize(
        "body",
        [
            {"symbol": "NVDA", "side": "buy", "type": "market", "time_in_force": "day",
             "notional": "5", "client_extra": 1},
            {"symbol": "NVDA", "side": "sell", "type": "market", "time_in_force": "day",
             "notional": "5"},  # notional sell
            {"symbol": "NVDA", "side": "sell", "type": "market", "time_in_force": "day",
             "qty": "-1"},
            {"symbol": "NVDA", "side": "buy", "type": "limit", "time_in_force": "day",
             "notional": "5"},
            {"symbol": "NVDA", "side": "buy", "type": "market", "time_in_force": "gtc",
             "notional": "5"},
            {"symbol": "NVDA", "side": "buy", "type": "market", "time_in_force": "day",
             "qty": "1"},  # buys must be dollar-based
            {"symbol": "NVDA", "side": "sell_short", "type": "market", "time_in_force": "day",
             "qty": "1"},
        ],
    )  # fmt: skip
    async def test_order_body_restricted(self, body: dict) -> None:
        fake = FakeAlpaca()
        with pytest.raises(ForbiddenEndpointError):
            await _live(fake)._request("POST", "/v2/orders", json=body)
        assert fake.requests == []

    async def test_ticker_containing_blocked_word_is_fine(self) -> None:
        fake = FakeAlpaca()
        fake.positions["ACHC"] = _pos("ACHC", "1", 1, 1)
        await _live(fake)._request("GET", "/v2/positions/ACHC")  # "ach" inside a ticker
        assert fake.requests[-1].url.path == "/v2/positions/ACHC"


def test_regular_session_hours() -> None:
    assert _is_regular_session(datetime(2026, 11, 9, 15, 0, tzinfo=UTC))  # Mon 10:00 ET
    assert not _is_regular_session(datetime(2026, 11, 9, 13, 0, tzinfo=UTC))  # 08:00 ET
    assert not _is_regular_session(datetime(2026, 11, 8, 15, 0, tzinfo=UTC))  # Sunday


def test_reexported_from_trading_module() -> None:
    from moneyclaw.execution import trading

    assert trading.AlpacaExecutor is AlpacaExecutor
