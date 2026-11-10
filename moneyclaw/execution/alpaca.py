"""Alpaca stock executor — dollar-based (fractional) market orders for US stocks/ETFs.

Safety properties, enforced here regardless of what calls this class:

* **Trading and read-only account access only.** Every HTTP request goes through
  ``_request``, which only permits an explicit allowlist of Trading API endpoints
  (orders, positions, account read, clock, asset lookup). Anything touching funding,
  ACH, bank relationships, transfers, journals or wires is refused before any
  network call. Only the two official Trading API hosts can be used.
* **Hard budget caps.** ``buy`` refuses an order that is above the per-trade limit or
  would push total deployed capital (cost basis of open positions plus orders
  still filling) above ``max_portfolio_budget`` — even if the account holds more.
  Limits are read live from the shared ``RiskSettings``, so Telegram changes apply
  immediately.
* **Cash only, never margin.** Before every buy the account is re-read and the order
  is refused unless ``non_marginable_buying_power`` (settled cash, excluding margin)
  is strictly greater than the order amount. If the account can't be read, or
  trading is blocked, the buy is refused (fail closed).
* **No short selling.** Sells are quantity-based only. The held quantity is
  re-fetched right before each sell and the order never exceeds it; anything that is
  not a long position is refused. Notional sells and position liquidation endpoints
  are not used.
* **dry_run by default.** In dry-run mode nothing is sent to Alpaca; fills are
  simulated at the caller's reference price against a simulated cash balance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
import structlog

from moneyclaw.config.settings import AlpacaSettings, RiskSettings

log = structlog.get_logger()

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
MIN_NOTIONAL = 1.0  # Alpaca's minimum for notional (fractional) orders

_NY = ZoneInfo("America/New_York")
_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")

# (method, path regex) — the ONLY requests this client may make.
_ALLOWED_ENDPOINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("GET", re.compile(r"^/v2/account$")),
    ("GET", re.compile(r"^/v2/clock$")),
    ("GET", re.compile(r"^/v2/assets/[A-Z.]{1,7}$")),
    ("GET", re.compile(r"^/v2/positions$")),
    ("GET", re.compile(r"^/v2/positions/[A-Z.]{1,7}$")),
    ("GET", re.compile(r"^/v2/orders$")),
    ("GET", re.compile(r"^/v2/orders/[A-Za-z0-9-]{1,64}$")),
    ("POST", re.compile(r"^/v2/orders$")),
    ("DELETE", re.compile(r"^/v2/orders/[A-Za-z0-9-]{1,64}$")),
)
# Belt and braces: refuse anything mentioning money movement, even if it matched above.
_FORBIDDEN_WORDS = ("transfer", "ach", "bank", "fund", "journal", "wire", "relationship")
_ALLOWED_ORDER_KEYS = {"symbol", "side", "type", "time_in_force", "notional", "qty"}
_QTY_STEP = Decimal("0.000000001")  # Alpaca supports up to 9 decimal places


def _validate_order(body: dict[str, Any]) -> None:
    """Only plain day market orders: buys by notional ($), sells by exact qty."""
    extra = set(body) - _ALLOWED_ORDER_KEYS
    if extra:
        raise ForbiddenEndpointError(f"Order fields not allowed: {sorted(extra)}")
    if body.get("type") != "market" or body.get("time_in_force") != "day":
        raise ForbiddenEndpointError("Only day market orders are allowed")
    side = body.get("side")
    if side == "buy":
        if "notional" not in body or "qty" in body:
            raise ForbiddenEndpointError("Buys must be dollar-based (notional only)")
    elif side == "sell":
        if "qty" not in body or "notional" in body:
            raise ForbiddenEndpointError("Sells must specify an exact share qty")
        try:
            if Decimal(str(body["qty"])) <= 0:
                raise ForbiddenEndpointError("Sell qty must be positive")
        except InvalidOperation as e:
            raise ForbiddenEndpointError("Sell qty must be a number") from e
    else:
        raise ForbiddenEndpointError(f"Unknown order side {side!r}")


class ForbiddenEndpointError(PermissionError):
    """Raised when code tries to call an Alpaca endpoint outside the trading allowlist."""


class InsufficientCashError(RuntimeError):
    """Raised when settled cash can't be confirmed for a buy."""


@dataclass
class StockOrder:
    id: str
    symbol: str
    side: str  # "buy" | "sell"
    notional: float  # USD requested (buy) or estimated proceeds (sell)
    status: str  # filled | accepted | rejected | skipped | failed ...
    dry_run: bool
    qty: float = 0.0
    filled_avg_price: float | None = None
    realized_pl: float = 0.0  # sells only
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def ok(self) -> bool:
        return self.status not in {"rejected", "skipped", "failed", "canceled", "expired"}


@dataclass
class StockPosition:
    symbol: str
    qty: float
    avg_entry_price: float
    cost_basis: float
    market_value: float
    unrealized_pl: float
    side: str = "long"

    @property
    def unrealized_plpc(self) -> float:
        return self.unrealized_pl / self.cost_basis if self.cost_basis else 0.0


def _is_regular_session(now: datetime | None = None) -> bool:
    """Approximate NYSE regular hours (Mon–Fri 9:30–16:00 ET, holidays ignored)."""
    ny = (now or datetime.now(UTC)).astimezone(_NY)
    return ny.weekday() < 5 and time(9, 30) <= ny.time() < time(16, 0)


class AlpacaExecutor:
    """Place dollar-based stock orders on Alpaca (or simulate them in dry-run)."""

    def __init__(
        self,
        settings: AlpacaSettings,
        risk_settings: RiskSettings,
        dry_run: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._risk = risk_settings
        self._dry_run = dry_run
        self._base_url = PAPER_URL if settings.paper else LIVE_URL
        self._transport = transport  # injectable for tests
        self._positions: dict[str, StockPosition] = {}
        self._in_flight: dict[str, float] = {}  # live buys not yet visible as positions
        self._orders: list[StockOrder] = []
        self._last_price: dict[str, float] = {}
        self._sim_cash = float(settings.dry_run_cash)  # dry-run settled cash

        if not dry_run and not self.has_credentials:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY are required when not in dry-run"
            )

    # ------------------------------------------------------------------ props

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def paper(self) -> bool:
        return self._settings.paper

    @property
    def has_credentials(self) -> bool:
        return bool(self._settings.api_key and self._settings.secret_key)

    @property
    def order_history(self) -> list[StockOrder]:
        return list(self._orders)

    @property
    def deployed_capital(self) -> float:
        """USD at risk: cost basis of open positions plus live buys still filling."""
        return sum(p.cost_basis for p in self._positions.values()) + sum(self._in_flight.values())

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self._risk.max_portfolio_budget - self.deployed_capital)

    def max_buy_amount(self) -> float:
        """Largest buy currently allowed by the per-trade and portfolio caps."""
        return max(0.0, min(self._risk.max_trade_amount, self.remaining_budget))

    def mark_price(self, symbol: str, price: float) -> None:
        """Feed a recent price (used to value simulated positions in dry-run)."""
        if price > 0:
            self._last_price[symbol] = price
            pos = self._positions.get(symbol)
            if pos and self._dry_run:
                pos.market_value = pos.qty * price
                pos.unrealized_pl = pos.market_value - pos.cost_basis

    # ------------------------------------------------------------------ http

    async def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None) -> Any:
        method = method.upper()
        # Ticker segments (all caps, e.g. ACHC) are exempt from the word check.
        lowered = "/".join(s for s in path.split("/") if not s.isupper()).lower()
        if any(w in lowered for w in _FORBIDDEN_WORDS) or not any(
            m == method and rx.match(path) for m, rx in _ALLOWED_ENDPOINTS
        ):
            log.error("alpaca.forbidden_endpoint", method=method, path=path)
            raise ForbiddenEndpointError(f"Alpaca endpoint not allowed: {method} {path}")
        if method == "POST":
            _validate_order(json or {})
        if not self.has_credentials:
            raise RuntimeError("Alpaca credentials are not configured")

        headers = {
            "APCA-API-KEY-ID": self._settings.api_key,
            "APCA-API-SECRET-KEY": self._settings.secret_key,
        }
        async with httpx.AsyncClient(
            base_url=self._base_url, headers=headers, timeout=15, transport=self._transport
        ) as client:
            resp = await client.request(method, path, json=json)
            resp.raise_for_status()
            return resp.json() if resp.content else None

    # ------------------------------------------------------------------ reads

    async def is_market_open(self) -> bool:
        if self.has_credentials:
            try:
                clock = await self._request("GET", "/v2/clock")
                return bool(clock.get("is_open"))
            except Exception:
                log.warning("alpaca.clock_failed", exc_info=True)
        return _is_regular_session()

    async def get_account(self) -> dict[str, float]:
        """Account summary. ``non_marginable_buying_power`` = settled cash usable for buys."""
        if self._dry_run:
            value = sum(p.market_value for p in self._positions.values())
            return {
                "equity": self._sim_cash + value,
                "cash": self._sim_cash,
                "buying_power": self._sim_cash,
                "non_marginable_buying_power": self._sim_cash,
            }
        a = await self._request("GET", "/v2/account")
        keys = ("equity", "cash", "buying_power", "non_marginable_buying_power")
        return {k: float(a.get(k) or 0) for k in keys}

    async def available_cash(self) -> float:
        """Settled, non-margin cash available for buys. Raises if it can't be confirmed."""
        if self._dry_run:
            return self._sim_cash
        try:
            a = await self._request("GET", "/v2/account")
        except Exception as e:
            raise InsufficientCashError(f"could not read account: {e}") from e
        for flag in ("trading_blocked", "account_blocked", "trade_suspended_by_user"):
            if a.get(flag):
                raise InsufficientCashError(f"account {flag.replace('_', ' ')}")
        raw = a.get("non_marginable_buying_power")
        if raw is None:
            raise InsufficientCashError("account has no non_marginable_buying_power")
        return float(raw)

    async def refresh(self) -> None:
        """Re-sync positions from Alpaca (live/paper). No-op in dry-run."""
        if self._dry_run:
            return
        try:
            rows = await self._request("GET", "/v2/positions")
        except Exception:
            log.warning("alpaca.refresh_failed", exc_info=True)
            return
        positions = {}
        for r in rows or []:
            p = StockPosition(
                symbol=r["symbol"],
                qty=float(r.get("qty") or 0),
                avg_entry_price=float(r.get("avg_entry_price") or 0),
                cost_basis=float(r.get("cost_basis") or 0),
                market_value=float(r.get("market_value") or 0),
                unrealized_pl=float(r.get("unrealized_pl") or 0),
                side=str(r.get("side") or "long"),
            )
            positions[p.symbol] = p
        # Buys that now show up as positions are no longer "in flight"
        for sym in list(self._in_flight):
            if sym in positions:
                del self._in_flight[sym]
        self._positions = positions

    async def get_positions(self) -> list[StockPosition]:
        await self.refresh()
        return list(self._positions.values())

    def get_position(self, symbol: str) -> StockPosition | None:
        return self._positions.get(symbol)

    # ------------------------------------------------------------------ orders

    def _record(self, order: StockOrder) -> StockOrder:
        self._orders.append(order)
        log.info(
            "alpaca.order",
            symbol=order.symbol,
            side=order.side,
            notional=order.notional,
            status=order.status,
            dry_run=order.dry_run,
            reason=order.reason,
        )
        return order

    def _reject(self, symbol: str, side: str, notional: float, reason: str) -> StockOrder:
        return self._record(
            StockOrder(
                id=f"rej_{uuid4().hex[:8]}",
                symbol=symbol,
                side=side,
                notional=notional,
                status="rejected",
                dry_run=self._dry_run,
                reason=reason,
            )
        )

    async def buy(self, symbol: str, notional: float, price: float | None = None) -> StockOrder:
        """Dollar-based market buy. Enforces per-trade/total-budget caps and cash-only."""
        symbol = symbol.upper()
        notional = round(float(notional), 2)
        if not _SYMBOL_RE.match(symbol):
            return self._reject(symbol, "buy", notional, "invalid symbol")
        if notional < MIN_NOTIONAL:
            return self._reject(symbol, "buy", notional, f"below ${MIN_NOTIONAL:.2f} minimum")
        if notional > self._risk.max_trade_amount + 1e-9:
            return self._reject(
                symbol, "buy", notional, f"exceeds max trade ${self._risk.max_trade_amount:.2f}"
            )
        if self.deployed_capital + notional > self._risk.max_portfolio_budget + 1e-9:
            return self._reject(
                symbol,
                "buy",
                notional,
                f"would exceed portfolio budget ${self._risk.max_portfolio_budget:.2f}"
                f" (deployed ${self.deployed_capital:.2f})",
            )
        price = price or self._last_price.get(symbol)

        if self._dry_run:
            if not price:
                return self._reject(symbol, "buy", notional, "no reference price for simulation")
            if not self._sim_cash > notional:
                return self._reject(
                    symbol, "buy", notional, f"insufficient settled cash (${self._sim_cash:.2f})"
                )
            self._sim_cash -= notional
            qty = notional / price
            pos = self._positions.get(symbol)
            if pos:
                pos.qty += qty
                pos.cost_basis += notional
                pos.avg_entry_price = pos.cost_basis / pos.qty
                pos.market_value = pos.qty * price
                pos.unrealized_pl = pos.market_value - pos.cost_basis
            else:
                self._positions[symbol] = StockPosition(symbol, qty, price, notional, notional, 0.0)
            self._last_price[symbol] = price
            return self._record(
                StockOrder(
                    id=f"dry_{uuid4().hex[:8]}",
                    symbol=symbol,
                    side="buy",
                    notional=notional,
                    status="filled",
                    dry_run=True,
                    qty=qty,
                    filled_avg_price=price,
                )
            )

        if not await self.is_market_open():
            return self._record(
                StockOrder(
                    id=f"skip_{uuid4().hex[:8]}",
                    symbol=symbol,
                    side="buy",
                    notional=notional,
                    status="skipped",
                    dry_run=False,
                    reason="market closed",
                )
            )
        # Cash-only guard: settled, non-margin cash must exceed the order amount.
        try:
            cash = await self.available_cash()
        except InsufficientCashError as e:
            return self._reject(symbol, "buy", notional, f"cash check failed: {e}")
        if not cash > notional:
            return self._reject(
                symbol,
                "buy",
                notional,
                f"insufficient settled cash: non-marginable buying power ${cash:.2f}"
                f" is not more than ${notional:.2f}",
            )
        try:
            r = await self._request(
                "POST",
                "/v2/orders",
                json={
                    "symbol": symbol,
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day",  # required for fractional orders
                    "notional": f"{notional:.2f}",
                },
            )
        except Exception as e:
            log.exception("alpaca.buy_failed", symbol=symbol)
            return self._reject(symbol, "buy", notional, f"order failed: {e}")
        self._in_flight[symbol] = self._in_flight.get(symbol, 0.0) + notional
        return self._record(
            StockOrder(
                id=str(r.get("id")),
                symbol=symbol,
                side="buy",
                notional=notional,
                status=str(r.get("status", "accepted")),
                dry_run=False,
                qty=float(r.get("filled_qty") or 0),
                filled_avg_price=float(r["filled_avg_price"])
                if r.get("filled_avg_price")
                else None,
            )
        )

    async def close_position(self, symbol: str, price: float | None = None) -> StockOrder:
        """Sell exactly the shares currently held in ``symbol`` (never more)."""
        return await self.sell(symbol, qty=None, price=price)

    async def sell(
        self, symbol: str, qty: float | None = None, price: float | None = None
    ) -> StockOrder:
        """Market-sell up to ``qty`` shares (default: the whole position).

        Never sells more than is currently held, so it can't open a short position.
        """
        symbol = symbol.upper()
        if not _SYMBOL_RE.match(symbol):
            return self._reject(symbol, "sell", 0.0, "invalid symbol")
        if qty is not None and not qty > 0:
            return self._reject(symbol, "sell", 0.0, "sell qty must be positive")

        if self._dry_run:
            pos = self._positions.get(symbol)
            if not pos or not pos.qty > 0:
                return self._reject(symbol, "sell", 0.0, "no long position to sell")
            sell_qty = pos.qty if qty is None else min(qty, pos.qty)
            price = price or self._last_price.get(symbol) or pos.avg_entry_price
            fraction = sell_qty / pos.qty
            cost = pos.cost_basis * fraction
            proceeds = sell_qty * price
            if fraction >= 1 - 1e-12:
                del self._positions[symbol]
            else:
                pos.qty -= sell_qty
                pos.cost_basis -= cost
                pos.market_value = pos.qty * price
                pos.unrealized_pl = pos.market_value - pos.cost_basis
            self._sim_cash += proceeds
            return self._record(
                StockOrder(
                    id=f"dry_{uuid4().hex[:8]}",
                    symbol=symbol,
                    side="sell",
                    notional=round(proceeds, 2),
                    status="filled",
                    dry_run=True,
                    qty=sell_qty,
                    filled_avg_price=price,
                    realized_pl=round(proceeds - cost, 2),
                )
            )

        # Live: re-read this position right now, straight from Alpaca.
        try:
            r = await self._request("GET", f"/v2/positions/{symbol}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                self._positions.pop(symbol, None)
                return self._reject(symbol, "sell", 0.0, "no long position to sell")
            return self._reject(symbol, "sell", 0.0, f"could not read position: {e}")
        except Exception as e:
            return self._reject(symbol, "sell", 0.0, f"could not read position: {e}")
        try:
            held = Decimal(str(r.get("qty") or "0"))
        except InvalidOperation:
            return self._reject(symbol, "sell", 0.0, "unreadable position qty")
        if str(r.get("side") or "long") != "long" or held <= 0:
            return self._reject(symbol, "sell", 0.0, "no long position to sell")

        sell_qty = held
        if qty is not None:
            sell_qty = min(held, Decimal(str(qty)).quantize(_QTY_STEP, rounding=ROUND_DOWN))
        if sell_qty <= 0:
            return self._reject(symbol, "sell", 0.0, "sell qty rounds to zero")
        fraction = float(sell_qty / held)
        market_value = float(r.get("market_value") or 0) * fraction
        unrealized = float(r.get("unrealized_pl") or 0) * fraction

        if not await self.is_market_open():
            return self._record(
                StockOrder(
                    id=f"skip_{uuid4().hex[:8]}",
                    symbol=symbol,
                    side="sell",
                    notional=round(market_value, 2),
                    status="skipped",
                    dry_run=False,
                    reason="market closed",
                )
            )
        try:
            o = await self._request(
                "POST",
                "/v2/orders",
                json={
                    "symbol": symbol,
                    "side": "sell",
                    "type": "market",
                    "time_in_force": "day",
                    "qty": format(sell_qty.normalize(), "f"),
                },
            )
        except Exception as e:
            log.exception("alpaca.sell_failed", symbol=symbol)
            return self._reject(symbol, "sell", market_value, f"order failed: {e}")
        if sell_qty == held:
            self._positions.pop(symbol, None)
        return self._record(
            StockOrder(
                id=str((o or {}).get("id", "")),
                symbol=symbol,
                side="sell",
                notional=round(market_value, 2),
                status=str((o or {}).get("status", "accepted")),
                dry_run=False,
                qty=float(sell_qty),
                realized_pl=round(unrealized, 2),  # P&L at time of exit (approx.)
            )
        )
