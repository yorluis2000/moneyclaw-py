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
* **dry_run by default.** In dry-run mode nothing is sent to Alpaca; fills are
  simulated at the caller's reference price and tracked in memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
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
    ("DELETE", re.compile(r"^/v2/positions/[A-Z.]{1,7}$")),
    ("GET", re.compile(r"^/v2/orders$")),
    ("GET", re.compile(r"^/v2/orders/[A-Za-z0-9-]{1,64}$")),
    ("POST", re.compile(r"^/v2/orders$")),
    ("DELETE", re.compile(r"^/v2/orders/[A-Za-z0-9-]{1,64}$")),
)
# Belt and braces: refuse anything mentioning money movement, even if it matched above.
_FORBIDDEN_WORDS = ("transfer", "ach", "bank", "fund", "journal", "wire", "relationship")
_ALLOWED_ORDER_KEYS = {"symbol", "side", "type", "time_in_force", "notional", "qty"}


class ForbiddenEndpointError(PermissionError):
    """Raised when code tries to call an Alpaca endpoint outside the trading allowlist."""


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
        if json is not None and method == "POST":
            extra = set(json) - _ALLOWED_ORDER_KEYS
            if extra:
                raise ForbiddenEndpointError(f"Order fields not allowed: {sorted(extra)}")
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
        if self._dry_run:
            return {
                "equity": self._risk.max_portfolio_budget
                + sum(p.unrealized_pl for p in self._positions.values()),
                "cash": self.remaining_budget,
                "buying_power": self.remaining_budget,
            }
        a = await self._request("GET", "/v2/account")
        return {k: float(a.get(k) or 0) for k in ("equity", "cash", "buying_power")}

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
        """Dollar-based market buy. Enforces per-trade and total-budget caps."""
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
        """Sell the entire position in ``symbol`` at market."""
        symbol = symbol.upper()
        if not _SYMBOL_RE.match(symbol):
            return self._reject(symbol, "sell", 0.0, "invalid symbol")

        if self._dry_run:
            pos = self._positions.get(symbol)
            if not pos:
                return self._reject(symbol, "sell", 0.0, "no open position")
            price = price or self._last_price.get(symbol) or pos.avg_entry_price
            proceeds = pos.qty * price
            realized = proceeds - pos.cost_basis
            del self._positions[symbol]
            return self._record(
                StockOrder(
                    id=f"dry_{uuid4().hex[:8]}",
                    symbol=symbol,
                    side="sell",
                    notional=round(proceeds, 2),
                    status="filled",
                    dry_run=True,
                    qty=pos.qty,
                    filled_avg_price=price,
                    realized_pl=round(realized, 2),
                )
            )

        await self.refresh()
        pos = self._positions.get(symbol)
        if not pos:
            return self._reject(symbol, "sell", 0.0, "no open position")
        if not await self.is_market_open():
            return self._record(
                StockOrder(
                    id=f"skip_{uuid4().hex[:8]}",
                    symbol=symbol,
                    side="sell",
                    notional=pos.market_value,
                    status="skipped",
                    dry_run=False,
                    reason="market closed",
                )
            )
        try:
            r = await self._request("DELETE", f"/v2/positions/{symbol}")
        except Exception as e:
            log.exception("alpaca.sell_failed", symbol=symbol)
            return self._reject(symbol, "sell", pos.market_value, f"order failed: {e}")
        del self._positions[symbol]
        return self._record(
            StockOrder(
                id=str((r or {}).get("id", "")),
                symbol=symbol,
                side="sell",
                notional=round(pos.market_value, 2),
                status=str((r or {}).get("status", "accepted")),
                dry_run=False,
                qty=pos.qty,
                realized_pl=round(pos.unrealized_pl, 2),  # P&L at time of exit (approx.)
            )
        )
