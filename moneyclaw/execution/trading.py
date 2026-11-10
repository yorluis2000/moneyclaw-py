"""Trading execution — crypto via ccxt (TradeExecutor) and US stocks via Alpaca.

Stock trading lives in :mod:`moneyclaw.execution.alpaca` and is re-exported here, so
``from moneyclaw.execution.trading import AlpacaExecutor`` works alongside the ccxt
executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

log = structlog.get_logger()


@dataclass
class Order:
    """Represents a placed (or simulated) order."""

    id: str
    exchange: str
    symbol: str
    side: str  # "buy" or "sell"
    type: str  # "market" or "limit"
    amount: float
    price: float | None = None
    filled: float = 0.0
    status: str = "open"  # open, closed, canceled
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    dry_run: bool = False


class ExchangeManager:
    """Manage connections to crypto exchanges via ccxt."""

    def __init__(self) -> None:
        self._exchanges: dict[str, object] = {}

    def connect(self, exchange_id: str, api_key: str = "", secret: str = "") -> None:
        """Connect to an exchange. In dry_run mode, keys can be empty."""
        import ccxt

        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {exchange_id}")
        self._exchanges[exchange_id] = exchange_class(
            {"apiKey": api_key, "secret": secret, "enableRateLimit": True}
        )
        log.info("exchange.connected", exchange=exchange_id)

    def get(self, exchange_id: str) -> object:
        """Get a connected exchange instance."""
        if exchange_id not in self._exchanges:
            raise ValueError(f"Exchange not connected: {exchange_id}")
        return self._exchanges[exchange_id]

    async def get_balance(self, exchange_id: str) -> dict:
        """Fetch account balance from exchange."""
        ex = self.get(exchange_id)
        return await ex.fetch_balance()  # type: ignore[union-attr]

    async def get_total_value(self, exchange_id: str) -> float:
        """Calculate total portfolio value in USD."""
        try:
            balance = await self.get_balance(exchange_id)
            total = balance.get("total", {})
            # Sum up all non-zero balances
            # In real implementation, we'd convert each to USD using current prices
            usd_value = total.get("USDT", 0) + total.get("USD", 0) + total.get("BUSD", 0)

            # For crypto assets, we'd need price conversion
            # For now, return a simplified calculation
            total_value = usd_value
            for asset, amount in total.items():
                if asset not in ["USDT", "USD", "BUSD"] and amount > 0:
                    # In dry_run mode, use placeholder values
                    # In live mode, fetch prices and calculate
                    total_value += amount * 0  # Would multiply by price

            return total_value
        except Exception as e:
            log.warning("exchange.balance_error", exchange=exchange_id, error=str(e))
            return 0.0

    @property
    def connected(self) -> list[str]:
        return list(self._exchanges)


class TradeExecutor:
    """Execute trades through exchanges or simulate in dry_run mode.

    IMPORTANT: dry_run=True by default. No real orders without explicit opt-in.
    """

    def __init__(
        self,
        exchange_manager: ExchangeManager,
        dry_run: bool = True,
    ) -> None:
        self._em = exchange_manager
        self._dry_run = dry_run
        self._order_counter = 0
        self._orders: list[Order] = []

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def exchange_manager(self) -> ExchangeManager:
        return self._em

    def _next_id(self) -> str:
        self._order_counter += 1
        return f"dry_{self._order_counter}" if self._dry_run else f"ord_{self._order_counter}"

    async def market_buy(self, exchange_id: str, symbol: str, amount: float) -> Order:
        """Place a market buy order (or simulate)."""
        return await self._place("buy", "market", exchange_id, symbol, amount)

    async def market_sell(self, exchange_id: str, symbol: str, amount: float) -> Order:
        """Place a market sell order (or simulate)."""
        return await self._place("sell", "market", exchange_id, symbol, amount)

    async def limit_buy(self, exchange_id: str, symbol: str, amount: float, price: float) -> Order:
        """Place a limit buy order (or simulate)."""
        return await self._place("buy", "limit", exchange_id, symbol, amount, price)

    async def limit_sell(self, exchange_id: str, symbol: str, amount: float, price: float) -> Order:
        """Place a limit sell order (or simulate)."""
        return await self._place("sell", "limit", exchange_id, symbol, amount, price)

    async def _place(
        self,
        side: str,
        order_type: str,
        exchange_id: str,
        symbol: str,
        amount: float,
        price: float | None = None,
    ) -> Order:
        order = Order(
            id=self._next_id(),
            exchange=exchange_id,
            symbol=symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=price,
            dry_run=self._dry_run,
        )

        if self._dry_run:
            order.filled = amount
            order.status = "closed"
            log.info(
                "trade.dry_run",
                side=side,
                type=order_type,
                symbol=symbol,
                amount=amount,
                price=price,
            )
        else:
            ex = self._em.get(exchange_id)
            try:
                if order_type == "market":
                    result = await ex.create_order(symbol, "market", side, amount)  # type: ignore[union-attr]
                else:
                    result = await ex.create_order(symbol, "limit", side, amount, price)  # type: ignore[union-attr]
                order.id = str(result.get("id", order.id))
                order.filled = float(result.get("filled", 0))
                order.status = result.get("status", "open")
                log.info("trade.placed", order_id=order.id, status=order.status)
            except Exception:
                order.status = "failed"
                log.exception("trade.error", symbol=symbol, side=side)

        self._orders.append(order)
        return order

    async def get_open_orders(self, exchange_id: str) -> list[Order]:
        """Get open orders — from local list in dry_run, from exchange otherwise."""
        if self._dry_run:
            return [o for o in self._orders if o.exchange == exchange_id and o.status == "open"]
        ex = self._em.get(exchange_id)
        open_orders = await ex.fetch_open_orders()  # type: ignore[union-attr]
        return [
            Order(
                id=str(o["id"]),
                exchange=exchange_id,
                symbol=o.get("symbol", ""),
                side=o.get("side", ""),
                type=o.get("type", ""),
                amount=float(o.get("amount", 0)),
                price=float(o["price"]) if o.get("price") else None,
                filled=float(o.get("filled", 0)),
                status=o.get("status", "open"),
            )
            for o in open_orders
        ]

    async def cancel_order(self, exchange_id: str, order_id: str) -> bool:
        """Cancel an order."""
        if self._dry_run:
            for o in self._orders:
                if o.id == order_id and o.status == "open":
                    o.status = "canceled"
                    return True
            return False
        try:
            ex = self._em.get(exchange_id)
            await ex.cancel_order(order_id)  # type: ignore[union-attr]
            return True
        except Exception:
            log.exception("trade.cancel_error", order_id=order_id)
            return False

    @property
    def order_history(self) -> list[Order]:
        return list(self._orders)


# --- US stocks/ETFs (Alpaca): cash-only, no shorting, budget-capped ---------------
from moneyclaw.execution.alpaca import (  # noqa: E402
    AlpacaExecutor,
    ForbiddenEndpointError,
    InsufficientCashError,
    StockOrder,
    StockPosition,
)

__all__ = [
    "AlpacaExecutor",
    "ExchangeManager",
    "ForbiddenEndpointError",
    "InsufficientCashError",
    "Order",
    "StockOrder",
    "StockPosition",
    "TradeExecutor",
]
