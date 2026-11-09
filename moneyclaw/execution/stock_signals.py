"""Turn a stock buy/sell Opportunity into an Alpaca order and a strategy Result.

Opportunity convention used by stock strategies:
    data["symbol"]  ticker, data["side"] "buy" | "sell", data["price"] reference price
    money_involved  USD notional for buys (sells close the whole position)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moneyclaw.plugins.base import Opportunity, Result

if TYPE_CHECKING:
    from moneyclaw.execution.alpaca import AlpacaExecutor


async def execute_stock_signal(executor: AlpacaExecutor, opp: Opportunity) -> Result:
    symbol = str(opp.data.get("symbol", ""))
    side = opp.data.get("side")
    price = opp.data.get("price")

    if side == "buy":
        order = await executor.buy(symbol, opp.money_involved, price=price)
    elif side == "sell":
        order = await executor.close_position(symbol, price=price)
    else:
        return Result(success=False, details={"error": f"unknown side {side!r}"})

    verb = "Bought" if side == "buy" else "Sold"
    if order.ok:
        message = f"{verb} {symbol} ${order.notional:.2f}"
        if order.filled_avg_price:
            message += f" @ {order.filled_avg_price:.2f}"
        if side == "sell":
            message += f" (P&L {order.realized_pl:+.2f})"
        if order.status not in {"filled"}:
            message += f" [{order.status}]"
    else:
        message = f"{side.upper()} {symbol} not placed: {order.reason or order.status}"

    return Result(
        success=order.ok,
        profit_loss=order.realized_pl if order.ok else 0.0,
        details={
            "order_id": order.id,
            "symbol": symbol,
            "side": side,
            "notional": order.notional,
            "qty": order.qty,
            "status": order.status,
            "reason": order.reason,
            "dry_run": order.dry_run,
            "message": message,
        },
    )
