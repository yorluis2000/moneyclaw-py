"""FastAPI web dashboard — HTMX-powered monitoring and control."""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

_STATIC_DIR = Path(__file__).parent / "static"

if TYPE_CHECKING:
    from moneyclaw.agent.brain import AgentBrain
    from moneyclaw.agent.memory import Memory
    from moneyclaw.agent.strategy_chat import StrategyChatInterface
    from moneyclaw.execution.risk import RiskManager
    from moneyclaw.execution.trading import TradeExecutor
    from moneyclaw.llm.router import LLMRouter
    from moneyclaw.plugins.registry import StrategyRegistry


class ChatRequest(BaseModel):
    """Request body for chat API."""
    message: str


class StrategyChatRequest(BaseModel):
    """Request body for strategy chat API."""
    message: str


class StrategyChatResponse(BaseModel):
    """Response body for strategy chat API."""
    message: str
    success: bool = True
    data: dict | None = None
    actions: list[str] | None = None


def create_app(
    brain: AgentBrain,
    memory: Memory,
    llm: LLMRouter,
    strategies: StrategyRegistry,
    risk: RiskManager,
    executor: TradeExecutor | None = None,
    strategy_chat: StrategyChatInterface | None = None,
) -> FastAPI:
    """Create the FastAPI app with all routes."""
    app = FastAPI(title="MoneyClaw", version="0.1.0")
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # --- Template Setup ---
    _TEMPLATE_DIR = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        dry_run = bool(risk.status().get("dry_run", True))
        return templates.TemplateResponse(request, "index.html", {"dry_run": dry_run})

    # --- API Endpoints (JSON) ---

    @app.get("/api/status")
    async def api_status() -> dict:
        return await brain.get_status()

    @app.get("/api/strategies")
    async def api_strategies() -> list[dict]:
        return strategies.status()

    @app.get("/api/history")
    async def api_history() -> list[dict]:
        return await memory.get_history()

    @app.get("/api/cost")
    async def api_cost() -> dict:
        summary = llm.cost_tracker.get_daily_summary()
        return {
            "today_cost": llm.cost_tracker.today_cost,
            "today_calls": llm.cost_tracker.today_calls,
            "total_cost": llm.cost_tracker.get_total_cost(),
            "over_budget": llm.cost_tracker.is_over_budget(),
        } | (
            {
                "by_layer": summary.cost_by_layer,
                "calls_by_layer": summary.calls_by_layer,
            }
            if summary
            else {}
        )

    @app.get("/api/risk")
    async def api_risk() -> dict:
        return risk.status()

    @app.get("/api/orders")
    async def api_orders() -> list[dict]:
        if not executor:
            return []
        return [
            {
                "id": o.id,
                "exchange": o.exchange,
                "symbol": o.symbol,
                "side": o.side,
                "type": o.type,
                "amount": o.amount,
                "price": o.price,
                "filled": o.filled,
                "status": o.status,
                "dry_run": o.dry_run,
                "timestamp": o.timestamp.isoformat(),
            }
            for o in executor.order_history
        ]

    @app.get("/api/pending")
    async def api_pending() -> list[dict]:
        return await memory.get_pending()

    @app.post("/api/pause")
    async def api_pause() -> dict:
        risk.pause()
        return {"status": "paused"}

    @app.post("/api/resume")
    async def api_resume() -> dict:
        risk.resume()
        return {"status": "resumed"}

    @app.post("/api/approve/{opp_id}")
    async def api_approve(opp_id: str) -> HTMLResponse:
        ok = await memory.approve(opp_id)
        if ok:
            return HTMLResponse('<span class="text-green-400">Approved</span>')
        return HTMLResponse('<span class="text-red-400">Not found</span>', status_code=404)

    @app.post("/api/reject/{opp_id}")
    async def api_reject(opp_id: str) -> HTMLResponse:
        ok = await memory.reject(opp_id)
        if ok:
            return HTMLResponse('<span class="text-gray-400">Rejected</span>')
        return HTMLResponse('<span class="text-red-400">Not found</span>', status_code=404)

    # --- Chat API ---

    @app.post("/api/chat")
    async def api_chat(payload: ChatRequest) -> dict:
        """AI chat interface with intelligent routing."""
        msg = payload.message.lower()

        # Always try AI strategy chat first if available
        if strategy_chat:
            response = await strategy_chat.handle_message(payload.message)
            
            # If AI understood the intent, return its response
            if response.success or response.data:
                return {
                    "response": response.message,
                    "success": response.success,
                    "data": response.data,
                    "actions": response.actions
                }
            
            # If AI didn't understand, fall through to default responses
        
        # Fallback: Simple command parsing
        if "status" in msg:
            return {"response": f"System is currently {'RUNNING' if brain.is_running else 'STOPPED'}. P&L: ${await memory.today_pnl():.2f}"}
        elif "list strategies" in msg:
            names = [s.name for s in strategies.active]
            return {"response": f"Active strategies: {', '.join(names)}"}
        elif "risk" in msg:
            r = risk.status()
            return {"response": f"Risk Level: LOW. Daily Loss: ${r['daily_loss']:.2f}"}
        elif "help" in msg or "帮助" in msg:
            return {"response": "我可以帮你：\n1. 创建/管理交易策略\n2. 查看系统状态\n3. 查看风险信息\n\n试试说：'创建一个定投BTC的策略' 或 '查看所有策略'"}

        return {"response": f"我不太理解 '{payload.message}'。输入 'help' 或 '帮助' 查看支持的命令。"}

    # --- Strategy Management API ---

    @app.post("/api/strategy/chat", response_model=StrategyChatResponse)
    async def api_strategy_chat(payload: StrategyChatRequest) -> StrategyChatResponse:
        """AI-powered strategy management chat interface."""
        if not strategy_chat:
            return StrategyChatResponse(
                message="AI strategy management is not enabled.",
                success=False
            )

        response = await strategy_chat.handle_message(payload.message)
        return StrategyChatResponse(
            message=response.message,
            success=response.success,
            data=response.data,
            actions=response.actions
        )

    @app.post("/api/strategy/confirm")
    async def api_strategy_confirm(payload: dict) -> dict:
        """Confirm saving a generated strategy."""
        if not strategy_chat:
            return {"message": "AI strategy management is not enabled.", "success": False}

        strategy_data = payload.get("strategy")
        if not strategy_data:
            return {"message": "No strategy data provided.", "success": False}

        try:
            from moneyclaw.agent.strategy_generator import GeneratedStrategy

            strategy = GeneratedStrategy(**strategy_data)
            response = await strategy_chat.confirm_save_strategy(strategy)
            return {"message": response.message, "success": response.success}
        except Exception as e:
            return {"message": f"Failed to save strategy: {e}", "success": False}

    @app.get("/api/strategy/templates")
    async def api_strategy_templates() -> dict:
        """Get available strategy templates."""
        templates = {
            "dca": {
                "name": "Dollar Cost Averaging (DCA)",
                "description": "定期定额投资策略",
                "example": "创建一个每天定投100美元BTC的策略"
            },
            "price_alert": {
                "name": "Price Alert",
                "description": "价格提醒策略",
                "example": "当ETH价格突破3000美元时提醒我"
            },
            "rebalance": {
                "name": "Smart Rebalance",
                "description": "智能再平衡策略",
                "example": "创建一个BTC和ETH 50/50再平衡策略"
            },
            "funding_arbitrage": {
                "name": "Funding Rate Arbitrage",
                "description": "资金费率套利策略",
                "example": "监控资金费率并进行套利"
            }
        }
        return {"templates": templates}

    @app.get("/api/strategies/{strategy_name}")
    async def api_strategy_detail(strategy_name: str) -> dict:
        """Get detailed information about a specific strategy."""
        strategy = strategies.get(strategy_name)
        if not strategy:
            return {"error": "Strategy not found", "success": False}

        # Get real execution history from memory
        history = await memory.get_strategy_history(strategy_name, limit=20)

        # Get real stats
        stats = await memory.get_strategy_stats(strategy_name)

        # Format history for frontend
        formatted_history = []
        for h in history:
            from datetime import datetime
            executed_at = h["executed_at"]
            if isinstance(executed_at, (int, float)):
                dt = datetime.fromtimestamp(executed_at)
            else:
                dt = datetime.fromisoformat(str(executed_at))

            profit = h.get("profit_loss", 0)
            formatted_history.append({
                "action": h.get("title", "Trade Execution"),
                "success": profit >= 0,
                "result": f"{'+' if profit >= 0 else ''}{profit:.2f}",
                "time": dt.strftime("%m-%d %H:%M"),
                "timestamp": executed_at,
                "details": h.get("details")
            })

        # Calculate actual ROI from history if available
        roi_estimate = strategy.estimate_roi()
        if stats["total_executions"] > 0 and stats["avg_pnl"] != 0:
            # Use actual average P&L as a component of ROI
            actual_roi = stats["avg_pnl"] * stats["total_executions"]
            # Blend estimated and actual
            roi_estimate = (roi_estimate + actual_roi) / 2

        return {
            "name": strategy.name,
            "description": strategy.description,
            "enabled": strategies.is_enabled(strategy_name),
            "risk_level": strategy.risk_level,
            "roi_estimate": roi_estimate,
            "executions": stats["total_executions"],
            "success_rate": f"{stats['success_rate']:.0f}%" if stats['total_executions'] > 0 else "N/A",
            "avg_pnl": stats["avg_pnl"],
            "recent_pnl_24h": stats["recent_pnl"],
            "recent_executions_24h": stats["recent_executions"],
            "history": formatted_history,
            "success": True
        }

    # --- Strategy Version Management API ---

    @app.get("/api/strategy/{strategy_name}/versions")
    async def api_strategy_versions(strategy_name: str) -> dict:
        """Get version history for a strategy."""
        from moneyclaw.agent.strategy_version import StrategyVersionManager

        version_manager = StrategyVersionManager()
        versions = version_manager.list_versions(strategy_name)
        stats = version_manager.get_strategy_stats(strategy_name)

        return {
            "strategy_name": strategy_name,
            "total_versions": len(versions),
            "stats": stats,
            "versions": [
                {
                    "version_id": v.version_id,
                    "created_at": v.created_at,
                    "author": v.author,
                    "change_summary": v.change_summary,
                    "code_hash": v.code_hash,
                    "tags": v.tags,
                }
                for v in versions[:20]  # 只返回最近20个版本
            ],
            "success": True
        }

    @app.post("/api/strategy/{strategy_name}/rollback")
    async def api_strategy_rollback(strategy_name: str, payload: dict) -> dict:
        """Rollback a strategy to a specific version."""
        from moneyclaw.agent.strategy_version import StrategyVersionManager

        version_id = payload.get("version_id")
        if not version_id:
            return {"error": "version_id is required", "success": False}

        version_manager = StrategyVersionManager()
        result = version_manager.rollback_to_version(strategy_name, version_id)

        if not result:
            return {"error": f"Version {version_id} not found", "success": False}

        version, code = result

        # 保存回滚的代码
        strategy_dir = Path("strategies") / strategy_name
        code_file = strategy_dir / "__init__.py"
        if code_file.exists():
            code_file.write_text(code, encoding="utf-8")

        return {
            "message": f"Strategy {strategy_name} rolled back to version {version_id[:8]}",
            "version_id": version_id,
            "created_at": version.created_at,
            "success": True
        }

    @app.get("/api/strategy/{strategy_name}/version/{version_id}/code")
    async def api_strategy_version_code(strategy_name: str, version_id: str) -> dict:
        """Get code for a specific version."""
        from moneyclaw.agent.strategy_version import StrategyVersionManager

        version_manager = StrategyVersionManager()
        code = version_manager.get_version_code(strategy_name, version_id)
        version = version_manager.get_version(strategy_name, version_id)

        if not code or not version:
            return {"error": "Version not found", "success": False}

        return {
            "version_id": version_id,
            "strategy_name": strategy_name,
            "code": code,
            "created_at": version.created_at,
            "author": version.author,
            "change_summary": version.change_summary,
            "success": True
        }

    @app.get("/api/strategy/versions/all")
    async def api_all_strategy_versions() -> dict:
        """Get all strategies with version history."""
        from moneyclaw.agent.strategy_version import StrategyVersionManager

        version_manager = StrategyVersionManager()
        all_strategies = version_manager.list_all_strategies_with_versions()

        return {
            "strategies": [
                {
                    "name": name,
                    "version_count": len(versions),
                    "latest_version": versions[0].version_id[:8] if versions else None,
                    "latest_created": versions[0].created_at if versions else None,
                }
                for name, versions in all_strategies.items()
            ],
            "success": True
        }

    # --- HTMX Partials ---

    @app.get("/htmx/cards", response_class=HTMLResponse)
    async def htmx_cards() -> str:
        status = await brain.get_status()
        pnl_val = status["today_pnl"]
        pnl_color = "text-green-400" if pnl_val >= 0 else "text-red-400"
        pnl_sign = "+" if pnl_val >= 0 else ""
        order_count = len(executor.order_history) if executor else 0
        return f"""
  <div class="bg-gray-800 p-4 rounded">
    <div class="text-sm text-gray-400">Status</div>
    <div class="text-xl">{"🟢 Running" if status["running"] else "🔴 Stopped"}</div>
  </div>
  <div class="bg-gray-800 p-4 rounded">
    <div class="text-sm text-gray-400">Today P&L</div>
    <div class="text-xl {pnl_color}">{pnl_sign}${pnl_val:.2f}</div>
  </div>
  <div class="bg-gray-800 p-4 rounded">
    <div class="text-sm text-gray-400">Strategies</div>
    <div class="text-xl">{status["strategies_active"]} active</div>
  </div>
  <div class="bg-gray-800 p-4 rounded">
    <div class="text-sm text-gray-400">Orders</div>
    <div class="text-xl">{order_count}</div>
  </div>"""

    @app.get("/htmx/history", response_class=HTMLResponse)
    async def htmx_history() -> str:
        history = await memory.get_history(limit=10)
        rows = ""
        for h in history:
            pnl = h["profit_loss"]
            color = "text-green-400" if pnl >= 0 else "text-red-400"
            sign = "+" if pnl >= 0 else ""
            ts = datetime.fromtimestamp(h["executed_at"], tz=UTC).strftime("%m-%d %H:%M")
            title = html.escape(str(h["title"])[:50])
            rows += (
                f'<tr class="border-b border-gray-700">'
                f'<td class="py-2 px-3 text-gray-400">{ts}</td>'
                f'<td class="py-2 px-3">{html.escape(str(h["strategy"]))}</td>'
                f'<td class="py-2 px-3">{title}</td>'
                f'<td class="py-2 px-3 {color}">{sign}${pnl:.2f}</td>'
                f"</tr>"
            )
        empty = '<tr><td colspan="4" class="py-4 text-center text-gray-500">No trades yet</td></tr>'
        return f"""
  <div class="text-sm text-gray-400 mb-2">Recent Trades</div>
  <table class="w-full text-sm">
    <thead><tr class="text-gray-400 border-b border-gray-600">
      <th class="py-2 px-3 text-left">Time</th>
      <th class="py-2 px-3 text-left">Strategy</th>
      <th class="py-2 px-3 text-left">Title</th>
      <th class="py-2 px-3 text-left">P&L</th>
    </tr></thead>
    <tbody>{rows or empty}</tbody>
  </table>"""

    return app
