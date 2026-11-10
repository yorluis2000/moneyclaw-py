"""Telegram bot — primary interface for controlling MoneyClaw."""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any

import structlog
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

if TYPE_CHECKING:
    from moneyclaw.agent.brain import AgentBrain
    from moneyclaw.agent.memory import Memory
    from moneyclaw.agent.strategy_chat import StrategyChatInterface
    from moneyclaw.execution.alpaca import AlpacaExecutor
    from moneyclaw.execution.risk import RiskManager
    from moneyclaw.llm.router import LLMRouter
    from moneyclaw.plugins.registry import StrategyRegistry

log = structlog.get_logger()

MAX_LIMIT_USD = 100_000.0  # sanity cap on limits typed from the phone

HELP_TEXT = (
    "MoneyClaw commands\n\n"
    "/status — agent status & today's P&L\n"
    "/positions — holdings, P&L, budget used\n"
    "/limits — current risk limits\n"
    "/budget <usd> — max total $ in open positions\n"
    "/maxtrade <usd> — max $ per trade\n"
    "/maxloss <usd> — daily realized loss limit\n"
    "/pause — stop all trading\n"
    "/resume — resume trading\n"
    "/approve <id> · /reject <id> — answer approval requests\n"
    "/strategies · /history · /cost · /ask <question>"
)


def parse_usd(text: str) -> float:
    """Parse '25', '$25', '25.50' → float. Raises ValueError if not a sane positive amount."""
    value = float(text.strip().replace("$", "").replace(",", ""))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("amount must be positive")
    if value > MAX_LIMIT_USD:
        raise ValueError(f"amount must be at most ${MAX_LIMIT_USD:,.0f}")
    return round(value, 2)


class TelegramBot:
    """Telegram bot for controlling MoneyClaw and receiving notifications."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        brain: AgentBrain,
        memory: Memory,
        llm: LLMRouter,
        strategies: StrategyRegistry,
        risk: RiskManager,
        strategy_chat: StrategyChatInterface | None = None,
        stock_executor: AlpacaExecutor | None = None,
    ) -> None:
        self._bot = Bot(token=token)
        self._dp = Dispatcher()
        self._chat_id = chat_id
        self._brain = brain
        self._memory = memory
        self._llm = llm
        self._strategies = strategies
        self._risk = risk
        self._strategy_chat = strategy_chat
        self._stock_executor = stock_executor
        self._pending_strategy_confirmations: dict[str, Any] = {}
        self._register_handlers()

    # ------------------------------------------------------------ command logic
    # Pure text-returning helpers so they can be unit tested without Telegram.

    def limits_text(self) -> str:
        r = self._risk.status()
        deployed = r.get("deployed") or 0.0
        return (
            f"Risk limits{' [PAUSED]' if r['paused'] else ''}\n"
            f"Budget: ${r['max_portfolio_budget']:.2f} (deployed ${deployed:.2f})\n"
            f"Max trade: ${r['max_trade_amount']:.2f}\n"
            f"Approval needed at: ${r['approval_threshold']:.2f}+\n"
            f"Daily loss: ${r['daily_loss']:.2f} / ${r['daily_loss_limit']:.2f}\n"
            f"Mode: {'DRY RUN' if r['dry_run'] else 'LIVE'}"
        )

    def _set_limit(self, arg: str, field: str, label: str, usage: str) -> str:
        if not arg.strip():
            return f"Usage: {usage}\n\n{self.limits_text()}"
        try:
            value = parse_usd(arg)
            self._risk.update_limits(**{field: value})
        except ValueError as e:
            return f"Invalid amount: {e}. Usage: {usage}"
        note = ""
        r = self._risk.status()
        if field == "max_trade_amount" and value > r["max_portfolio_budget"]:
            budget = r["max_portfolio_budget"]
            note = f"\nNote: trades are still capped by the ${budget:.2f} budget."
        if field == "max_portfolio_budget" and (r.get("deployed") or 0) > value:
            note = "\nNote: already above this budget — no new buys until positions are sold."
        return (
            f"{label} set to ${value:.2f}.{note}\n"
            "(Until restart; set RISK_* in .env to keep it.)"
        )

    def budget_text(self, arg: str) -> str:
        return self._set_limit(arg, "max_portfolio_budget", "Portfolio budget", "/budget 50")

    def maxtrade_text(self, arg: str) -> str:
        return self._set_limit(arg, "max_trade_amount", "Max trade", "/maxtrade 20")

    def maxloss_text(self, arg: str) -> str:
        return self._set_limit(arg, "max_daily_loss", "Daily loss limit", "/maxloss 20")

    async def positions_text(self) -> str:
        ex = self._stock_executor
        if not ex:
            return "Stock trading is not configured."
        try:
            positions = await ex.get_positions()
        except Exception as e:
            log.exception("telegram.positions_error")
            return f"Could not load positions: {e}"
        mode = "DRY RUN" if ex.dry_run else ("PAPER" if ex.paper else "LIVE")
        r = self._risk.status()
        try:
            acct = await ex.get_account()
            cash_line = (
                f"Cash available (settled, no margin): ${acct['non_marginable_buying_power']:.2f}"
                f"\nAccount equity: ${acct['equity']:.2f}"
            )
        except Exception as e:
            cash_line = f"Cash available: unavailable ({e})"
        if not positions:
            body = "No open positions."
        else:
            lines = []
            for p in sorted(positions, key=lambda p: p.symbol):
                lines.append(
                    f"{p.symbol}: {p.qty:.4f} sh, ${p.market_value:.2f}"
                    f" ({p.unrealized_pl:+.2f}, {p.unrealized_plpc:+.1%})"
                )
            cost = sum(p.cost_basis for p in positions)
            value = sum(p.market_value for p in positions)
            lines.append(f"\nTotal: ${value:.2f} (cost ${cost:.2f}, P&L {value - cost:+.2f})")
            body = "\n".join(lines)
        return (
            f"Positions [{mode}]\n\n{body}\n\n"
            f"{cash_line}\n"
            f"Budget used: ${ex.deployed_capital:.2f} / ${r['max_portfolio_budget']:.2f}"
            f" (room ${ex.remaining_budget:.2f})\n"
            f"Realized loss today: ${r['daily_loss']:.2f} / ${r['daily_loss_limit']:.2f}"
        )

    def _authorized_chat_id(self) -> int | None:
        try:
            return int(str(self._chat_id).strip())
        except ValueError:
            return None

    def _register_handlers(self) -> None:
        dp = self._dp

        # Only the owner's chat may control the bot. Without TELEGRAM_CHAT_ID set,
        # every message just gets told its chat id so the owner can configure it.
        owner = self._authorized_chat_id()
        if owner is None:
            log.warning("telegram.no_chat_id", hint="set TELEGRAM_CHAT_ID to enable commands")

            @dp.message()
            async def setup_only(message: Message) -> None:
                await message.answer(
                    f"Commands are disabled until TELEGRAM_CHAT_ID is set.\n"
                    f"Your chat id is {message.chat.id}."
                )

            return
        dp.message.filter(F.chat.id == owner)

        @dp.message(Command("help"))
        async def cmd_help(message: Message) -> None:
            await message.answer(HELP_TEXT)

        @dp.message(Command("budget"))
        async def cmd_budget(message: Message) -> None:
            await message.answer(self.budget_text(_arg(message)))

        @dp.message(Command("maxtrade"))
        async def cmd_maxtrade(message: Message) -> None:
            await message.answer(self.maxtrade_text(_arg(message)))

        @dp.message(Command("maxloss"))
        async def cmd_maxloss(message: Message) -> None:
            await message.answer(self.maxloss_text(_arg(message)))

        @dp.message(Command("limits"))
        async def cmd_limits(message: Message) -> None:
            await message.answer(self.limits_text())

        @dp.message(Command("positions"))
        async def cmd_positions(message: Message) -> None:
            await message.answer(await self.positions_text())

        @dp.message(Command("start"))
        async def cmd_start(message: Message) -> None:
            await message.answer("MoneyClaw is active.\n\n" + HELP_TEXT)

        @dp.message(Command("status"))
        async def cmd_status(message: Message) -> None:
            status = await self._brain.get_status()
            text = (
                f"Running: {'Yes' if status['running'] else 'No'}\n"
                f"Strategies: {status['strategies_active']} active\n"
                f"Today P&L: ${status['today_pnl']:.2f}\n"
                f"Pending approvals: {status['pending_approvals']}\n\n"
                f"{status['llm_cost']}"
            )
            await message.answer(text)

        @dp.message(Command("cost"))
        async def cmd_cost(message: Message) -> None:
            await message.answer(self._llm.cost_tracker.format_status())

        @dp.message(Command("strategies"))
        async def cmd_strategies(message: Message) -> None:
            statuses = self._strategies.status()
            if not statuses:
                await message.answer("No strategies loaded.")
                return
            lines = []
            for s in statuses:
                status_icon = "ON" if s["enabled"] else "OFF"
                lines.append(
                    f"[{status_icon}] {s['name']} ({s['risk_level']})\n"
                    f"  {s['description']}\n"
                    f"  Est. ROI: {s['roi_estimate']:.1f}x"
                )
            await message.answer("\n\n".join(lines))

        @dp.message(Command("pause"))
        async def cmd_pause(message: Message) -> None:
            self._risk.pause()
            await message.answer("Trading paused. No new orders until /resume.")

        @dp.message(Command("resume"))
        async def cmd_resume(message: Message) -> None:
            self._risk.resume()
            await message.answer("Trading resumed.\n\n" + self.limits_text())

        @dp.message(Command("approve"))
        async def cmd_approve(message: Message) -> None:
            args = (message.text or "").split(maxsplit=1)
            if len(args) < 2:
                await message.answer("Usage: /approve <id>")
                return
            opp_id = args[1].strip()
            if await self._memory.approve(opp_id):
                await message.answer(f"Approved: {opp_id}")
            else:
                await message.answer(f"Not found or already processed: {opp_id}")

        @dp.message(Command("reject"))
        async def cmd_reject(message: Message) -> None:
            args = (message.text or "").split(maxsplit=1)
            if len(args) < 2:
                await message.answer("Usage: /reject <id>")
                return
            opp_id = args[1].strip()
            if await self._memory.reject(opp_id):
                await message.answer(f"Rejected: {opp_id}")
            else:
                await message.answer(f"Not found or already processed: {opp_id}")

        @dp.message(Command("history"))
        async def cmd_history(message: Message) -> None:
            history = await self._memory.get_history(limit=10)
            if not history:
                await message.answer("No history yet.")
                return
            lines = []
            for h in history:
                pnl = h["profit_loss"]
                sign = "+" if pnl >= 0 else ""
                lines.append(f"{h['strategy']}: {h['title']} ({sign}${pnl:.2f})")
            await message.answer("\n".join(lines))

        @dp.message(Command("ask"))
        async def cmd_ask(message: Message) -> None:
            args = (message.text or "").split(maxsplit=1)
            if len(args) < 2:
                await message.answer("Usage: /ask <question>")
                return
            from moneyclaw.llm.router import LLMLayer, TaskRequest

            try:
                response = await self._llm.complete(
                    TaskRequest(
                        prompt=args[1],
                        system="You are MoneyClaw, a money-saving AI assistant. Be concise.",
                        min_layer=LLMLayer.CHEAP,
                        max_layer=LLMLayer.PREMIUM,
                        complexity=0.5,
                    )
                )
            except Exception:
                log.info("telegram.ask_no_llm")
                await message.answer(
                    "No LLM available (DAILY_LLM_BUDGET=0 blocks paid models; run Ollama"
                    " locally or raise the budget). Trading doesn't need it."
                )
                return
            await message.answer(response.text[:4000])  # Telegram message limit

        @dp.message(Command("strategy"))
        async def cmd_strategy(message: Message) -> None:
            """AI策略管理命令."""
            if not self._strategy_chat:
                await message.answer(
                    "AI策略管理功能未启用。请确保系统已正确配置。"
                )
                return

            args = (message.text or "").split(maxsplit=1)
            if len(args) < 2:
                await message.answer(
                    "🤖 **AI策略管理系统**\n\n"
                    "使用自然语言管理策略:\n"
                    "• `/strategy 创建一个定投BTC的策略`\n"
                    "• `/strategy 优化crypto_dca`\n"
                    "• `/strategy 列出所有策略`\n"
                    "• `/strategy 禁用crypto_funding`\n\n"
                    "或直接发送策略相关消息。"
                )
                return

            user_message = args[1]
            await self._handle_strategy_message(message, user_message)

        @dp.message(F.text)
        async def fallback(message: Message) -> None:
            # 先检查是否有待确认的策略保存
            if await self._handle_strategy_confirmation(message):
                return

            # 检查是否是策略相关消息
            if self._strategy_chat and message.text:
                text = message.text.strip()
                # 策略相关关键词
                strategy_keywords = [
                    "策略", "创建", "生成", "优化", "启用", "禁用", "删除",
                    "strategy", "create", "generate", "optimize", "enable",
                    "disable", "delete", "list", "列出"
                ]
                if any(kw in text.lower() for kw in strategy_keywords):
                    await self._handle_strategy_message(message, text)
                    return

            await message.answer("I don't understand that.\n\n" + HELP_TEXT)

    async def _handle_strategy_message(self, message: Message, text: str) -> None:
        """处理策略相关的自然语言消息."""
        if not self._strategy_chat:
            return

        import structlog
        log = structlog.get_logger()
        log.info("telegram.strategy_command", text=text[:50])

        try:
            response = await self._strategy_chat.handle_message(text)

            # 检查是否需要确认保存策略
            if response.data and response.data.get("pending_confirm"):
                strategy = response.data.get("strategy")
                if strategy:
                    # 存储待确认的策略
                    self._pending_strategy_confirmations[str(message.from_user.id)] = strategy

                    # 添加确认按钮提示
                    response_msg = response.message + "\n\n💡 回复 '是' 或 'yes' 确认保存，回复其他内容取消。"
                    await message.answer(response_msg[:4000])
                    return

            await message.answer(response.message[:4000])

        except Exception as e:
            log.exception("telegram.strategy_error")
            await message.answer(f"处理策略命令时出错: {e}")

    async def _handle_strategy_confirmation(self, message: Message) -> bool:
        """处理策略保存确认. 返回是否处理了确认."""
        user_id = str(message.from_user.id)
        text = message.text or ""

        # 检查是否有待确认的策略
        if user_id not in self._pending_strategy_confirmations:
            return False

        # 检查确认回复
        if text.strip().lower() in ("是", "yes", "确认", "保存", "ok", "y"):
            strategy = self._pending_strategy_confirmations.pop(user_id)
            try:
                from moneyclaw.agent.strategy_generator import GeneratedStrategy

                if isinstance(strategy, GeneratedStrategy):
                    path = await self._strategy_chat.confirm_save_strategy(strategy)
                    await message.answer(f"✅ 策略已保存！\n路径: {path.message}")
                return True
            except Exception as e:
                await message.answer(f"❌ 保存策略失败: {e}")
                return True
        else:
            # 用户取消
            self._pending_strategy_confirmations.pop(user_id, None)
            await message.answer("❎ 已取消保存策略。")
            return True

    @property
    def bot(self) -> Bot:
        return self._bot

    async def start(self) -> None:
        """Poll for Telegram updates; retry with backoff so a Telegram outage
        never takes down the trading loop or dashboard."""
        delay = 5.0
        while True:
            log.info("telegram.starting")
            try:
                await self._dp.start_polling(self._bot, handle_signals=False)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("telegram.polling_failed", error=str(e), retry_in=delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 300.0)

    async def stop(self) -> None:
        await self._bot.session.close()


def _arg(message: Message) -> str:
    parts = (message.text or "").split(maxsplit=1)
    return parts[1] if len(parts) > 1 else ""
