"""Scheduled jobs — data collection, daily reports, risk resets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from moneyclaw.agent.brain import AgentBrain
    from moneyclaw.agent.memory import Memory
    from moneyclaw.config.settings import Settings
    from moneyclaw.data.feeds.crypto import CryptoFeed
    from moneyclaw.data.storage import MarketStorage
    from moneyclaw.execution.risk import RiskManager
    from moneyclaw.interface.telegram.notify import Notifier
    from moneyclaw.scheduler.engine import Scheduler

log = structlog.get_logger()


def register_all_jobs(
    *,
    scheduler: Scheduler,
    brain: AgentBrain,
    risk: RiskManager,
    notifier: Notifier | None,
    memory: Memory,
    crypto_feed: CryptoFeed,
    storage: MarketStorage,
    settings: Settings,
    collect_crypto: bool = True,
) -> None:
    """Register all scheduled jobs."""

    # Daily risk reset at midnight UTC (counters only; limits changed via Telegram persist)
    async def risk_reset() -> None:
        risk.reset_daily()
        log.info("job.risk_reset")

    scheduler.add_cron("risk_reset", risk_reset, hour=0, minute=0)

    # Collect crypto prices every N minutes
    async def collect_crypto_prices() -> None:
        for coin in ("bitcoin", "ethereum", "solana"):
            quote = await crypto_feed.get_price(coin)
            if quote:
                storage.store_quotes([quote])
        log.info("job.crypto_prices_collected")

    if collect_crypto:  # Skipped in stocks-only mode
        scheduler.add_interval(
            "collect_crypto_prices",
            collect_crypto_prices,
            seconds=settings.data.price_poll_interval,
        )

    # Daily report at 21:00 UTC (send via Telegram)
    async def daily_report() -> None:
        today_pnl = await memory.today_pnl()
        history = await memory.get_history(limit=20)
        trade_count = len(history)

        report_lines = [
            f"P&L: ${today_pnl:+.2f}",
            f"Trades: {trade_count}",
            f"Dry run: {'Yes' if risk.is_dry_run else 'No'}",
            "",
            "Risk status:",
            f"  Daily loss: ${risk.status()['daily_loss']:.2f}"
            f" / ${risk.status()['daily_loss_limit']:.2f}",
            f"  Consecutive losses: {risk.status()['consecutive_losses']}",
        ]

        llm_status = brain._llm.cost_tracker.format_status()
        report_lines.append(f"\n{llm_status}")

        # Strategy summary
        strategies = brain._strategies.status()
        if strategies:
            report_lines.append("\nStrategies:")
            for s in strategies:
                status = "ON" if s["enabled"] else "OFF"
                report_lines.append(f"  [{status}] {s['name']} ({s['risk_level']})")

        report = "\n".join(report_lines)
        if notifier:
            await notifier.daily_report(report)
        log.info("job.daily_report_sent")

    scheduler.add_cron("daily_report", daily_report, hour=21, minute=0)

    log.info("jobs.registered", count=len(scheduler.jobs))
