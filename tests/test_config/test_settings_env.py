"""Every settings section must read the .env file, not only process env vars."""

from __future__ import annotations

from moneyclaw.config.settings import (
    AlpacaSettings,
    ExchangeSettings,
    RiskSettings,
    TelegramSettings,
)


def test_sections_read_env_file(tmp_path, monkeypatch) -> None:
    for var in ("TELEGRAM_TOKEN", "RISK_MAX_PORTFOLIO_BUDGET", "ALPACA_PAPER", "EXCHANGE_DRY_RUN"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "TELEGRAM_TOKEN=abc\n"
        "RISK_MAX_PORTFOLIO_BUDGET=75\n"
        "ALPACA_API_KEY=k\nALPACA_SECRET_KEY=s\nALPACA_PAPER=false\n"
        "EXCHANGE_DRY_RUN=false\n"
        "UNRELATED_KEY=1\n"
    )
    assert TelegramSettings(_env_file=env).token == "abc"
    assert RiskSettings(_env_file=env).max_portfolio_budget == 75
    alpaca = AlpacaSettings(_env_file=env)
    assert (alpaca.api_key, alpaca.secret_key, alpaca.paper) == ("k", "s", False)
    assert ExchangeSettings(_env_file=env).dry_run is False


def test_safe_defaults() -> None:
    assert AlpacaSettings().paper is True
    assert ExchangeSettings().dry_run is True
