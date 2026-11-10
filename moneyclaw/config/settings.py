"""Pydantic Settings — type-safe configuration loaded from env/.env file."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Every settings section reads the project's .env file (not just process env vars).
_ENV = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class TelegramSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TELEGRAM_", **_ENV)

    token: str = ""
    chat_id: str = ""


class LLMSettings(BaseSettings):
    """LLM provider configuration — SmartRouter auto-discovers models from API keys."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
        extra="ignore",
    )

    # Provider API keys — SmartRouter will auto-discover available models
    # Just provide the keys, no need to specify model names

    # Local models (Ollama)
    ollama_base_url: str = "http://localhost:11434"

    # Cloud providers — SmartRouter discovers models automatically
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    deepseek_api_key: str = ""
    groq_api_key: str = ""
    google_api_key: str = ""  # For Gemini
    moonshot_api_key: str = ""  # For Moonshot (月之暗面/Kimi)

    # Budget and routing configuration
    # USD/day for paid LLM APIs. 0 (default) = paid models are never discovered or
    # called; strategies run on rule-based (Layer 0) scoring and local Ollama only.
    daily_llm_budget: float = 0.0
    budget_caution_threshold: float = 0.6  # Enter caution mode at 60% usage
    budget_critical_threshold: float = 0.85  # Enter critical mode at 85% usage
    reserve_budget_for_urgent: float = 0.1  # Reserve 10% for urgent tasks
    enable_auto_downgrade: bool = True  # Automatically downgrade when budget is low

    # Performance learning
    enable_performance_learning: bool = True  # Track and learn from model performance
    min_calls_for_learning: int = 3  # Minimum calls before using performance data

    # LLM discovery settings (renamed to avoid pydantic protected namespace warning)
    llm_discovery_timeout: float = 10.0  # Timeout for model discovery (seconds)
    llm_cache_ttl: int = 900  # Cache discovered models for 15 minutes


class RiskSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_", **_ENV)

    max_trade_amount: float = 20.0  # Max single trade in USD
    max_daily_loss: float = 20.0  # Stop trading after this much realized loss per day
    approval_threshold: float = 15.0  # Trades at/above this need Telegram approval
    max_portfolio_budget: float = 50.0  # Max USD deployed across all open stock positions
    cooldown_after_losses: int = 3  # Pause after N consecutive losses
    per_strategy_daily_loss: float = 30.0  # Per-strategy daily loss cap (0 = disabled)
    max_position_ratio: float = 0.5  # Max ratio of portfolio in single asset (0-1)
    dry_run: bool = True  # Default: simulate all trades


class ExchangeConfig(BaseSettings):
    """Config for a single exchange connection."""

    api_key: str = ""
    secret: str = ""
    password: str = ""  # Some exchanges need a passphrase


class ExchangeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXCHANGE_", **_ENV)

    # Default exchange for strategies that don't specify one
    default_exchange: str = "binance"
    dry_run: bool = True  # Global dry_run override
    # Per-exchange configs loaded from env: BINANCE_API_KEY, etc.
    binance_api_key: str = ""
    binance_secret: str = ""
    okx_api_key: str = ""
    okx_secret: str = ""
    okx_password: str = ""


class AlpacaSettings(BaseSettings):
    """Alpaca stock trading (trading + account/position reads only)."""

    model_config = SettingsConfigDict(env_prefix="ALPACA_", **_ENV)

    api_key: str = ""
    secret_key: str = ""
    paper: bool = True  # True → paper-api.alpaca.markets, False → real money
    dry_run_cash: float = 100.0  # simulated settled cash when EXCHANGE_DRY_RUN=true


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATA_", **_ENV)

    duckdb_path: str = "data/market.duckdb"
    price_poll_interval: int = 300  # 5 minutes
    news_poll_interval: int = 900  # 15 minutes


class Settings(BaseSettings):
    """Root settings — loads from .env file and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configs
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    data: DataSettings = Field(default_factory=DataSettings)

    # General
    db_path: str = "data/moneyclaw.db"
    web_host: str = "0.0.0.0"
    web_port: int = 8080
    log_level: str = "INFO"

    # Agent behavior
    scan_interval: int = 60  # Seconds between scans
    strategies_dir: str = "strategies"
