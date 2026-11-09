"""Shared test setup."""

from __future__ import annotations

from pydantic_settings import BaseSettings

from moneyclaw.config import settings as _settings_mod

# Tests must not depend on a developer's local .env (real keys, custom limits).
for _obj in vars(_settings_mod).values():
    if isinstance(_obj, type) and issubclass(_obj, BaseSettings):
        _obj.model_config["env_file"] = None
