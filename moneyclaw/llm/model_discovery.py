"""Model discovery — 自动发现并注册各Provider的可用模型."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from moneyclaw.llm.model_intelligence import create_profile_from_model_id
from moneyclaw.llm.model_profile import ModelProfile

log = structlog.get_logger()

# Provider模型列表API配置
PROVIDER_CONFIGS: dict[str, dict[str, Any]] = {
    "openai": {
        "list_url": "https://api.openai.com/v1/models",
        "api_key_env": "OPENAI_API_KEY",
        "model_key": "id",
        "extract_id": lambda m: m.get("id", ""),
        "filter": lambda m: m.get("id", "").startswith("gpt-"),
    },
    "anthropic": {
        # Anthropic 没有模型列表API，使用固定列表
        "fixed_models": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
        ],
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "deepseek": {
        "list_url": "https://api.deepseek.com/models",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_key": "id",
        "extract_id": lambda m: m.get("id", ""),
        "header_name": "Authorization",
        "header_format": "Bearer {key}",
    },
    "groq": {
        "list_url": "https://api.groq.com/openai/v1/models",
        "api_key_env": "GROQ_API_KEY",
        "model_key": "id",
        "extract_id": lambda m: m.get("id", ""),
        "header_name": "Authorization",
        "header_format": "Bearer {key}",
    },
    "ollama": {
        "list_url": "{base_url}/api/tags",
        "api_key_env": None,  # Ollama不需要API key
        "base_url_env": "OLLAMA_BASE_URL",
        "default_base_url": "http://localhost:11434",
        "model_key": "name",
        "extract_id": lambda m: m.get("name", ""),
    },
    "gemini": {
        # Google Gemini API 模型列表
        "list_url": "https://generativelanguage.googleapis.com/v1/models",
        "api_key_env": "GOOGLE_API_KEY",
        "api_key_query": "key",
        "model_key": "name",
        "extract_id": lambda m: m.get("name", "").replace("models/", ""),
        "filter": lambda m: m.get("name", "").startswith("models/gemini"),
    },
    "moonshot": {
        # Moonshot (月之暗面) API 模型列表
        "list_url": "https://api.moonshot.cn/v1/models",
        "api_key_env": "MOONSHOT_API_KEY",
        "model_key": "id",
        "extract_id": lambda m: m.get("id", ""),
        "header_name": "Authorization",
        "header_format": "Bearer {key}",
    },
}


# Providers that run locally and never bill per token.
LOCAL_PROVIDERS = frozenset({"ollama"})


@dataclass
class DiscoveryResult:
    """模型发现结果."""

    provider: str
    models: list[ModelProfile]
    success: bool
    error: str | None = None


class ModelDiscoveryService:
    """自动模型发现服务.

    根据配置的API keys，自动查询各Provider的可用模型列表，
    并创建对应的ModelProfile。
    """

    def __init__(
        self,
        timeout: float = 10.0,
        api_keys: dict[str, str] | None = None,
        local_only: bool = False,
    ) -> None:
        self._timeout = timeout
        self._results: dict[str, DiscoveryResult] = {}
        self._api_keys = api_keys or {}
        # $0 guard: only discover free local models (Ollama); never paid cloud APIs,
        # even if their keys are present in the environment.
        self._local_only = local_only

    async def discover_all(
        self,
        custom_configs: dict[str, dict[str, Any]] | None = None,
    ) -> list[ModelProfile]:
        """并行发现所有配置了API key的Provider的模型.

        Args:
            custom_configs: 自定义Provider配置，覆盖默认配置

        Returns:
            所有发现的模型Profile列表
        """
        configs = {**PROVIDER_CONFIGS, **(custom_configs or {})}
        tasks = []

        for provider, config in configs.items():
            if self._local_only and provider not in LOCAL_PROVIDERS:
                continue
            if self._should_discover(provider, config):
                tasks.append(self._discover_provider(provider, config))

        if not tasks:
            log.warning("discovery.no_providers_configured")
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_models: list[ModelProfile] = []
        for result in results:
            if isinstance(result, Exception):
                log.error("discovery.provider_failed", error=str(result))
            elif result.success:
                all_models.extend(result.models)
                self._results[result.provider] = result
                log.info(
                    "discovery.provider_success",
                    provider=result.provider,
                    models=len(result.models),
                )
            else:
                log.warning(
                    "discovery.provider_error",
                    provider=result.provider,
                    error=result.error,
                )

        log.info("discovery.complete", total_models=len(all_models))
        return all_models

    def _should_discover(self, provider: str, config: dict[str, Any]) -> bool:
        """检查是否应该发现该Provider的模型."""
        import os

        # 检查API key - 优先使用传入的api_keys，其次环境变量
        api_key_env = config.get("api_key_env")
        if api_key_env:
            # 先从传入的api_keys查找，再从环境变量查找
            api_key = self._api_keys.get(api_key_env) or os.environ.get(api_key_env)
            if not api_key:
                # Ollama特殊处理：不需要API key
                if provider == "ollama":
                    return True
                return False

        return True

    async def _discover_provider(
        self,
        provider: str,
        config: dict[str, Any],
    ) -> DiscoveryResult:
        """发现单个Provider的模型."""
        try:
            # 处理固定模型列表（如Anthropic）
            if "fixed_models" in config:
                models = self._create_profiles_from_list(
                    provider, config["fixed_models"]
                )
                return DiscoveryResult(provider=provider, models=models, success=True)

            # 处理API查询
            models = await self._fetch_models_from_api(provider, config)
            return DiscoveryResult(provider=provider, models=models, success=True)

        except Exception as e:
            log.exception("discovery.provider_exception", provider=provider)
            return DiscoveryResult(
                provider=provider,
                models=[],
                success=False,
                error=str(e),
            )

    async def _fetch_models_from_api(
        self,
        provider: str,
        config: dict[str, Any],
    ) -> list[ModelProfile]:
        """从API获取模型列表."""
        import os

        # 构建URL
        list_url = config["list_url"]
        if "{base_url}" in list_url:
            base_url_env = config.get("base_url_env", "")
            default_url = config.get("default_base_url", "http://localhost:11434")
            base_url = os.environ.get(base_url_env, default_url)
            list_url = list_url.format(base_url=base_url.rstrip("/"))

        # 构建请求头
        headers: dict[str, str] = {}
        params: dict[str, str] = {}

        api_key_env = config.get("api_key_env")
        if api_key_env:
            # 优先使用传入的api_keys，其次环境变量
            api_key = self._api_keys.get(api_key_env) or os.environ.get(api_key_env, "")
            if api_key:
                header_name = config.get("header_name", "Authorization")
                header_format = config.get("header_format", "Bearer {key}")

                # 检查是否使用query参数传递API key
                api_key_query = config.get("api_key_query")
                if api_key_query:
                    params[api_key_query] = api_key
                else:
                    headers[header_name] = header_format.format(key=api_key)

        # 发送请求
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(list_url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as e:
            log.warning("discovery.connection_failed", provider=provider, url=list_url, error=str(e))
            return []
        except httpx.TimeoutException as e:
            log.warning("discovery.timeout", provider=provider, url=list_url, error=str(e))
            return []
        except httpx.HTTPStatusError as e:
            log.warning("discovery.http_error", provider=provider, url=list_url, status=e.response.status_code)
            return []

        # 解析模型列表
        model_data_list = data.get("data", []) if isinstance(data, dict) else data
        model_ids: list[str] = []

        extract_fn = config.get("extract_id", lambda m: m.get("id", ""))
        filter_fn = config.get("filter")

        for model_data in model_data_list:
            model_id = extract_fn(model_data)
            if not model_id:
                continue

            # 应用过滤器
            if filter_fn and not filter_fn(model_data):
                continue

            # 构建完整的model_id格式
            full_model_id = f"{provider}/{model_id}"
            model_ids.append(full_model_id)

        return self._create_profiles_from_list(provider, model_ids, model_data_list)

    def _create_profiles_from_list(
        self,
        provider: str,
        model_ids: list[str],
        raw_data: list[dict] | None = None,
    ) -> list[ModelProfile]:
        """从模型ID列表创建ModelProfile."""
        profiles = []
        raw_data_map = {d.get("id", ""): d for d in (raw_data or [])}

        for model_id in model_ids:
            # 提取原始API数据作为元数据
            model_name = model_id.split("/")[-1] if "/" in model_id else model_id
            metadata = raw_data_map.get(model_name, {})

            try:
                profile = create_profile_from_model_id(model_id, provider, **metadata)
                profiles.append(profile)
            except Exception:
                log.warning("discovery.create_profile_failed", model_id=model_id)

        return profiles

    def get_results(self) -> dict[str, DiscoveryResult]:
        """获取所有发现结果."""
        return self._results.copy()

    def get_discovered_models(self, provider: str | None = None) -> list[ModelProfile]:
        """获取已发现的模型列表."""
        if provider:
            result = self._results.get(provider)
            return result.models if result else []

        all_models: list[ModelProfile] = []
        for result in self._results.values():
            all_models.extend(result.models)
        return all_models
