"""Tests for multiplexer configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tomli_w

from vibe.core.config import (
    ModelConfig,
    ModelPoolEntry,
    MultiplexerConfig,
    MultiplexerMode,
    ProviderConfig,
    VibeConfig,
)


# -----------------------------------------------------------------------------
# ModelPoolEntry Tests
# -----------------------------------------------------------------------------


class TestModelPoolEntry:
    """Tests for ModelPoolEntry validation."""

    def test_default_values(self) -> None:
        entry = ModelPoolEntry(model="test")

        assert entry.model == "test"
        assert entry.weight == 1
        assert entry.max_concurrent is None
        assert entry.is_fallback is False

    def test_custom_weight(self) -> None:
        entry = ModelPoolEntry(model="test", weight=5)
        assert entry.weight == 5

    def test_weight_minimum(self) -> None:
        with pytest.raises(ValueError):
            ModelPoolEntry(model="test", weight=0)

    def test_max_concurrent_minimum(self) -> None:
        with pytest.raises(ValueError):
            ModelPoolEntry(model="test", max_concurrent=0)

    def test_valid_max_concurrent(self) -> None:
        entry = ModelPoolEntry(model="test", max_concurrent=10)
        assert entry.max_concurrent == 10

    def test_is_fallback_true(self) -> None:
        entry = ModelPoolEntry(model="test", is_fallback=True)
        assert entry.is_fallback is True


# -----------------------------------------------------------------------------
# MultiplexerConfig Tests
# -----------------------------------------------------------------------------


class TestMultiplexerConfig:
    """Tests for MultiplexerConfig validation."""

    def test_default_values(self) -> None:
        config = MultiplexerConfig()

        assert config.enabled is False
        assert config.mode == MultiplexerMode.SINGLE
        assert config.pool == []
        assert config.rate_limit_disable_duration_ms == 60_000
        assert config.persistent_error_disable_duration_ms == 15_000

    def test_enabled_true(self) -> None:
        config = MultiplexerConfig(enabled=True)
        assert config.enabled is True

    def test_failover_mode(self) -> None:
        config = MultiplexerConfig(mode=MultiplexerMode.FAILOVER)
        assert config.mode == MultiplexerMode.FAILOVER

    def test_load_balance_mode(self) -> None:
        config = MultiplexerConfig(mode=MultiplexerMode.LOAD_BALANCE)
        assert config.mode == MultiplexerMode.LOAD_BALANCE

    def test_mode_from_string(self) -> None:
        config = MultiplexerConfig(mode="failover")
        assert config.mode == MultiplexerMode.FAILOVER

    def test_pool_with_entries(self) -> None:
        config = MultiplexerConfig(
            pool=[
                ModelPoolEntry(model="primary", weight=2),
                ModelPoolEntry(model="fallback", is_fallback=True),
            ]
        )
        assert len(config.pool) == 2
        assert config.pool[0].model == "primary"
        assert config.pool[1].is_fallback is True

    def test_rate_limit_duration_minimum(self) -> None:
        with pytest.raises(ValueError):
            MultiplexerConfig(rate_limit_disable_duration_ms=500)

    def test_persistent_error_duration_minimum(self) -> None:
        with pytest.raises(ValueError):
            MultiplexerConfig(persistent_error_disable_duration_ms=500)


# -----------------------------------------------------------------------------
# VibeConfig Multiplexer Validation Tests
# -----------------------------------------------------------------------------


class TestVibeConfigMultiplexerValidation:
    """Tests for VibeConfig multiplexer pool validation."""

    @pytest.fixture
    def base_config(self) -> dict[str, Any]:
        """Base configuration with required fields."""
        return {
            "active_model": "devstral-2",
            "providers": [
                {
                    "name": "mistral",
                    "api_base": "https://api.mistral.ai/v1",
                    "api_key_env_var": "MISTRAL_API_KEY",
                    "backend": "mistral",
                }
            ],
            "models": [
                {
                    "name": "mistral-vibe-cli-latest",
                    "provider": "mistral",
                    "alias": "devstral-2",
                },
                {
                    "name": "devstral-small-latest",
                    "provider": "mistral",
                    "alias": "devstral-small",
                },
            ],
        }

    def test_disabled_multiplexer_no_validation(
        self, base_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Disabled multiplexer skips pool validation."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        base_config["multiplexer"] = {
            "enabled": False,
            "pool": [{"model": "invalid-alias"}],
        }

        # Should not raise, because enabled=False
        config = VibeConfig(**base_config)
        assert config.multiplexer.enabled is False

    def test_empty_pool_with_enabled(
        self, base_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty pool with enabled=True should pass (uses active_model)."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        base_config["multiplexer"] = {
            "enabled": True,
            "pool": [],
        }

        config = VibeConfig(**base_config)
        assert config.multiplexer.enabled is True
        assert len(config.multiplexer.pool) == 0

    def test_valid_pool_reference(
        self, base_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pool entry referencing valid model alias passes."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        base_config["multiplexer"] = {
            "enabled": True,
            "pool": [{"model": "devstral-2"}],
        }

        config = VibeConfig(**base_config)
        assert len(config.multiplexer.pool) == 1
        assert config.multiplexer.pool[0].model == "devstral-2"

    def test_invalid_pool_reference(
        self, base_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pool entry referencing unknown model alias fails."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        base_config["multiplexer"] = {
            "enabled": True,
            "pool": [{"model": "nonexistent-alias"}],
        }

        with pytest.raises(ValueError, match="unknown model alias"):
            VibeConfig(**base_config)

    def test_all_fallback_pool_rejected(
        self, base_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pool with only fallback models is rejected."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        base_config["multiplexer"] = {
            "enabled": True,
            "pool": [
                {"model": "devstral-2", "is_fallback": True},
                {"model": "devstral-small", "is_fallback": True},
            ],
        }

        with pytest.raises(ValueError, match="at least one non-fallback model"):
            VibeConfig(**base_config)

    def test_mixed_pool_accepted(
        self, base_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pool with primary and fallback models is accepted."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        base_config["multiplexer"] = {
            "enabled": True,
            "pool": [
                {"model": "devstral-2", "is_fallback": False},
                {"model": "devstral-small", "is_fallback": True},
            ],
        }

        config = VibeConfig(**base_config)
        assert len(config.multiplexer.pool) == 2

    def test_pool_model_api_key_validation(
        self, base_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pool models require API keys for their providers."""
        # Don't set the API key
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        base_config["multiplexer"] = {
            "enabled": True,
            "pool": [{"model": "devstral-2"}],
        }

        with pytest.raises(RuntimeError, match="Missing .* environment variable"):
            VibeConfig(**base_config)

    def test_pool_with_weighted_entries(
        self, base_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pool entries can have custom weights."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        base_config["multiplexer"] = {
            "enabled": True,
            "mode": "load_balance",
            "pool": [
                {"model": "devstral-2", "weight": 70},
                {"model": "devstral-small", "weight": 30},
            ],
        }

        config = VibeConfig(**base_config)
        assert config.multiplexer.pool[0].weight == 70
        assert config.multiplexer.pool[1].weight == 30

    def test_pool_with_max_concurrent(
        self, base_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pool entries can have max_concurrent limits."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        base_config["multiplexer"] = {
            "enabled": True,
            "pool": [
                {"model": "devstral-2", "max_concurrent": 5},
            ],
        }

        config = VibeConfig(**base_config)
        assert config.multiplexer.pool[0].max_concurrent == 5


# -----------------------------------------------------------------------------
# TOML Configuration Loading Tests
# -----------------------------------------------------------------------------


class TestMultiplexerConfigFromTOML:
    """Tests for loading multiplexer config from TOML files."""

    def test_load_basic_multiplexer_config(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Load basic multiplexer configuration from TOML."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        config_content = {
            "active_model": "devstral-latest",
            "providers": [
                {
                    "name": "mistral",
                    "api_base": "https://api.mistral.ai/v1",
                    "api_key_env_var": "MISTRAL_API_KEY",
                    "backend": "mistral",
                }
            ],
            "models": [
                {
                    "name": "mistral-vibe-cli-latest",
                    "provider": "mistral",
                    "alias": "devstral-latest",
                }
            ],
            "multiplexer": {
                "enabled": True,
                "mode": "failover",
                "pool": [{"model": "devstral-latest"}],
            },
        }

        config_file = config_dir / "config.toml"
        config_file.write_text(tomli_w.dumps(config_content), encoding="utf-8")

        config = VibeConfig.load()
        assert config.multiplexer.enabled is True
        assert config.multiplexer.mode == MultiplexerMode.FAILOVER

    def test_load_multiplexer_with_durations(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Load multiplexer with custom duration settings."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        config_content = {
            "active_model": "devstral-latest",
            "providers": [
                {
                    "name": "mistral",
                    "api_base": "https://api.mistral.ai/v1",
                    "api_key_env_var": "MISTRAL_API_KEY",
                    "backend": "mistral",
                }
            ],
            "models": [
                {
                    "name": "mistral-vibe-cli-latest",
                    "provider": "mistral",
                    "alias": "devstral-latest",
                }
            ],
            "multiplexer": {
                "enabled": True,
                "rate_limit_disable_duration_ms": 120000,
                "persistent_error_disable_duration_ms": 30000,
                "pool": [{"model": "devstral-latest"}],
            },
        }

        config_file = config_dir / "config.toml"
        config_file.write_text(tomli_w.dumps(config_content), encoding="utf-8")

        config = VibeConfig.load()
        assert config.multiplexer.rate_limit_disable_duration_ms == 120000
        assert config.multiplexer.persistent_error_disable_duration_ms == 30000


# -----------------------------------------------------------------------------
# Environment Variable Override Tests
# -----------------------------------------------------------------------------


class TestMultiplexerEnvOverrides:
    """Tests for environment variable overrides of multiplexer config."""

    def test_env_override_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """VIBE_MULTIPLEXER__ENABLED environment variable works."""
        # Note: Pydantic settings with nested models may need special handling
        # This test documents expected behavior
        pass  # Placeholder for env override testing


# -----------------------------------------------------------------------------
# MultiplexerMode Enum Tests
# -----------------------------------------------------------------------------


class TestMultiplexerMode:
    """Tests for MultiplexerMode enum."""

    def test_single_mode_value(self) -> None:
        assert MultiplexerMode.SINGLE.value == "single"

    def test_failover_mode_value(self) -> None:
        assert MultiplexerMode.FAILOVER.value == "failover"

    def test_load_balance_mode_value(self) -> None:
        assert MultiplexerMode.LOAD_BALANCE.value == "load_balance"

    def test_mode_from_string(self) -> None:
        assert MultiplexerMode("single") == MultiplexerMode.SINGLE
        assert MultiplexerMode("failover") == MultiplexerMode.FAILOVER
        assert MultiplexerMode("load_balance") == MultiplexerMode.LOAD_BALANCE
