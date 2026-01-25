"""Tests for multiplexer slash commands."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from vibe.cli.commands import CommandRegistry
from vibe.core.types import MultiplexerStats, PerModelStats


# -----------------------------------------------------------------------------
# CommandRegistry Tests
# -----------------------------------------------------------------------------


class TestMuxCommandRegistry:
    """Tests for /mux command registration."""

    def test_mux_command_registered(self) -> None:
        registry = CommandRegistry()
        assert "mux" in registry.commands

    def test_mux_command_aliases(self) -> None:
        registry = CommandRegistry()
        cmd = registry.commands["mux"]
        assert "/mux" in cmd.aliases

    def test_mux_command_handler(self) -> None:
        registry = CommandRegistry()
        cmd = registry.commands["mux"]
        assert cmd.handler == "_handle_mux"

    def test_find_mux_command(self) -> None:
        registry = CommandRegistry()
        cmd = registry.find_command("/mux")

        assert cmd is not None
        assert cmd.handler == "_handle_mux"

    def test_mux_command_in_help(self) -> None:
        registry = CommandRegistry()
        help_text = registry.get_help_text()

        assert "/mux" in help_text

    def test_mux_command_excluded(self) -> None:
        registry = CommandRegistry(excluded_commands=["mux"])
        assert "mux" not in registry.commands
        assert registry.find_command("/mux") is None


# -----------------------------------------------------------------------------
# MultiplexerStats Formatting Tests
# -----------------------------------------------------------------------------


class TestMuxStatsFormatting:
    """Tests for multiplexer stats formatting."""

    def test_format_disabled_multiplexer(self) -> None:
        """Test formatting when multiplexer is disabled."""
        stats = MultiplexerStats(enabled=False)

        assert stats.enabled is False
        assert stats.mode == "single"
        assert stats.models_in_pool == 0

    def test_format_enabled_multiplexer(self) -> None:
        """Test formatting when multiplexer is enabled."""
        stats = MultiplexerStats(
            enabled=True,
            mode="failover",
            models_in_pool=2,
            models_available=2,
            models_disabled=0,
        )

        assert stats.enabled is True
        assert stats.mode == "failover"
        assert stats.models_in_pool == 2

    def test_format_with_per_model_stats(self) -> None:
        """Test formatting with per-model statistics."""
        per_model = {
            "devstral-2": PerModelStats(
                success_count=10,
                rate_limit_count=2,
                fail_count=0,
                is_disabled=False,
            ),
            "devstral-small": PerModelStats(
                success_count=5,
                rate_limit_count=0,
                fail_count=1,
                is_disabled=False,
            ),
        }

        stats = MultiplexerStats(
            enabled=True,
            mode="load_balance",
            models_in_pool=2,
            models_available=2,
            per_model=per_model,
        )

        assert "devstral-2" in stats.per_model
        assert stats.per_model["devstral-2"].success_count == 10

    def test_format_with_disabled_model(self) -> None:
        """Test formatting with a disabled model."""
        import time

        disabled_until = time.time() + 60  # 60 seconds from now

        per_model = {
            "devstral-2": PerModelStats(
                success_count=5,
                rate_limit_count=3,
                fail_count=0,
                is_disabled=True,
                disabled_until_timestamp=disabled_until,
            ),
        }

        stats = MultiplexerStats(
            enabled=True,
            mode="failover",
            models_in_pool=2,
            models_available=1,
            models_disabled=1,
            per_model=per_model,
        )

        assert stats.models_disabled == 1
        assert stats.per_model["devstral-2"].is_disabled is True


# -----------------------------------------------------------------------------
# PerModelStats Tests
# -----------------------------------------------------------------------------


class TestPerModelStats:
    """Tests for PerModelStats computed fields."""

    def test_total_requests(self) -> None:
        stats = PerModelStats(
            success_count=10,
            rate_limit_count=2,
            fail_count=1,
        )
        assert stats.total_requests == 13

    def test_success_rate_with_requests(self) -> None:
        stats = PerModelStats(
            success_count=8,
            rate_limit_count=1,
            fail_count=1,
        )
        assert stats.success_rate == 0.8

    def test_success_rate_no_requests(self) -> None:
        stats = PerModelStats(
            success_count=0,
            rate_limit_count=0,
            fail_count=0,
        )
        assert stats.success_rate == 0.0

    def test_disabled_state(self) -> None:
        import time

        future_time = time.time() + 100
        stats = PerModelStats(
            is_disabled=True,
            disabled_until_timestamp=future_time,
        )
        assert stats.is_disabled is True
        assert stats.disabled_until_timestamp == future_time


# -----------------------------------------------------------------------------
# MultiplexerStats Tests
# -----------------------------------------------------------------------------


class TestMultiplexerStats:
    """Tests for MultiplexerStats model."""

    def test_default_values(self) -> None:
        stats = MultiplexerStats()

        assert stats.enabled is False
        assert stats.mode == "single"
        assert stats.models_in_pool == 0
        assert stats.models_available == 0
        assert stats.models_disabled == 0
        assert stats.last_used_model is None
        assert stats.per_model == {}

    def test_with_all_values(self) -> None:
        stats = MultiplexerStats(
            enabled=True,
            mode="load_balance",
            models_in_pool=3,
            models_available=2,
            models_disabled=1,
            last_used_model="devstral-2",
            per_model={
                "model1": PerModelStats(success_count=5),
            },
        )

        assert stats.enabled is True
        assert stats.mode == "load_balance"
        assert stats.models_in_pool == 3
        assert stats.models_available == 2
        assert stats.models_disabled == 1
        assert stats.last_used_model == "devstral-2"
        assert len(stats.per_model) == 1
