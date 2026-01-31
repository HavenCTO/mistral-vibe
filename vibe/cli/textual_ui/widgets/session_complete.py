from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from vibe.cli.textual_ui.widgets.messages import NonSelectableStatic
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.types import MultiplexerStats


class SessionCompleteMessage(Static):
    """Display session completion with multiplexer usage statistics."""

    def __init__(self, stats: MultiplexerStats | None) -> None:
        super().__init__()
        self.add_class("session-complete-message")
        self._stats = stats

    def compose(self) -> ComposeResult:
        with Vertical(classes="session-complete-container"):
            # Header
            with Horizontal(classes="session-complete-header"):
                yield NonSelectableStatic("✓", classes="session-complete-icon")
                yield NoMarkupStatic("Session complete", classes="session-complete-title")

            # Multiplexer stats (only if multiplexer is enabled and has data)
            if self._stats and self._stats.enabled and self._stats.per_model:
                yield NoMarkupStatic(
                    f"Mode: {self._stats.mode} | Pool: {self._stats.models_available}/{self._stats.models_in_pool} models available",
                    classes="session-complete-summary"
                )

                # Per-model stats
                with Vertical(classes="session-complete-models"):
                    yield NoMarkupStatic(
                        "Model Usage:",
                        classes="session-complete-models-header"
                    )

                    for model_name, model_stats in self._stats.per_model.items():
                        if model_stats.total_requests > 0:
                            status = "●" if not model_stats.is_disabled else "○"
                            status_class = (
                                "model-enabled" if not model_stats.is_disabled else "model-disabled"
                            )
                            yield NoMarkupStatic(
                                f"  {status} {model_name}: {model_stats.success_count} ok, "
                                f"{model_stats.rate_limit_count} rate-limited, "
                                f"{model_stats.fail_count} failed "
                                f"({model_stats.success_rate*100:.0f}% success)",
                                classes=f"session-complete-model-line {status_class}"
                            )
