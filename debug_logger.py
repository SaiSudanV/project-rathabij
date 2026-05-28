"""
debug_logger.py — Real-Time Interpretability Engine
====================================================
Streams a live, human-readable log of the AI's internal
decision-making to a secondary terminal window.

Features:
  - Auto-maps action slot numbers to character move names
  - Shows confidence percentages for each action
  - Displays current emotional state with color coding
  - Shows detected player patterns being exploited
  - Tracks difficulty scaler status
  - Rich terminal formatting via the `rich` library

Example output:
┌─────────────────────────────────────────────────────────┐
│ 🧠 LNN Brain — Shadow Knight           [CONFIDENT]     │
├─────────────────────────────────────────────────────────┤
│ ⚔️  Actions: [Dodge Left + Heavy Slash]  Conf: 89%      │
│ 💬 Speech:  "You're predictable."                       │
│ 😤 Mood:    Confident (aggr: 0.3, conf: 0.8, frus: 0.1)│
│ 🎯 Profile: dodge_left_preference (92%) → attack_right  │
│ 📊 Diff:    0.62 | React: 120ms | Win: 55%             │
│ ⏱️  Latency: 2.3ms                                      │
└─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import time
import threading
from typing import Any

import yaml

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# Mood color mapping
MOOD_COLORS = {
    "neutral": "white",
    "confident": "green",
    "cocky": "bright_green",
    "tilted": "red",
    "desperate": "bright_red",
    "cold": "cyan",
}

MOOD_EMOJIS = {
    "neutral": "|",
    "confident": "!",
    "cocky": "*",
    "tilted": "@",
    "desperate": "!",
    "cold": "#",
}


class DebugLogger:
    """
    Real-time debug visualization of the AI's decision-making.

    Usage:
        logger = DebugLogger("config.yaml")
        logger.start()

        # Every frame:
        logger.log_frame(
            actions=[0, 1, 0, 0, 1, ...],
            action_confidences=[0.1, 0.89, 0.05, ...],
            speech="you're predictable",
            mood="confident",
            emotions={"aggression": 0.3, "confidence": 0.8, ...},
            patterns=[{"name": "dodge_left", "confidence": 0.92}],
            difficulty_status={"difficulty": "0.62", ...},
            latency_ms=2.3,
        )
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        debug_cfg = cfg.get("debug", {})
        self.enabled = debug_cfg.get("enabled", True)
        self.log_actions = debug_cfg.get("log_actions", True)
        self.log_emotions = debug_cfg.get("log_emotions", True)
        self.log_profiler = debug_cfg.get("log_profiler", True)
        self.confidence_threshold = debug_cfg.get("log_confidence_threshold", 0.3)

        # Action name mapping (populated from UDP handshake)
        self.action_map: dict[str, str] = {}
        self.character_name: str = "Unknown"

        # State
        self._frame_count = 0
        self._console = Console() if HAS_RICH else None
        self._live: Live | None = None

    def set_action_map(self, action_map: dict[str, str], character_name: str = "") -> None:
        """Set the action map from the UDP handshake."""
        self.action_map = action_map
        self.character_name = character_name

    def _translate_action(self, slot: int) -> str:
        """Translate slot index to move name."""
        return self.action_map.get(str(slot), f"action_{slot}")

    def log_frame(
        self,
        actions: list[int] | None = None,
        action_confidences: list[float] | None = None,
        speech: str = "",
        mood: str = "neutral",
        emotions: dict[str, float] | None = None,
        patterns: list[dict] | None = None,
        difficulty_status: dict | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Log a single frame of AI decision-making.

        Called every game tick with the AI's current internal state.
        """
        if not self.enabled:
            return

        self._frame_count += 1

        if HAS_RICH:
            self._log_rich(
                actions, action_confidences, speech, mood,
                emotions, patterns, difficulty_status, latency_ms,
            )
        else:
            self._log_plain(
                actions, action_confidences, speech, mood,
                emotions, patterns, difficulty_status, latency_ms,
            )

    def _log_rich(
        self,
        actions, action_confidences, speech, mood,
        emotions, patterns, difficulty_status, latency_ms,
    ):
        """Rich terminal output with colors and formatting."""
        console = self._console

        # ── Build Action String ──
        action_str = ""
        if actions and self.log_actions:
            active_moves = []
            for i, pressed in enumerate(actions):
                if pressed:
                    name = self._translate_action(i)
                    conf = action_confidences[i] if action_confidences else 0.0
                    if conf >= self.confidence_threshold:
                        active_moves.append(f"{name} ({conf:.0%})")
            action_str = " + ".join(active_moves) if active_moves else "idle"

        # ── Build Emotion String ──
        emotion_str = ""
        if emotions and self.log_emotions:
            parts = [f"{k}: {v:.2f}" for k, v in emotions.items()]
            emotion_str = ", ".join(parts)

        # ── Build Pattern String ──
        pattern_str = ""
        if patterns and self.log_profiler:
            for p in patterns:
                pattern_str += f"  (P) {p['name']} ({p.get('confidence', '?')}) -> {p.get('exploit', '?')}\n"

        # ── Build Difficulty String ──
        diff_str = ""
        if difficulty_status:
            diff_str = (
                f"Diff: {difficulty_status.get('difficulty', '?')} | "
                f"React: {difficulty_status.get('reaction_delay_ms', '?')}ms | "
                f"Win: {difficulty_status.get('recent_winrate', '?')}"
            )

        # ── Print ──
        mood_color = MOOD_COLORS.get(mood, "white")
        mood_emoji = MOOD_EMOJIS.get(mood, "❓")

        console.clear()
        console.print(
            Panel(
                f"[bold]Actions:[/bold] {action_str}\n"
                f"[bold]Speech:[/bold]  {speech or '(silent)'}\n"
                f"[bold]{mood_emoji} Mood:[/bold]    [{mood_color}]{mood}[/{mood_color}] ({emotion_str})\n"
                f"{pattern_str}"
                f"[bold]Difficulty:[/bold] {diff_str}\n"
                f"[bold]Latency:[/bold]  {latency_ms:.1f}ms | Frame: {self._frame_count}",
                title=f"LNN Brain - {self.character_name}",
                subtitle=f"[{mood_color}]{mood.upper()}[/{mood_color}]",
                border_style=mood_color,
            )
        )

    def _log_plain(
        self,
        actions, action_confidences, speech, mood,
        emotions, patterns, difficulty_status, latency_ms,
    ):
        """Fallback plain text logging (no rich library)."""
        parts = [f"[Frame {self._frame_count}]"]

        if actions:
            active = [str(i) for i, a in enumerate(actions) if a]
            parts.append(f"Actions: [{', '.join(active)}]")

        if speech:
            parts.append(f'Speech: "{speech}"')

        parts.append(f"Mood: {mood}")

        if emotions:
            emo_str = ", ".join(f"{k}={v:.2f}" for k, v in emotions.items())
            parts.append(f"Emotions: {emo_str}")

        parts.append(f"Latency: {latency_ms:.1f}ms")

        print(" | ".join(parts))

    def start(self) -> None:
        """Initialize the debug display."""
        if not self.enabled:
            return
        if HAS_RICH:
            self._console.print("[bold green]Debug Logger Started[/bold green]")
        else:
            print("[DEBUG] Logger started (install 'rich' for enhanced display)")

    def stop(self) -> None:
        """Clean up the debug display."""
        if HAS_RICH and self._console:
            self._console.print("[bold red]Debug Logger Stopped[/bold red]")
        else:
            print("[DEBUG] Logger stopped")
