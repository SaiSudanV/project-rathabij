"""
difficulty_scaler.py — Adaptive Difficulty System
=================================================
Secretly adjusts the AI's effective skill level to keep
matches competitive and engaging. The player should never
know this is happening.

Mechanism:
  - Tracks the AI's win rate over a sliding window.
  - If the AI wins too much, it increases its reaction delay
    (making it slightly slower to respond).
  - If the AI loses too much, it decreases reaction delay
    (making it faster and more precise).
  - The target is a configurable win rate (default 50%).
"""

from __future__ import annotations
from collections import deque

import yaml


class DifficultyScaler:
    """
    Adaptive difficulty controller.

    Usage:
        scaler = DifficultyScaler("config.yaml")
        scaler.record_result(won=True)
        scaler.record_result(won=False)
        delay = scaler.get_reaction_delay_ms()
        noise = scaler.get_action_noise()
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        diff_cfg = cfg.get("difficulty", {})
        self.enabled = diff_cfg.get("enabled", True)
        self.target_winrate = diff_cfg.get("target_winrate", 0.5)
        self.reaction_min_ms = diff_cfg.get("reaction_delay_min_ms", 50)
        self.reaction_max_ms = diff_cfg.get("reaction_delay_max_ms", 300)
        self.adjustment_speed = diff_cfg.get("adjustment_speed", 0.05)

        # Sliding window of match results (True = AI won)
        self.results: deque[bool] = deque(maxlen=20)

        # Current difficulty level: 0.0 = easiest, 1.0 = hardest
        self._difficulty = 0.5

    @property
    def difficulty(self) -> float:
        return self._difficulty

    def record_result(self, won: bool) -> None:
        """Record a match result."""
        if not self.enabled:
            return
        self.results.append(won)
        self._adjust()

    def _adjust(self) -> None:
        """Adjust difficulty based on recent win rate."""
        if len(self.results) < 3:
            return

        current_winrate = sum(self.results) / len(self.results)
        error = current_winrate - self.target_winrate

        # If AI is winning too much (error > 0), decrease difficulty
        # If AI is losing too much (error < 0), increase difficulty
        self._difficulty -= error * self.adjustment_speed
        self._difficulty = max(0.0, min(1.0, self._difficulty))

    def get_reaction_delay_ms(self) -> float:
        """
        Get the current reaction delay in milliseconds.

        Higher difficulty = lower delay (faster reactions).
        Lower difficulty = higher delay (slower reactions).
        """
        if not self.enabled:
            return self.reaction_min_ms

        # Invert: difficulty 1.0 → min delay, difficulty 0.0 → max delay
        delay_range = self.reaction_max_ms - self.reaction_min_ms
        delay = self.reaction_max_ms - (self._difficulty * delay_range)
        return delay

    def get_action_noise(self) -> float:
        """
        Get noise level to inject into action logits.

        Lower difficulty = more noise (less precise actions).
        Higher difficulty = less noise (more precise actions).

        Returns:
            Noise standard deviation (0.0 to 0.5).
        """
        if not self.enabled:
            return 0.0
        return (1.0 - self._difficulty) * 0.5

    def get_aim_accuracy(self) -> float:
        """
        Get aim accuracy multiplier.

        1.0 = perfect accuracy, 0.5 = misses half the time.
        """
        if not self.enabled:
            return 1.0
        return 0.5 + (self._difficulty * 0.5)

    def get_status(self) -> dict:
        """Get human-readable difficulty status for debug logging."""
        winrate = sum(self.results) / len(self.results) if self.results else 0.5
        return {
            "enabled": self.enabled,
            "difficulty": f"{self._difficulty:.2f}",
            "reaction_delay_ms": f"{self.get_reaction_delay_ms():.0f}",
            "action_noise": f"{self.get_action_noise():.3f}",
            "aim_accuracy": f"{self.get_aim_accuracy():.0%}",
            "recent_winrate": f"{winrate:.0%}",
            "target_winrate": f"{self.target_winrate:.0%}",
            "matches_tracked": len(self.results),
        }

    def reset(self) -> None:
        """Reset difficulty to default."""
        self.results.clear()
        self._difficulty = 0.5
