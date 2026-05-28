"""
player_profiler.py — Player Habit Tracker & Exploiter
=====================================================
Observes player behavior over a rolling time window and
detects repeating patterns. Once a pattern is detected
with sufficient confidence, it feeds the exploit signal
to the action head.

Tracked patterns:
  - Dodge direction preference (left vs right)
  - Post-attack behavior (retreat vs follow-up)
  - Range preference (close vs mid vs long)
  - Panic behavior at low HP
  - Attack timing patterns
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from collections import deque

import yaml


@dataclass
class PlayerAction:
    """A single observed player action."""
    action_type: str      # "dodge_left", "dodge_right", "attack", "retreat", "approach", etc.
    timestamp: float
    context: dict = field(default_factory=dict)  # e.g., {"hp_pct": 0.3, "distance": 4.2}


@dataclass
class DetectedPattern:
    """A pattern detected in the player's behavior."""
    name: str             # Human-readable pattern name
    confidence: float     # 0.0 to 1.0
    exploit_hint: str     # Suggested counter-action
    sample_count: int     # Number of observations supporting this


class PlayerProfiler:
    """
    Tracks and analyzes player behavior in real-time.

    Usage:
        profiler = PlayerProfiler("config.yaml")
        profiler.observe("dodge_left", context={"after": "heavy_attack"})
        profiler.observe("dodge_left", context={"after": "heavy_attack"})
        profiler.observe("dodge_left", context={"after": "heavy_attack"})
        patterns = profiler.analyze()
        # → [DetectedPattern("dodge_left_after_heavy", confidence=0.85, exploit="attack_right_after_heavy")]
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        prof_cfg = cfg.get("player_profiler", {})
        self.window_seconds = prof_cfg.get("window_seconds", 60)
        self.min_samples = prof_cfg.get("min_samples", 5)
        self.exploit_confidence = prof_cfg.get("exploit_confidence", 0.7)

        self.observations: deque[PlayerAction] = deque(maxlen=500)

        # Counters for specific behaviors
        self._dodge_counts = {"left": 0, "right": 0, "none": 0}
        self._post_attack = {"retreat": 0, "follow_up": 0, "idle": 0}
        self._range_pref = {"close": 0, "mid": 0, "long": 0}
        self._panic_actions = {"flee": 0, "aggro": 0, "freeze": 0}

    def observe(self, action_type: str, context: dict | None = None) -> None:
        """Record a player action observation."""
        self.observations.append(PlayerAction(
            action_type=action_type,
            timestamp=time.time(),
            context=context or {},
        ))
        self._update_counters(action_type, context or {})

    def _update_counters(self, action_type: str, context: dict) -> None:
        """Update behavior counters based on the observed action."""
        # Dodge direction tracking
        if action_type in ("dodge_left", "dodge_right"):
            direction = action_type.split("_")[1]
            self._dodge_counts[direction] += 1
        elif action_type == "no_dodge":
            self._dodge_counts["none"] += 1

        # Post-attack behavior
        if context.get("after") == "attack":
            if action_type in ("retreat", "move_back"):
                self._post_attack["retreat"] += 1
            elif action_type in ("attack", "follow_up"):
                self._post_attack["follow_up"] += 1
            else:
                self._post_attack["idle"] += 1

        # Range preference
        distance = context.get("distance", None)
        if distance is not None:
            if distance < 3.0:
                self._range_pref["close"] += 1
            elif distance < 8.0:
                self._range_pref["mid"] += 1
            else:
                self._range_pref["long"] += 1

        # Panic behavior (when at low HP)
        hp_pct = context.get("hp_pct", 1.0)
        if hp_pct < 0.25:
            if action_type in ("retreat", "move_back", "flee"):
                self._panic_actions["flee"] += 1
            elif action_type in ("attack", "aggro"):
                self._panic_actions["aggro"] += 1
            else:
                self._panic_actions["freeze"] += 1

    def _prune_old(self) -> None:
        """Remove observations outside the analysis window."""
        cutoff = time.time() - self.window_seconds
        while self.observations and self.observations[0].timestamp < cutoff:
            self.observations.popleft()

    def analyze(self) -> list[DetectedPattern]:
        """
        Analyze current observations and return detected patterns.

        Returns:
            List of DetectedPattern objects with confidence >= threshold.
        """
        self._prune_old()
        patterns = []

        # ── Dodge direction bias ──
        total_dodges = sum(self._dodge_counts.values())
        if total_dodges >= self.min_samples:
            for direction in ("left", "right"):
                ratio = self._dodge_counts[direction] / total_dodges
                if ratio >= self.exploit_confidence:
                    opposite = "right" if direction == "left" else "left"
                    patterns.append(DetectedPattern(
                        name=f"dodge_{direction}_preference",
                        confidence=ratio,
                        exploit_hint=f"attack_{opposite}_anticipate",
                        sample_count=total_dodges,
                    ))

        # ── Post-attack behavior ──
        total_post = sum(self._post_attack.values())
        if total_post >= self.min_samples:
            for behavior, count in self._post_attack.items():
                ratio = count / total_post
                if ratio >= self.exploit_confidence:
                    if behavior == "retreat":
                        exploit = "chase_after_their_attack"
                    elif behavior == "follow_up":
                        exploit = "block_then_counter"
                    else:
                        exploit = "punish_idle_window"
                    patterns.append(DetectedPattern(
                        name=f"post_attack_{behavior}",
                        confidence=ratio,
                        exploit_hint=exploit,
                        sample_count=total_post,
                    ))

        # ── Range preference ──
        total_range = sum(self._range_pref.values())
        if total_range >= self.min_samples:
            for rng, count in self._range_pref.items():
                ratio = count / total_range
                if ratio >= self.exploit_confidence:
                    if rng == "close":
                        exploit = "maintain_distance_and_zone"
                    elif rng == "long":
                        exploit = "close_gap_aggressively"
                    else:
                        exploit = "force_awkward_range"
                    patterns.append(DetectedPattern(
                        name=f"range_{rng}_preference",
                        confidence=ratio,
                        exploit_hint=exploit,
                        sample_count=total_range,
                    ))

        # ── Panic behavior ──
        total_panic = sum(self._panic_actions.values())
        if total_panic >= self.min_samples:
            for behavior, count in self._panic_actions.items():
                ratio = count / total_panic
                if ratio >= self.exploit_confidence:
                    if behavior == "flee":
                        exploit = "cut_off_escape_routes"
                    elif behavior == "aggro":
                        exploit = "bait_and_punish_desperation"
                    else:
                        exploit = "pressure_frozen_target"
                    patterns.append(DetectedPattern(
                        name=f"panic_{behavior}",
                        confidence=ratio,
                        exploit_hint=exploit,
                        sample_count=total_panic,
                    ))

        return patterns

    def get_summary(self) -> dict:
        """Get a human-readable summary of the current player profile."""
        patterns = self.analyze()
        return {
            "total_observations": len(self.observations),
            "detected_patterns": [
                {
                    "name": p.name,
                    "confidence": f"{p.confidence:.0%}",
                    "exploit": p.exploit_hint,
                    "samples": p.sample_count,
                }
                for p in patterns
            ],
        }

    def reset(self) -> None:
        """Reset all observations and counters."""
        self.observations.clear()
        self._dodge_counts = {"left": 0, "right": 0, "none": 0}
        self._post_attack = {"retreat": 0, "follow_up": 0, "idle": 0}
        self._range_pref = {"close": 0, "mid": 0, "long": 0}
        self._panic_actions = {"flee": 0, "aggro": 0, "freeze": 0}
