"""
emotion_engine.py — Emotional State Machine
=============================================
Tracks the AI's internal mood based on recent match events.
Outputs emotion labels and modulation parameters for voice
(Kokoro TTS pitch/speed) and combat (aggression scaling).

Moods:
  - neutral:    Default state, balanced play
  - confident:  Winning moderately, calm and assertive
  - cocky:      Dominating, starts taking risks, slow deep voice
  - tilted:     Losing streak, erratic play, fast high voice
  - desperate:  Very low HP, all-in aggression
  - cold:       Focused, calculated, minimal talking
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import yaml


class Mood(Enum):
    NEUTRAL = "neutral"
    CONFIDENT = "confident"
    COCKY = "cocky"
    TILTED = "tilted"
    DESPERATE = "desperate"
    COLD = "cold"


@dataclass
class EmotionState:
    """Current emotional state of the AI."""
    mood: Mood = Mood.NEUTRAL
    aggression: float = 0.5       # 0.0 = passive, 1.0 = full aggro
    confidence: float = 0.5       # 0.0 = no confidence, 1.0 = maximum
    frustration: float = 0.0      # 0.0 = calm, 1.0 = full tilt
    focus: float = 0.7            # 0.0 = scattered, 1.0 = laser focus
    voice_pitch: float = 1.0      # Kokoro TTS pitch multiplier
    voice_speed: float = 1.0      # Kokoro TTS speed multiplier


@dataclass
class MatchEvent:
    """A single event that occurred during the match."""
    event_type: str               # "kill", "death", "damage_dealt", "damage_taken", "combo", "miss"
    timestamp: float              # time.time() when it happened
    value: float = 0.0            # Optional magnitude (e.g., damage amount)


class EmotionEngine:
    """
    Processes a rolling window of match events to determine the
    AI's current emotional state and voice modulation parameters.

    Usage:
        engine = EmotionEngine("config.yaml")
        engine.record_event("kill")
        engine.record_event("death")
        state = engine.update(current_hp_pct=0.4)
        print(state.mood, state.voice_pitch)
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg.get("emotion", {})
        self.tilt_threshold = self.cfg.get("tilt_threshold", -3)
        self.cocky_threshold = self.cfg.get("cocky_threshold", 3)
        self.confident_threshold = self.cfg.get("confident_threshold", 1)
        self.desperate_hp_pct = self.cfg.get("desperate_hp_pct", 0.15)
        self.mood_decay_seconds = self.cfg.get("mood_decay_seconds", 30)

        # Voice modulation
        self.voice_pitch_tilted = self.cfg.get("voice_pitch_tilted", 1.15)
        self.voice_speed_tilted = self.cfg.get("voice_speed_tilted", 1.2)
        self.voice_pitch_cocky = self.cfg.get("voice_pitch_cocky", 0.9)
        self.voice_speed_cocky = self.cfg.get("voice_speed_cocky", 0.85)

        # Event history
        self.events: list[MatchEvent] = []
        self.state = EmotionState()

    def record_event(self, event_type: str, value: float = 0.0) -> None:
        """Record a match event (kill, death, damage_dealt, etc.)."""
        self.events.append(MatchEvent(
            event_type=event_type,
            timestamp=time.time(),
            value=value,
        ))

    def _recent_events(self) -> list[MatchEvent]:
        """Get events within the decay window."""
        cutoff = time.time() - self.mood_decay_seconds
        self.events = [e for e in self.events if e.timestamp > cutoff]
        return self.events

    def _compute_net_score(self) -> float:
        """Compute a rolling net score from recent events."""
        score = 0.0
        for event in self._recent_events():
            if event.event_type == "kill":
                score += 1.0
            elif event.event_type == "death":
                score -= 1.0
            elif event.event_type == "damage_dealt":
                score += 0.1
            elif event.event_type == "damage_taken":
                score -= 0.05
            elif event.event_type == "combo":
                score += 0.5
            elif event.event_type == "miss":
                score -= 0.02
        return score

    def update(self, current_hp_pct: float = 1.0) -> EmotionState:
        """
        Update the emotional state based on recent events and current HP.

        Args:
            current_hp_pct: Current HP as a fraction (0.0 to 1.0).

        Returns:
            Updated EmotionState.
        """
        net_score = self._compute_net_score()

        # ── Determine Mood ──
        if current_hp_pct <= self.desperate_hp_pct:
            mood = Mood.DESPERATE
        elif net_score <= self.tilt_threshold:
            mood = Mood.TILTED
        elif net_score >= self.cocky_threshold:
            mood = Mood.COCKY
        elif net_score >= self.confident_threshold:
            mood = Mood.CONFIDENT
        else:
            mood = Mood.NEUTRAL

        # ── Compute Emotion Dimensions ──
        aggression = min(1.0, max(0.0, 0.5 + net_score * -0.1))
        confidence = min(1.0, max(0.0, 0.5 + net_score * 0.15))
        frustration = min(1.0, max(0.0, -net_score * 0.2))
        focus = min(1.0, max(0.0, 0.7 - abs(net_score) * 0.05))

        # Override for desperate
        if mood == Mood.DESPERATE:
            aggression = 1.0
            frustration = 0.8
            focus = 0.3

        # ── Voice Modulation ──
        if mood == Mood.TILTED:
            voice_pitch = self.voice_pitch_tilted
            voice_speed = self.voice_speed_tilted
        elif mood in (Mood.COCKY, Mood.CONFIDENT):
            voice_pitch = self.voice_pitch_cocky
            voice_speed = self.voice_speed_cocky
        elif mood == Mood.DESPERATE:
            voice_pitch = 1.1
            voice_speed = 1.3
        else:
            voice_pitch = 1.0
            voice_speed = 1.0

        self.state = EmotionState(
            mood=mood,
            aggression=aggression,
            confidence=confidence,
            frustration=frustration,
            focus=focus,
            voice_pitch=voice_pitch,
            voice_speed=voice_speed,
        )
        return self.state

    def reset(self) -> None:
        """Reset all events and return to neutral state."""
        self.events.clear()
        self.state = EmotionState()

    def get_mood_label(self) -> str:
        """Get a human-readable mood label for logging."""
        return self.state.mood.value

    def get_voice_params(self) -> dict[str, float]:
        """Get current voice modulation parameters for Kokoro TTS."""
        return {
            "pitch": self.state.voice_pitch,
            "speed": self.state.voice_speed,
        }

    def to_tensor_dict(self) -> dict[str, float]:
        """Export emotion values for the debug logger."""
        return {
            "aggression": self.state.aggression,
            "confidence": self.state.confidence,
            "frustration": self.state.frustration,
            "focus": self.state.focus,
        }
