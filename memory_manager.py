"""
memory_manager.py — Persistent Cross-Session Memory
====================================================
Saves and loads the LNN's hidden states and player profiles
between game sessions. This allows the AI to:
  - Remember your playstyle from last time
  - Open with a returning-player taunt ("oh, you're back")
  - Continue adapting from where it left off
"""

from __future__ import annotations
import os
import time
import json
import threading
from pathlib import Path
from typing import Any

import torch
import yaml


class MemoryManager:
    """
    Manages persistent storage of the AI's memory across sessions.

    Stores:
        - CfC hidden states (the AI's temporal memory)
        - Player profile data (detected habits)
        - Emotion history (last known mood)
        - Match statistics (cumulative record)
        - Session metadata (timestamps, play count)

    Usage:
        mm = MemoryManager("config.yaml")
        mm.save_session(hx_list, player_data, emotion_data, match_stats)
        restored = mm.load_session()
        if restored:
            print(f"Welcome back! Last seen {restored['last_seen_ago']}")
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        mem_cfg = cfg.get("memory", {})
        self.enabled = mem_cfg.get("persistent", True)
        self.save_dir = Path(mem_cfg.get("save_dir", "checkpoints/session_memory"))
        self.save_interval = mem_cfg.get("save_interval_seconds", 60)

        # Ensure save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Auto-save timer
        self._auto_save_timer: threading.Timer | None = None
        self._last_save_data: dict | None = None

    def save_session(
        self,
        hx_list: list[torch.Tensor] | None = None,
        player_profile: dict | None = None,
        emotion_data: dict | None = None,
        match_stats: dict | None = None,
    ) -> str:
        """
        Save the current session state to disk.

        Args:
            hx_list:        CfC hidden states (list of tensors).
            player_profile: Player habit data from PlayerProfiler.
            emotion_data:   Current emotion state dict.
            match_stats:    Cumulative match statistics.

        Returns:
            Path to the saved checkpoint.
        """
        if not self.enabled:
            return ""

        save_path = self.save_dir / "latest_session.pt"
        meta_path = self.save_dir / "session_meta.json"

        # Build checkpoint
        checkpoint = {
            "timestamp": time.time(),
            "session_count": self._get_session_count() + 1,
        }

        if hx_list is not None:
            checkpoint["hx_list"] = [h.cpu() for h in hx_list]

        if player_profile is not None:
            checkpoint["player_profile"] = player_profile

        if emotion_data is not None:
            checkpoint["emotion_data"] = emotion_data

        if match_stats is not None:
            checkpoint["match_stats"] = match_stats

        # Save PyTorch checkpoint
        torch.save(checkpoint, save_path)

        # Save human-readable metadata
        meta = {
            "last_saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_saved_unix": time.time(),
            "session_count": checkpoint["session_count"],
            "has_hidden_states": hx_list is not None,
            "has_player_profile": player_profile is not None,
            "has_match_stats": match_stats is not None,
        }

        if match_stats:
            meta["cumulative_stats"] = match_stats

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return str(save_path)

    def load_session(self, device: torch.device = torch.device("cpu")) -> dict | None:
        """
        Load the most recent session from disk.

        Args:
            device: Device to load tensors onto.

        Returns:
            dict with restored data, or None if no session exists.
            Includes:
                - hx_list: restored hidden states
                - player_profile: restored habits
                - emotion_data: last known emotion
                - match_stats: cumulative record
                - session_count: how many times the player has returned
                - last_seen_ago: human-readable time since last session
                - is_returning: True if this is a returning player
        """
        if not self.enabled:
            return None

        save_path = self.save_dir / "latest_session.pt"
        if not save_path.exists():
            return None

        try:
            checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        except Exception:
            return None

        # Compute time since last session
        last_ts = checkpoint.get("timestamp", 0)
        elapsed = time.time() - last_ts
        last_seen_ago = self._format_elapsed(elapsed)

        result = {
            "is_returning": True,
            "session_count": checkpoint.get("session_count", 1),
            "last_seen_ago": last_seen_ago,
            "elapsed_seconds": elapsed,
        }

        # Restore hidden states
        if "hx_list" in checkpoint:
            result["hx_list"] = [h.to(device) for h in checkpoint["hx_list"]]

        # Restore player profile
        if "player_profile" in checkpoint:
            result["player_profile"] = checkpoint["player_profile"]

        # Restore emotion data
        if "emotion_data" in checkpoint:
            result["emotion_data"] = checkpoint["emotion_data"]

        # Restore match stats
        if "match_stats" in checkpoint:
            result["match_stats"] = checkpoint["match_stats"]

        return result

    def start_auto_save(
        self,
        get_state_fn,
    ) -> None:
        """
        Start periodic auto-saving.

        Args:
            get_state_fn: Callable that returns (hx_list, player_profile,
                          emotion_data, match_stats) tuple.
        """
        if not self.enabled:
            return

        def _save_loop():
            try:
                hx, prof, emo, stats = get_state_fn()
                self.save_session(hx, prof, emo, stats)
            except Exception:
                pass
            # Schedule next save
            self._auto_save_timer = threading.Timer(
                self.save_interval, _save_loop
            )
            self._auto_save_timer.daemon = True
            self._auto_save_timer.start()

        _save_loop()

    def stop_auto_save(self) -> None:
        """Stop the auto-save timer."""
        if self._auto_save_timer is not None:
            self._auto_save_timer.cancel()
            self._auto_save_timer = None

    def _get_session_count(self) -> int:
        """Get the current session count from metadata."""
        meta_path = self.save_dir / "session_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                return meta.get("session_count", 0)
            except Exception:
                return 0
        return 0

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Format elapsed time into a human-readable string."""
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"

    def get_returning_player_context(self) -> dict | None:
        """
        Quick check for returning player context.
        Used by the Yapping Controller to generate opening taunts.

        Returns:
            dict with returning player info, or None if new player.
        """
        session = self.load_session()
        if session is None or not session.get("is_returning"):
            return None

        return {
            "times_played": session["session_count"],
            "last_seen": session["last_seen_ago"],
            "has_profile": "player_profile" in session,
        }
