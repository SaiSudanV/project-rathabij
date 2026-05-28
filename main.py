"""
main.py — LNN Enemy AI Orchestrator
=====================================
The main entry point that wires together:
  - CombatLNN (21.4M param combat brain)
  - LFM2Handler (700M language brain)
  - UDPBridge (Godot ↔ PyTorch communication)
  - VoiceHandler (Moonshine STT + Kokoro TTS)
  - EmotionEngine (mood tracking)
  - PlayerProfiler (habit exploitation)
  - DifficultyScaler (adaptive difficulty)
  - MemoryManager (cross-session persistence)
  - DebugLogger (real-time interpretability)

Run with:
    python main.py
    (then start the Godot game)
"""

from __future__ import annotations
import sys
import time
import signal
import argparse
from pathlib import Path

import torch
import yaml

from model import CombatLNN
from lfm2_handler import LFM2Handler
from udp_bridge import UDPBridge, GamePacket, ActionPacket
from voice_handler import VoiceHandler, VoiceEvent
from emotion_engine import EmotionEngine
from player_profiler import PlayerProfiler as PlayerProfilerTracker
from difficulty_scaler import DifficultyScaler
from memory_manager import MemoryManager
from debug_logger import DebugLogger


class LNNOrchestrator:
    """
    The master controller that runs all AI subsystems.

    Receives game state via UDP → processes through the CombatLNN →
    returns actions via UDP. Simultaneously manages voice, emotions,
    player profiling, and the debug display.
    """

    def __init__(self, config_path: str = "config.yaml", checkpoint: str | None = None):
        print("=" * 60)
        print("  🧠 LNN Enemy AI — Dual Liquid Neural Network")
        print("=" * 60)

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        model_cfg = self.cfg.get("model", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {self.device}")

        # ── Combat Brain ──
        self.model = CombatLNN(
            state_dim=model_cfg.get("state_dim", 64),
            hidden_size=model_cfg.get("hidden_size", 896),
            num_action_slots=model_cfg.get("num_action_slots", 40),
            num_cfc_layers=model_cfg.get("num_cfc_layers", 4),
        ).to(self.device)
        self.model.eval()

        if checkpoint:
            self._load_checkpoint(checkpoint)

        print(f"  Combat Brain: {self.model.param_count_m:.1f}M params")

        # ── Language Brain ──
        self.lfm2 = LFM2Handler(config_path)
        print(f"  Language Brain: LFM2-700M ({self.lfm2.backend})")

        # ── Subsystems ──
        self.bridge = UDPBridge(config_path)
        self.voice = VoiceHandler(config_path)
        self.emotions = EmotionEngine(config_path)
        self.profiler = PlayerProfilerTracker(config_path)
        self.difficulty = DifficultyScaler(config_path)
        self.memory = MemoryManager(config_path)
        self.debug = DebugLogger(config_path)

        # ── Runtime State ──
        self.hx_list: list[torch.Tensor] | None = None
        self._frame_count = 0
        self._running = False

        print("=" * 60)

    def _load_checkpoint(self, path: str) -> None:
        """Load trained weights."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        print(f"  Checkpoint loaded: {path}")

    def _on_state(self, packet: GamePacket) -> ActionPacket:
        """
        Main brain callback — processes a game state and returns actions.
        Called every frame by the UDP bridge.
        """
        self._frame_count += 1

        # ── Build state tensor ──
        gs = packet.game_state
        state_vector = self._build_state_vector(gs)
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).unsqueeze(0).to(self.device)

        # ── Forward pass through CombatLNN ──
        result = self.model.act(state_tensor, hx_list=self.hx_list)

        actions = result["actions"][0].cpu().numpy().astype(int).tolist()
        emotions = result["emotions"][0].cpu()
        profile = result["player_profile"][0].cpu()
        self.hx_list = result["hx_list"]

        # ── Apply difficulty scaling ──
        noise = self.difficulty.get_action_noise()
        if noise > 0:
            import numpy as np
            noise_vec = np.random.normal(0, noise, len(actions))
            logits = result["action_logits"][0].cpu().numpy() + noise_vec
            actions = (torch.sigmoid(torch.FloatTensor(logits)) > 0.5).int().tolist()

        # ── Process events ──
        for event in packet.events:
            self.emotions.record_event(event)
            self.profiler.observe(event, context={
                "distance": gs.get("distance", 0),
                "hp_pct": gs.get("player_hp", 100) / 100,
            })

        # Update emotion state
        enemy_hp_pct = gs.get("enemy_hp", 100) / 100
        emotion_state = self.emotions.update(enemy_hp_pct)

        # ── Generate trash talk (async, non-blocking) ──
        chat_message = ""
        if packet.player_speech_text or self._should_speak(packet.events):
            prompt = self.model.get_context_for_lfm2(
                emotions=emotions,
                profile=profile,
                game_events=packet.events,
                player_speech=packet.player_speech_text,
                score=gs.get("score"),
            )
            chat_message = self.lfm2.generate(prompt, mood=emotion_state.mood.value)

            # Push to voice pipeline
            self.voice.push_event(VoiceEvent(
                event_type="player_spoke" if packet.player_speech_text else "game_event",
                text=packet.player_speech_text,
                game_context={"prompt": prompt, "mood": emotion_state.mood.value},
            ))

        # ── Debug logging ──
        action_confs = torch.sigmoid(result["action_logits"][0]).cpu().tolist()
        patterns = self.profiler.analyze()
        self.debug.log_frame(
            actions=actions,
            action_confidences=action_confs,
            speech=chat_message,
            mood=emotion_state.mood.value,
            emotions=self.emotions.to_tensor_dict(),
            patterns=[{"name": p.name, "confidence": f"{p.confidence:.0%}", "exploit": p.exploit_hint} for p in patterns],
            difficulty_status=self.difficulty.get_status(),
            latency_ms=self.bridge._last_latency_ms,
        )

        return ActionPacket(
            actions=actions,
            chat_message=chat_message,
            emotion=emotion_state.mood.value,
            debug={
                "top_action": ", ".join(self.bridge.translate_actions(actions)[:3]),
                "confidence": max(action_confs) if action_confs else 0,
                "mood": emotion_state.mood.value,
            },
        )

    def _build_state_vector(self, gs: dict) -> list[float]:
        """Convert the game state dict into a 128-dim float vector."""
        vec = [0.0] * 128

        # Positions
        enemy_pos = gs.get("enemy_pos", [0, 0])
        player_pos = gs.get("player_pos", [0, 0])
        vec[0] = enemy_pos[0] / 10.0
        vec[1] = enemy_pos[1] / 10.0
        vec[2] = player_pos[0] / 10.0
        vec[3] = player_pos[1] / 10.0

        # HP
        vec[4] = gs.get("enemy_hp", 100) / 100.0
        vec[5] = gs.get("player_hp", 100) / 100.0

        # Velocities
        e_vel = gs.get("enemy_velocity", [0, 0])
        p_vel = gs.get("player_velocity", [0, 0])
        vec[6] = e_vel[0] / 6.0
        vec[7] = e_vel[1] / 6.0
        vec[8] = p_vel[0] / 6.0
        vec[9] = p_vel[1] / 6.0

        # Distance
        vec[10] = min(1.0, gs.get("distance", 5) / 20.0)

        # Opponent generic state
        vec[12] = float(gs.get("player_is_attacking", False))
        vec[13] = float(gs.get("player_is_blocking", False))
        vec[14] = float(gs.get("player_is_dodging", False))

        # AI Cooldowns
        cooldowns = gs.get("cooldowns", [])
        for i, cd in enumerate(cooldowns[:40]):
            if 15 + i < 128:
                vec[15 + i] = min(1.0, cd / 3.0)

        # EXACT Player Actions (slots 60-99)
        player_actions = gs.get("player_actions", [])
        for i in range(min(40, len(player_actions))):
            vec[60 + i] = float(player_actions[i])

        return vec

    def _should_speak(self, events: list[str]) -> bool:
        """Quick check if events warrant AI speech."""
        speech_events = {"got_kill", "died", "combo_landed", "match_start", "match_end"}
        return bool(set(events) & speech_events)

    def start(self) -> None:
        """Start all subsystems and begin listening."""
        self._running = True

        # Restore session memory
        session = self.memory.load_session(self.device)
        if session and session.get("is_returning"):
            print(f"\n  👋 Returning player detected! (Session #{session['session_count']}, last seen {session['last_seen_ago']})")
            if "hx_list" in session:
                self.hx_list = session["hx_list"]
                print("  🧠 Hidden states restored from last session")

        # Load LFM2
        try:
            self.lfm2.load_model()
        except Exception as e:
            print(f"  ⚠️  LFM2 load failed: {e}")
            print("  ⚠️  Language brain will use fallback responses")

        # Start debug logger
        self.debug.start()

        # Start voice pipeline
        def generate_fn(prompt, mood):
            return self.lfm2.generate(prompt, mood=mood)

        self.voice.start(generate_fn=generate_fn)

        # Start auto-save
        self.memory.start_auto_save(lambda: (
            self.hx_list,
            self.profiler.get_summary(),
            self.emotions.to_tensor_dict(),
            {"wins": 0, "losses": 0},
        ))

        # Start UDP bridge (blocking — this is the main loop)
        print("\n  🎮 Waiting for Godot connection...\n")
        self.bridge.start(on_state_callback=self._on_state, blocking=True)

    def stop(self) -> None:
        """Gracefully shut down all subsystems."""
        self._running = False

        # Final save
        self.memory.save_session(
            self.hx_list,
            self.profiler.get_summary(),
            self.emotions.to_tensor_dict(),
        )
        self.memory.stop_auto_save()

        self.voice.stop()
        self.bridge.stop()
        self.debug.stop()

        print("\n  🧠 LNN Enemy AI shut down. Session saved.")


def main():
    parser = argparse.ArgumentParser(description="LNN Enemy AI Server")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--checkpoint", default=None, help="Path to trained checkpoint")
    args = parser.parse_args()

    # Auto-detect latest checkpoint
    if args.checkpoint is None:
        ckpt_dir = Path("checkpoints/ppo")
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("*.pt"))
            if ckpts:
                args.checkpoint = str(ckpts[-1])
                print(f"  Auto-detected checkpoint: {args.checkpoint}")

    orchestrator = LNNOrchestrator(args.config, args.checkpoint)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n  Shutting down...")
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    orchestrator.start()


if __name__ == "__main__":
    main()
