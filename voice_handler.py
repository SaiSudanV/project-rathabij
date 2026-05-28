"""
voice_handler.py — Real-Time Voice Pipeline
=============================================
Handles the full voice-to-voice loop:
  1. Moonshine STT: Listens to microphone, transcribes to text
  2. LFM2 Language Brain: Generates contextual trash talk
  3. Kokoro TTS: Converts text response to spoken audio
  4. Yapping Controller: Manages when the AI speaks unprompted

The pipeline runs in a separate thread so it never blocks
the game loop or combat brain.
"""

from __future__ import annotations
import time
import random
import threading
import queue
from dataclasses import dataclass
from typing import Callable, Any

import yaml
import numpy as np


@dataclass
class VoiceEvent:
    """An event that may trigger the AI to speak."""
    event_type: str       # "player_spoke", "got_kill", "died", "match_start", etc.
    text: str = ""        # Player's transcribed speech (if applicable)
    game_context: dict | None = None  # Current game state summary


class YappingController:
    """
    Manages WHEN the AI speaks unprompted.

    Uses configurable probabilities and cooldowns to prevent
    the AI from being annoying while ensuring it speaks at
    impactful moments.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        yap_cfg = cfg.get("yapping", {})
        self.enabled = yap_cfg.get("enabled", True)
        self.cooldown = yap_cfg.get("global_cooldown_seconds", 15)

        # Trigger probabilities
        self.probs = {
            "kill": yap_cfg.get("prob_on_kill", 0.9),
            "death": yap_cfg.get("prob_on_death", 0.7),
            "player_miss": yap_cfg.get("prob_on_player_miss", 0.1),
            "low_hp_survive": yap_cfg.get("prob_on_low_hp_survive", 0.6),
            "combo_landed": yap_cfg.get("prob_on_combo_landed", 0.8),
            "match_start": yap_cfg.get("prob_on_match_start", 1.0),
            "match_end": yap_cfg.get("prob_on_match_end", 1.0),
            "player_returns": yap_cfg.get("prob_on_player_returns", 1.0),
        }

        self._last_speak_time = 0.0

    def should_speak(self, event_type: str) -> bool:
        """
        Decide if the AI should speak for this event.

        Always speaks when the player directly talks to it.
        For unprompted events, checks cooldown + probability.
        """
        if not self.enabled:
            return False

        # Always respond to direct player speech
        if event_type == "player_spoke":
            return True

        # Check cooldown
        elapsed = time.time() - self._last_speak_time
        if elapsed < self.cooldown:
            return False

        # Roll probability
        prob = self.probs.get(event_type, 0.05)
        return random.random() < prob

    def mark_spoke(self) -> None:
        """Record that the AI just spoke (reset cooldown)."""
        self._last_speak_time = time.time()

    def override_cooldown(self) -> None:
        """Force-reset cooldown (for high-priority events)."""
        self._last_speak_time = 0.0


class VoiceHandler:
    """
    Full voice pipeline manager.

    Runs STT, language generation, and TTS in a background thread.
    The game loop pushes VoiceEvents into a queue, and this handler
    processes them asynchronously.

    Usage:
        handler = VoiceHandler("config.yaml")
        handler.start(generate_fn=my_lfm2_generate_function)

        # From game loop:
        handler.push_event(VoiceEvent("got_kill"))
        handler.push_event(VoiceEvent("player_spoke", text="nice shot"))

        # Cleanup:
        handler.stop()
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        voice_cfg = cfg.get("voice", {})
        self.stt_model_name = voice_cfg.get("stt_model", "moonshine-tiny")
        self.tts_voice = voice_cfg.get("tts_voice", "af_heart")
        self.tts_speed = voice_cfg.get("tts_speed", 1.0)
        self.sample_rate = voice_cfg.get("sample_rate", 16000)

        emotion_cfg = cfg.get("emotion", {})
        self._voice_pitch_tilted = emotion_cfg.get("voice_pitch_tilted", 1.15)
        self._voice_speed_tilted = emotion_cfg.get("voice_speed_tilted", 1.2)
        self._voice_pitch_cocky = emotion_cfg.get("voice_pitch_cocky", 0.9)
        self._voice_speed_cocky = emotion_cfg.get("voice_speed_cocky", 0.85)

        # Components
        self.yapping = YappingController(config_path)
        self._event_queue: queue.Queue[VoiceEvent] = queue.Queue(maxsize=10)
        self._running = False
        self._thread: threading.Thread | None = None

        # STT and TTS modules (lazy loaded)
        self._stt = None
        self._tts = None

        # Callbacks
        self._generate_fn: Callable | None = None  # LFM2 generate function
        self._on_speech_heard: Callable | None = None  # Callback when player speaks

        # Stats
        self._total_utterances = 0
        self._last_stt_text = ""
        self._last_tts_text = ""
        self._is_listening = False

    def _init_stt(self) -> None:
        """Initialize Moonshine STT."""
        try:
            import moonshine
            self._stt = moonshine.load(self.stt_model_name)
            print(f"[Voice] Moonshine STT loaded: {self.stt_model_name}")
        except ImportError:
            print("[Voice] Moonshine not installed. Install with: pip install useful-moonshine")
            print("[Voice] STT will be disabled — AI will only speak unprompted.")
        except Exception as e:
            print(f"[Voice] Moonshine init error: {e}")

    def _init_tts(self) -> None:
        """Initialize Kokoro TTS."""
        try:
            import kokoro
            self._tts = kokoro.KPipeline(lang_code="a")
            print(f"[Voice] Kokoro TTS loaded: voice={self.tts_voice}")
        except ImportError:
            print("[Voice] Kokoro not installed. Install with: pip install kokoro")
            print("[Voice] TTS will be disabled — responses will be text-only.")
        except Exception as e:
            print(f"[Voice] Kokoro init error: {e}")

    def start(
        self,
        generate_fn: Callable[[str, str], str],
        on_speech_heard: Callable[[str], None] | None = None,
    ) -> None:
        """
        Start the voice pipeline.

        Args:
            generate_fn:     Function(prompt, mood) -> str
                             Typically CombatLNN.get_context_for_lfm2 + LFM2Handler.generate
            on_speech_heard: Optional callback when player speech is transcribed.
        """
        self._generate_fn = generate_fn
        self._on_speech_heard = on_speech_heard

        self._init_stt()
        self._init_tts()

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

        # Start mic listener in another thread
        if self._stt is not None:
            self._is_listening = True
            self._mic_thread = threading.Thread(target=self._mic_listener, daemon=True)
            self._mic_thread.start()

        print("[Voice] Pipeline started")

    def push_event(self, event: VoiceEvent) -> None:
        """Push a voice event from the game loop (non-blocking)."""
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            pass  # Drop events if queue is full (prevent backup)

    def _process_loop(self) -> None:
        """Main event processing loop (runs in background thread)."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=0.5)
                self._handle_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Voice] Error processing event: {e}")

    def _handle_event(self, event: VoiceEvent) -> None:
        """Process a single voice event."""
        # Check if we should speak
        if not self.yapping.should_speak(event.event_type):
            return

        if self._generate_fn is None:
            return

        # Generate response via LFM2
        mood = event.game_context.get("mood", "neutral") if event.game_context else "neutral"
        prompt = event.game_context.get("prompt", "") if event.game_context else ""

        if not prompt:
            # Build a basic prompt if none provided
            if event.text:
                prompt = f'Opponent said: "{event.text}". Respond.'
            else:
                prompt = f"Game event: {event.event_type}. React."

        response = self._generate_fn(prompt, mood)

        if response:
            self._last_tts_text = response
            self._total_utterances += 1
            self.yapping.mark_spoke()

            # Speak it via TTS
            self._speak(response, mood)

    def _speak(self, text: str, mood: str = "neutral") -> None:
        """Convert text to speech via Kokoro TTS."""
        if self._tts is None:
            print(f"[Voice] [TTS disabled] AI says: \"{text}\"")
            return

        try:
            # Adjust voice parameters based on mood
            speed = self.tts_speed
            if mood == "tilted":
                speed *= self._voice_speed_tilted
            elif mood in ("cocky", "confident"):
                speed *= self._voice_speed_cocky

            # Generate audio
            generator = self._tts(
                text,
                voice=self.tts_voice,
                speed=speed,
            )

            # Play audio segments
            import sounddevice as sd
            for _, _, audio in generator:
                if audio is not None:
                    sd.play(audio, samplerate=24000)
                    sd.wait()

        except Exception as e:
            print(f"[Voice] TTS error: {e}")
            print(f"[Voice] AI says: \"{text}\"")

    def _mic_listener(self) -> None:
        """
        Continuously listen to the microphone and transcribe.
        Pushes player_spoke events when speech is detected.
        """
        try:
            import sounddevice as sd
        except ImportError:
            print("[Voice] sounddevice not installed for mic input")
            return

        print("[Voice] Mic listener active — speak to trash talk back!")

        # Record in chunks
        chunk_duration = 3.0  # seconds
        chunk_samples = int(self.sample_rate * chunk_duration)

        while self._running and self._is_listening:
            try:
                # Record audio chunk
                audio = sd.rec(
                    chunk_samples,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()

                # Check if there's actual speech (simple energy threshold)
                energy = np.abs(audio).mean()
                if energy < 0.01:
                    continue  # Silence, skip

                # Transcribe via Moonshine
                text = self._transcribe(audio.flatten())
                if text and len(text.strip()) > 2:
                    self._last_stt_text = text
                    print(f"[Voice] Player said: \"{text}\"")

                    # Notify callback
                    if self._on_speech_heard:
                        self._on_speech_heard(text)

                    # Push event
                    self.push_event(VoiceEvent(
                        event_type="player_spoke",
                        text=text,
                    ))

            except Exception as e:
                if self._running:
                    print(f"[Voice] Mic error: {e}")
                time.sleep(1.0)

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio array to text via Moonshine STT."""
        if self._stt is None:
            return ""

        try:
            import moonshine
            result = moonshine.transcribe(self._stt, audio)
            if isinstance(result, list) and result:
                return result[0]
            return str(result) if result else ""
        except Exception as e:
            print(f"[Voice] STT error: {e}")
            return ""

    def stop(self) -> None:
        """Stop the voice pipeline."""
        self._running = False
        self._is_listening = False
        if self._thread:
            self._thread.join(timeout=3.0)
        print("[Voice] Pipeline stopped")

    def get_stats(self) -> dict:
        """Get pipeline statistics for debug logging."""
        return {
            "running": self._running,
            "listening": self._is_listening,
            "stt_loaded": self._stt is not None,
            "tts_loaded": self._tts is not None,
            "total_utterances": self._total_utterances,
            "last_heard": self._last_stt_text,
            "last_said": self._last_tts_text,
        }
