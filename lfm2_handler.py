"""
lfm2_handler.py — Liquid AI LFM2-700M Language Brain
=====================================================
Manages the LFM2-700M foundation model for generating
contextual, adult-level trash talk during gameplay.

Supports two backends:
  1. llama-cpp-python: Direct GGUF loading (recommended for GTX 1650)
  2. ollama: If the user has ollama installed with the model pulled

The CombatLNN feeds structured context (emotions, events, score,
player speech) into a prompt, and this handler generates the
actual text response.
"""

from __future__ import annotations
import os
import time
import threading
from pathlib import Path
from typing import Any

import yaml

# Fix for Windows: llama-cpp-python crashes if CUDA_PATH points to
# a nonexistent directory. Clear it if it doesn't exist.
_cuda_path = os.environ.get("CUDA_PATH", "")
if _cuda_path and not os.path.isdir(os.path.join(_cuda_path, "bin")):
    os.environ.pop("CUDA_PATH", None)


class LFM2Handler:
    """
    Interface to the Liquid AI LFM2-700M language model.

    Usage:
        handler = LFM2Handler("config.yaml")
        handler.load_model()

        response = handler.generate(
            prompt="You are a pro gamer...",
            mood="cocky",
        )
        print(response)  # "Three kills isn't luck, it's a pattern."
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        lang_cfg = cfg.get("language", {})
        self.backend = lang_cfg.get("backend", "lfm2")
        self.model_name = lang_cfg.get("model_name", "LiquidAI/LFM2-700M-GGUF")
        self.quantization = lang_cfg.get("quantization", "Q4_K_M")
        self.max_tokens = lang_cfg.get("max_tokens", 60)
        self.temperature = lang_cfg.get("temperature", 0.8)
        self.system_prompt = lang_cfg.get(
            "system_prompt",
            "You are a skilled, adult competitive gamer. Be witty, confident, and savage. No emojis. No caps spam. Sound like a pro.",
        )

        self._model = None
        self._loaded = False
        self._lock = threading.Lock()

        # Performance tracking
        self._last_gen_time_ms = 0.0
        self._total_generations = 0

    def load_model(self) -> None:
        """
        Load the LFM2 model based on the configured backend.

        For llama-cpp-python:
            Downloads/loads the GGUF file and initializes the model.

        For ollama:
            Verifies the model is available via the ollama API.
        """
        if self.backend == "ollama":
            self._load_ollama()
        else:
            self._load_llama_cpp()

        self._loaded = True

    def _load_llama_cpp(self) -> None:
        """Load via llama-cpp-python for direct GPU inference."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required. Install with:\n"
                "  pip install llama-cpp-python\n"
                "For GPU support on Windows:\n"
                "  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )

        # Attempt to find or download the GGUF file
        model_dir = Path("checkpoints/lfm2")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Check if GGUF file exists locally
        gguf_files = list(model_dir.glob("*.gguf"))

        if gguf_files:
            model_path = str(gguf_files[0])
            print(f"[LFM2] Loading local GGUF: {model_path}")
        else:
            # Download from Hugging Face
            print(f"[LFM2] Downloading {self.model_name} ({self.quantization})...")
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename=f"*{self.quantization}*.gguf",
                    local_dir=str(model_dir),
                )
            except Exception as e:
                print(f"[LFM2] Auto-download failed: {e}")
                print(f"[LFM2] Please manually download the GGUF to: {model_dir}/")
                print(f"[LFM2] Falling back to ollama backend...")
                self.backend = "ollama"
                self._load_ollama()
                return

        self._model = Llama(
            model_path=model_path,
            n_ctx=2048,          # Context window
            n_gpu_layers=-1,     # Offload all layers to GPU
            n_threads=4,         # CPU threads for non-GPU ops
            verbose=False,
        )
        print(f"[LFM2] Model loaded successfully via llama-cpp-python")

    def _load_ollama(self) -> None:
        """Verify ollama is available and model is pulled."""
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                print(f"[LFM2] Ollama available. Models: {models}")
                # Store the model tag for ollama
                self._ollama_model = self.model_name.split("/")[-1].lower()
            else:
                raise ConnectionError("Ollama not responding")
        except Exception as e:
            print(f"[LFM2] Ollama not available: {e}")
            print("[LFM2] Please install ollama and run: ollama pull liquidai/lfm2")
            self._model = None

    def generate(
        self,
        prompt: str,
        mood: str = "neutral",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a trash-talk response.

        Args:
            prompt:      Structured prompt from CombatLNN.get_context_for_lfm2()
            mood:        Current mood label (affects generation style)
            temperature: Override temperature (None = use config default)
            max_tokens:  Override max tokens (None = use config default)

        Returns:
            Generated text response (the trash talk).
        """
        if not self._loaded:
            return self._fallback_response(mood)

        temp = temperature or self.temperature
        max_tok = max_tokens or self.max_tokens

        start = time.time()

        with self._lock:
            if self.backend == "ollama":
                result = self._generate_ollama(prompt, temp, max_tok)
            else:
                result = self._generate_llama_cpp(prompt, temp, max_tok)

        self._last_gen_time_ms = (time.time() - start) * 1000
        self._total_generations += 1

        # Clean up the response
        result = self._clean_response(result)
        return result

    def _generate_llama_cpp(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate via llama-cpp-python."""
        if self._model is None:
            return self._fallback_response("neutral")

        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["\n\n", "---"],
        )

        return response["choices"][0]["message"]["content"]

    def _generate_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate via ollama API."""
        try:
            import requests
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self._ollama_model,
                    "prompt": f"{self.system_prompt}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
        except Exception as e:
            print(f"[LFM2] Ollama generation error: {e}")

        return self._fallback_response("neutral")

    def _clean_response(self, text: str) -> str:
        """Clean up the generated text."""
        # Remove any system prompt leakage
        text = text.strip()
        # Remove quotes if the model wrapped it
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        # Truncate to first sentence or two
        sentences = text.split(". ")
        if len(sentences) > 2:
            text = ". ".join(sentences[:2]) + "."
        return text

    def _fallback_response(self, mood: str) -> str:
        """
        Emergency fallback if LFM2 is unavailable.
        Returns a pre-written response based on mood.
        """
        fallbacks = {
            "cocky": [
                "You're making this too easy.",
                "I've seen better plays from bots.",
                "That all you got?",
            ],
            "tilted": [
                "Whatever, that was lag.",
                "You got lucky. Run it back.",
                "That won't happen again.",
            ],
            "confident": [
                "Good fight, but I'm still up.",
                "You're adapting. Not fast enough though.",
                "Getting closer. Still not there.",
            ],
            "desperate": [
                "I'm not done yet.",
                "One more round. Let's go.",
                "This isn't over.",
            ],
            "cold": [
                "Noted.",
                "Interesting.",
                "Predictable.",
            ],
            "neutral": [
                "Let's see what you've got.",
                "Next round.",
                "Focus.",
            ],
        }

        import random
        options = fallbacks.get(mood, fallbacks["neutral"])
        return random.choice(options)

    def get_stats(self) -> dict:
        """Get generation statistics for debug logging."""
        return {
            "backend": self.backend,
            "model": self.model_name,
            "loaded": self._loaded,
            "total_generations": self._total_generations,
            "last_gen_ms": f"{self._last_gen_time_ms:.0f}",
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded
