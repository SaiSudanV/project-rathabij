"""
data_pipeline.py — NLP Dataset Ingestion Pipeline
===================================================
Loads and processes datasets for fine-tuning the LFM2-700M
language brain on gamer trash talk.

Supports:
  1. Hugging Face Hub datasets (download directly)
  2. Custom local datasets (JSON, JSONL, TXT, CSV)
  3. Automatic tokenization and formatting

The pipeline converts raw text into structured prompt-response
pairs suitable for LFM2 fine-tuning with LoRA or full FT.
"""

from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Any, Iterator

import yaml


from dataclasses import dataclass, field


@dataclass
class ChatSample:
    """A single training sample for the language model."""
    situation: str
    mood: str
    response: str
    metadata: dict = field(default_factory=dict)


class DataPipeline:
    """
    Loads and processes NLP datasets for LFM2 fine-tuning.

    Usage:
        pipeline = DataPipeline("config.yaml")

        # Load from Hugging Face
        pipeline.load_huggingface("dffesalbon/dota-2-toxic-chat-data")

        # Load custom local dataset
        pipeline.load_local("datasets/my_trash_talk.jsonl")

        # Get all samples
        for sample in pipeline.get_samples():
            print(sample.situation, "→", sample.response)

        # Export for fine-tuning
        pipeline.export_for_finetuning("datasets/train.jsonl")
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        lang_cfg = cfg.get("language", {})
        self.system_prompt = lang_cfg.get(
            "system_prompt",
            "You are a skilled, adult competitive gamer. Be witty, confident, and savage.",
        )

        self.samples: list[ChatSample] = []
        self.dataset_dir = Path("datasets")
        self.dataset_dir.mkdir(exist_ok=True)

        # Stats
        self._sources: list[str] = []

    def load_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str | None = None,
        max_samples: int | None = None,
    ) -> int:
        """
        Load a dataset from the Hugging Face Hub.

        Args:
            dataset_name: HF dataset identifier (e.g., "dffesalbon/dota-2-toxic-chat-data")
            split:        Dataset split to load.
            text_column:  Column name containing the text. Auto-detected if None.
            max_samples:  Maximum number of samples to load.

        Returns:
            Number of samples loaded.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("[Data] Install datasets: pip install datasets")
            return 0

        print(f"[Data] Loading from HuggingFace: {dataset_name}...")
        try:
            ds = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"[Data] Failed to load {dataset_name}: {e}")
            return 0

        # Auto-detect text column
        if text_column is None:
            text_cols = [c for c in ds.column_names if "text" in c.lower() or "comment" in c.lower() or "message" in c.lower()]
            if text_cols:
                text_column = text_cols[0]
            else:
                text_column = ds.column_names[0]

        print(f"[Data] Using text column: '{text_column}'")

        count = 0
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            text = str(row.get(text_column, "")).strip()
            if len(text) < 3 or len(text) > 200:
                continue  # Skip too short or too long

            self.samples.append(ChatSample(
                situation="general_chat",
                mood="neutral",
                response=text,
                metadata={"source": dataset_name, "index": i},
            ))
            count += 1

        self._sources.append(f"HF:{dataset_name} ({count} samples)")
        print(f"[Data] Loaded {count} samples from {dataset_name}")
        return count

    def load_local(self, path: str) -> int:
        """
        Load a local dataset file.

        Supported formats:
          - .jsonl: One JSON object per line
              {"situation": "got_kill", "mood": "cocky", "response": "Too easy."}
          - .json:  JSON array of objects
          - .txt:   One response per line (situation/mood auto-assigned)
          - .csv:   CSV with columns: situation, mood, response

        Args:
            path: Path to the dataset file.

        Returns:
            Number of samples loaded.
        """
        filepath = Path(path)
        if not filepath.exists():
            print(f"[Data] File not found: {path}")
            return 0

        print(f"[Data] Loading local file: {path}")
        count = 0

        if filepath.suffix == ".jsonl":
            count = self._load_jsonl(filepath)
        elif filepath.suffix == ".json":
            count = self._load_json(filepath)
        elif filepath.suffix == ".txt":
            count = self._load_txt(filepath)
        elif filepath.suffix == ".csv":
            count = self._load_csv(filepath)
        else:
            print(f"[Data] Unsupported format: {filepath.suffix}")
            return 0

        self._sources.append(f"Local:{filepath.name} ({count} samples)")
        print(f"[Data] Loaded {count} samples from {filepath.name}")
        return count

    def _load_jsonl(self, path: Path) -> int:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self.samples.append(ChatSample(
                        situation=obj.get("situation", "general"),
                        mood=obj.get("mood", "neutral"),
                        response=obj.get("response", obj.get("text", "")),
                        metadata=obj.get("metadata", {}),
                    ))
                    count += 1
                except json.JSONDecodeError:
                    continue
        return count

    def _load_json(self, path: Path) -> int:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        count = 0
        for obj in data:
            self.samples.append(ChatSample(
                situation=obj.get("situation", "general"),
                mood=obj.get("mood", "neutral"),
                response=obj.get("response", obj.get("text", "")),
                metadata=obj.get("metadata", {}),
            ))
            count += 1
        return count

    def _load_txt(self, path: Path) -> int:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and len(line) > 2:
                    self.samples.append(ChatSample(
                        situation="general",
                        mood="neutral",
                        response=line,
                    ))
                    count += 1
        return count

    def _load_csv(self, path: Path) -> int:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(ChatSample(
                    situation=row.get("situation", "general"),
                    mood=row.get("mood", "neutral"),
                    response=row.get("response", row.get("text", "")),
                ))
                count += 1
        return count

    def load_directory(self, dir_path: str) -> int:
        """Load all supported files from a directory."""
        dirp = Path(dir_path)
        if not dirp.exists():
            print(f"[Data] Directory not found: {dir_path}")
            return 0

        total = 0
        for ext in [".jsonl", ".json", ".txt", ".csv"]:
            for filepath in dirp.glob(f"*{ext}"):
                total += self.load_local(str(filepath))
        return total

    def get_samples(self) -> list[ChatSample]:
        """Get all loaded samples."""
        return self.samples

    def get_samples_by_situation(self, situation: str) -> list[ChatSample]:
        """Filter samples by game situation."""
        return [s for s in self.samples if s.situation == situation]

    def get_samples_by_mood(self, mood: str) -> list[ChatSample]:
        """Filter samples by mood."""
        return [s for s in self.samples if s.mood == mood]

    def export_for_finetuning(self, output_path: str) -> None:
        """
        Export samples in a format suitable for LFM2 fine-tuning.

        Generates a JSONL file with chat-format entries:
        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
        """
        outpath = Path(output_path)
        outpath.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(outpath, "w", encoding="utf-8") as f:
            for sample in self.samples:
                if not sample.response.strip():
                    continue

                user_prompt = (
                    f"Game situation: {sample.situation}. "
                    f"Your mood: {sample.mood}. "
                    f"Say something."
                )

                entry = {
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": sample.response},
                    ]
                }

                f.write(json.dumps(entry) + "\n")
                count += 1

        print(f"[Data] Exported {count} samples to {output_path}")

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        situations = {}
        moods = {}
        for s in self.samples:
            situations[s.situation] = situations.get(s.situation, 0) + 1
            moods[s.mood] = moods.get(s.mood, 0) + 1

        return {
            "total_samples": len(self.samples),
            "sources": self._sources,
            "situations": situations,
            "moods": moods,
            "avg_response_length": (
                sum(len(s.response) for s in self.samples) / max(1, len(self.samples))
            ),
        }

    def clear(self) -> None:
        """Clear all loaded data."""
        self.samples.clear()
        self._sources.clear()
