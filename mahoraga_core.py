"""
mahoraga_core.py — Mahoraga Wheel + Hebbian Plasticity + Predator Choke
========================================================================
Updated for 12-action Categorical action space (v2).
"""

import math
import numpy as np
import json
import os

try:
    import pygame
except ImportError:
    pygame = None


class MahoragaWheel:
    """Live Adaptation Wheel with Persistence."""

    def __init__(self, adaptation_threshold=3, save_path="checkpoints/mahoraga_wheel.json"):
        self.save_path = save_path
        self.signatures = {}
        self.adaptation_threshold = adaptation_threshold
        self.spokes = 8
        self.rotation = 0.0
        self.active_spokes = 0
        self.load()

    def _hash_projectile(self, p):
        vx_bin = round(abs(p.vx) / 3.0)
        vy_bin = round(abs(p.vy) / 3.0)
        w_bin = round(p.radius / 5.0)
        return f"{vx_bin}_{vy_bin}_{w_bin}"

    def record_damage(self, projectiles, agent_x, agent_y):
        hit_proj = None
        min_d = 9999
        for p in projectiles:
            d = math.hypot(p.x - agent_x, p.y - agent_y)
            if d < 120 and d < min_d:
                min_d = d
                hit_proj = p

        if hit_proj:
            sig = self._hash_projectile(hit_proj)
            self.signatures[sig] = self.signatures.get(sig, 0) + 1
            self.active_spokes = min(self.spokes, sum(self.signatures.values()))
            if self.signatures[sig] == self.adaptation_threshold:
                print(f"MAHORAGA WHEEL CLICK! Adapted to: {sig}")
                self.rotation += 45
                self.save()
                return True
        return False

    def is_adapted(self, projectiles, agent_x, agent_y):
        for p in projectiles:
            sig = self._hash_projectile(p)
            if self.signatures.get(sig, 0) >= self.adaptation_threshold:
                if math.hypot(p.x - agent_x, p.y - agent_y) < 300:
                    self.rotation += 4
                    return True
        return False

    def save(self):
        if not self.save_path: return
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(self.signatures, f)

    def load(self):
        if not self.save_path: return
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r") as f:
                    self.signatures = json.load(f)
                self.active_spokes = min(self.spokes, sum(self.signatures.values()))
            except: pass

    def draw(self, screen, x, y):
        if pygame is None: return
        radius = 18
        color = (255, 230, 150) if self.active_spokes >= 1 else (100, 100, 130)
        pygame.draw.circle(screen, color, (x, y), radius, 2)
        for i in range(self.spokes):
            angle = math.radians(self.rotation + (i * 360 / self.spokes))
            sx = int(x + math.cos(angle) * radius)
            sy = int(y + math.sin(angle) * radius)
            spoke_color = (255, 255, 255) if i < self.active_spokes else (80, 80, 100)
            pygame.draw.line(screen, spoke_color, (x, y), (sx, sy), 2)


class HebbianPlasticity:
    """
    Persistent Action-Modulator for Categorical actions (v2).
    Tracks per-action bias for 12 discrete actions.
    """

    def __init__(self, num_actions=12, save_path="checkpoints/hebbian_weights.npy"):
        self.save_path = save_path
        self.num_actions = num_actions
        # Discovery Bias: 0.0 = Neutral, -3.0 = Inhibited, +3.0 = Preferred
        self.weights = np.zeros(num_actions, dtype=np.float32)
        self.load()

    def penalize(self, action_id=None):
        if action_id is not None and 0 <= action_id < self.num_actions:
            self.weights[action_id] -= 0.2
            self.weights = np.clip(self.weights, -3.0, 3.0)
            self.save()

    def reward(self, action_id=None):
        if action_id is not None and 0 <= action_id < self.num_actions:
            self.weights[action_id] += 0.1
            self.weights = np.clip(self.weights, -3.0, 3.0)
            self.save()

    def save(self):
        if not self.save_path: return
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        np.save(self.save_path, self.weights)

    def load(self):
        if not self.save_path: return
        if os.path.exists(self.save_path):
            try:
                loaded = np.load(self.save_path)
                if loaded.shape == (self.num_actions,):
                    self.weights = loaded
                else:
                    self.weights = np.zeros(self.num_actions, dtype=np.float32)
                    self.save()
            except:
                self.weights = np.zeros(self.num_actions, dtype=np.float32)

    def get_weights(self):
        """Returns the current discovery mask."""
        return self.weights.copy()

    def modulate(self, logits):
        """Additive bias for categorical actions."""
        return logits + self.weights


class PredatorChoke:
    """HP-based Flow State triggering."""

    @staticmethod
    def get_modifiers(hp_pct):
        if hp_pct < 0.4:
            return {"noise": 0.0, "cd_reduction": 0.3, "glow": True}
        return {"noise": 0.05, "cd_reduction": 0.0, "glow": False}
