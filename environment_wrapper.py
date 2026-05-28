"""
environment_wrapper.py — Tactical Predator Arena
==================================================
Gymnasium-compatible 1v1 headless combat environment with:
  - Stamina system (prevents button mashing)
  - Recovery frames (commitment cost for attacks)
  - Tactical reward shaping (cornering, whiff punishment, counters)
  - Action history observation (AI reads player habits)
  - Numba JIT physics acceleration

State Space (128-dim):
  [0-3]    Positions (agent xy, opponent xy)
  [4-5]    HP (normalized)
  [6-9]    Velocities
  [10]     Distance
  [11]     Angle to opponent
  [12-14]  Opponent state flags (attacking, blocking, dodging)
  [15-54]  Cooldown timers (40 slots)
  [55]     Agent stamina
  [56]     Agent recovery timer
  [57]     Agent exhausted timer
  [58]     Opponent stamina
  [59]     Opponent recovery timer
  [60-99]  Opponent last actions (40 slots)
  [100-103] Wall distances
  [104-119] Opponent action history (4 frames × 4 features)
  [120-127] Reserved
"""

from __future__ import annotations
import math
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


# ── Numba JIT-compiled hot functions ──────────────────────────

def _fast_distance(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx * dx + dy * dy)


def _fast_decay_cooldowns(cooldowns: np.ndarray, dt: float) -> np.ndarray:
    for i in range(len(cooldowns)):
        cooldowns[i] = max(0.0, cooldowns[i] - dt)
    return cooldowns


if HAS_NUMBA:
    _fast_distance = njit(_fast_distance, cache=True)
    _fast_decay_cooldowns = njit(_fast_decay_cooldowns, cache=True)

# Alias for reward logic
_distance = _fast_distance


@dataclass
class Fighter:
    """State of a single fighter in the arena."""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    hp: float = 100.0
    max_hp: float = 100.0
    is_attacking: bool = False
    is_blocking: bool = False
    is_dodging: bool = False
    facing: float = 0.0
    cooldowns: list = field(default_factory=lambda: [0.0] * 40)
    combo_counter: int = 0
    last_actions: list = field(default_factory=list)

    # Stamina system
    stamina: float = 100.0
    max_stamina: float = 100.0
    exhausted_timer: float = 0.0

    # Recovery frames
    recovery_timer: float = 0.0

    # Global attack cooldown (prevents slot-cycling exploit)
    global_attack_cd: float = 0.0

    # Stamina regen delay (stops regen for X seconds after action)
    stamina_regen_delay: float = 0.0

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def hp_pct(self) -> float:
        return self.hp / self.max_hp

    @property
    def is_exhausted(self) -> bool:
        return self.exhausted_timer > 0

    @property
    def is_recovering(self) -> bool:
        return self.recovery_timer > 0


MOVE_SPEED = 2.0
ATTACK_RANGE = 3.0
ATTACK_DAMAGE = {"light": 5.0, "heavy": 15.0, "special": 25.0}

# Stamina costs (defaults, overridden by config)
STAMINA_COSTS = {"light": 10.0, "heavy": 25.0, "special": 40.0, "block": 5.0, "dash": 20.0}
RECOVERY_TIMES = {"light": 0.3, "heavy": 0.8, "special": 1.2}


class ArenaEnv:
    """
    Headless 1v1 arena combat environment with tactical mechanics.

    Usage:
        env = ArenaEnv()
        obs = env.reset()
        for _ in range(1000):
            actions = np.random.randint(0, 2, size=40)
            obs, reward, done, truncated, info = env.step(actions)
            if done:
                obs = env.reset()
    """

    def __init__(self, arena_size: float = 20.0, max_steps: int = 1000,
                 state_dim: int = 128, num_actions: int = 40):
        self.arena_size = arena_size
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Load config
        self._load_config()

        if HAS_GYM:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32)
            self.action_space = spaces.MultiBinary(num_actions)

        self.agent = Fighter()
        self.opponent = Fighter()
        self.steps = 0
        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0
        self.combo_hits = 0

        # Action history buffer (last 4 frames of opponent behavior)
        self.opponent_history = deque(maxlen=4)
        for _ in range(4):
            self.opponent_history.append([0.0, 0.0, 0.0, 0.0])

        # Tracking for counter-attack detection
        self._agent_blocked_last_frame = False

        # Action slot mapping:
        # 0-3: Movement (up, down, left, right)
        # 4-9: Light attacks
        # 10-15: Heavy attacks
        # 16-21: Special attacks
        # 22: Block, 23: Dodge
        # 24-27: Reserved
        # 28-33: Movement abilities (dash, etc.)
        # 34-39: Reserved

    def _load_config(self):
        """Load reward/stamina params from config.yaml with safe defaults."""
        import yaml
        defaults = {
            "reward_kill": 1.0, "reward_death": -1.0,
            "reward_damage_dealt": 0.1, "reward_damage_taken": -0.05,
            "reward_combo_discovery": 0.5, "reward_hunt": -0.03,
            "reward_spam": -0.02, "reward_speed_kill": 5.0,
            "reward_cornering": 0.02, "reward_whiff": -0.03,
            "reward_counter": 0.3, "reward_flawless": 2.0,
            "stamina_light": 10.0, "stamina_heavy": 25.0,
            "stamina_special": 40.0, "stamina_block": 10.0,
            "stamina_dash": 20.0, "stamina_regen": 1.0,
            "reward_whiff": -0.3,  # Devastating whiff penalty
            "threat_radius": 6.0,
        }
        try:
            with open("config.yaml", "r") as f:
                cfg = yaml.safe_load(f).get("training", {})
            for k, v in defaults.items():
                setattr(self, k, cfg.get(k, v))
        except Exception:
            for k, v in defaults.items():
                setattr(self, k, v)

    def reset(self, seed=None) -> np.ndarray:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agent = Fighter(x=-self.arena_size / 4, y=0.0, hp=100.0)
        self.opponent = Fighter(x=self.arena_size / 4, y=0.0, hp=100.0)
        self.steps = 0
        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0
        self.combo_hits = 0
        self._agent_blocked_last_frame = False

        self.opponent_history = deque(maxlen=4)
        for _ in range(4):
            self.opponent_history.append([0.0, 0.0, 0.0, 0.0])

        return self._get_observation()

    def step(self, actions: np.ndarray, opponent_actions: np.ndarray | None = None):
        """Execute one timestep with full tactical reward shaping."""
        reward = -0.01  # Per-step penalty
        self.steps += 1

        # ── Distance & Tactical Rewards ──
        dist = _distance(self.agent.x, self.agent.y,
                         self.opponent.x, self.opponent.y)
        half = self.arena_size / 2

        # 1. Threat Radius: only penalize if OUTSIDE the zone
        if dist > self.threat_radius:
            # Normalize the penalty so it doesn't explode over 1000 steps
            reward += ((self.threat_radius - dist) / self.arena_size) * abs(self.reward_hunt)

        # 2. Cornering: reward if player is near a wall
        player_wall_dist = min(
            self.opponent.x + half, half - self.opponent.x,
            self.opponent.y + half, half - self.opponent.y)
        if player_wall_dist < 3.0:
            reward += self.reward_cornering * (1.0 - player_wall_dist / 3.0)

        # 5. HP Advantage: press harder when winning
        hp_diff = self.agent.hp_pct - self.opponent.hp_pct
        if hp_diff > 0.2 and dist < self.threat_radius:
            reward += 0.01

        # ── Process Actions ──
        agent_damage, spam_count = self._process_agent_actions(actions)
        reward += spam_count * self.reward_spam
        self.agent.last_actions = actions.tolist()

        if opponent_actions is not None:
            opponent_damage, _ = self._process_opponent_custom_actions(opponent_actions)
            self.opponent.last_actions = opponent_actions.tolist()
        else:
            opponent_damage = self._process_opponent_actions()
            self.opponent.last_actions = [0.0] * 40
            if self.opponent.is_attacking: self.opponent.last_actions[4] = 1.0
            if self.opponent.is_blocking: self.opponent.last_actions[22] = 1.0
            if self.opponent.is_dodging: self.opponent.last_actions[23] = 1.0

        # 3. Whiff Punishment: attacked but missed
        if self.agent.is_attacking and agent_damage == 0:
            reward += self.reward_whiff

        # ── Apply Damage ──
        if agent_damage > 0:
            if self.opponent.is_blocking:
                agent_damage *= 0.2
            elif self.opponent.is_dodging:
                if random.random() < 0.7:
                    agent_damage = 0

            self.opponent.hp -= agent_damage
            self.total_damage_dealt += agent_damage
            reward += agent_damage * self.reward_damage_dealt

            # 4. Counter-Attack: blocked last frame, hit this frame
            if self._agent_blocked_last_frame:
                reward += self.reward_counter

            # Combo tracking
            self.agent.combo_counter += 1
            if self.agent.combo_counter >= 3:
                reward += self.reward_combo_discovery
                self.combo_hits += 1
        else:
            self.agent.combo_counter = 0

        if opponent_damage > 0:
            if self.agent.is_blocking:
                opponent_damage *= 0.2
            elif self.agent.is_dodging:
                if random.random() < 0.7:
                    opponent_damage = 0

            self.agent.hp -= opponent_damage
            self.total_damage_taken += opponent_damage
            reward += opponent_damage * self.reward_damage_taken

        # Track if agent blocked this frame (for counter-attack next frame)
        self._agent_blocked_last_frame = (
            self.agent.is_blocking and opponent_damage > 0)

        # ── Update Action History ──
        dx_opp = self.opponent.x - self.agent.x
        moving_toward = 1.0 if (self.opponent.vx * (-dx_opp)) > 0 else 0.0
        self.opponent_history.append([
            float(self.opponent.is_attacking),
            float(self.opponent.is_blocking),
            float(self.opponent.is_dodging),
            moving_toward,
        ])

        # ── Stamina Regen ──
        # Only regen if NOT blocking AND delay has passed
        if not self.agent.is_blocking and self.agent.stamina_regen_delay <= 0:
            self.agent.stamina = min(self.agent.max_stamina,
                                     self.agent.stamina + self.stamina_regen)
        if not self.opponent.is_blocking and self.opponent.stamina_regen_delay <= 0:
            self.opponent.stamina = min(self.opponent.max_stamina,
                                         self.opponent.stamina + self.stamina_regen)

        # Decay delays
        self.agent.stamina_regen_delay = max(0.0, self.agent.stamina_regen_delay - 0.1)
        self.opponent.stamina_regen_delay = max(0.0, self.opponent.stamina_regen_delay - 0.1)

        # Exhaustion recovery
        self.agent.exhausted_timer = max(0.0, self.agent.exhausted_timer - 0.1)
        self.opponent.exhausted_timer = max(0.0, self.opponent.exhausted_timer - 0.1)

        # Recovery frame decay
        self.agent.recovery_timer = max(0.0, self.agent.recovery_timer - 0.1)
        self.opponent.recovery_timer = max(0.0, self.opponent.recovery_timer - 0.1)

        # Global attack cooldown decay
        self.agent.global_attack_cd = max(0.0, self.agent.global_attack_cd - 0.1)
        self.opponent.global_attack_cd = max(0.0, self.opponent.global_attack_cd - 0.1)

        # ── Check Win/Loss ──
        done = False
        if not self.opponent.alive:
            time_bonus = max(0.0, self.reward_speed_kill - (self.steps / 200.0))
            hp_bonus = self.agent.hp_pct * self.reward_flawless
            reward += self.reward_kill + time_bonus + hp_bonus
            done = True
        elif not self.agent.alive:
            reward += self.reward_death
            done = True

        truncated = self.steps >= self.max_steps

        # ── Decay cooldowns ──
        agent_cd = np.array(self.agent.cooldowns, dtype=np.float64)
        opp_cd = np.array(self.opponent.cooldowns, dtype=np.float64)
        self.agent.cooldowns = _fast_decay_cooldowns(agent_cd, 0.1).tolist()
        self.opponent.cooldowns = _fast_decay_cooldowns(opp_cd, 0.1).tolist()

        obs = self._get_observation()
        info = {
            "agent_hp": self.agent.hp, "opponent_hp": self.opponent.hp,
            "damage_dealt": self.total_damage_dealt,
            "damage_taken": self.total_damage_taken,
            "combos": self.combo_hits, "steps": self.steps,
            "agent_stamina": self.agent.stamina,
            "opponent_stamina": self.opponent.stamina,
        }
        return obs, reward, done, truncated, info

    # ── Action Processors ─────────────────────────────────────

    def _process_agent_actions(self, actions: np.ndarray) -> tuple[float, int]:
        """Process AI actions with stamina and recovery. Returns (damage, spam_count)."""
        damage = 0.0
        spam_count = 0
        self.agent.is_attacking = False
        self.agent.is_blocking = False
        self.agent.is_dodging = False

        is_exhausted = self.agent.is_exhausted
        is_recovering = self.agent.is_recovering

        # Movement (always allowed)
        if actions[0]: self.agent.vy += MOVE_SPEED
        if actions[1]: self.agent.vy -= MOVE_SPEED
        if actions[2]: self.agent.vx -= MOVE_SPEED
        if actions[3]: self.agent.vx += MOVE_SPEED

        # Light attacks (slots 4-9) — global cooldown prevents slot cycling
        if not is_exhausted and self.agent.global_attack_cd <= 0:
            for i in range(4, 10):
                if actions[i]:
                    if self.agent.cooldowns[i] <= 0 and self.agent.stamina >= self.stamina_light:
                        self.agent.is_attacking = True
                        self.agent.stamina -= self.stamina_light
                        self.agent.recovery_timer = RECOVERY_TIMES["light"]
                        self.agent.global_attack_cd = 0.8  # Much longer cooldown
                        self.agent.stamina_regen_delay = 1.5 # Wait before regening
                        if self._dist_agent_opp() < ATTACK_RANGE:
                            damage += ATTACK_DAMAGE["light"]
                        self.agent.cooldowns[i] = 0.5
                        break  # Only ONE attack per tick!
                    else:
                        spam_count += 1

            # Heavy attacks (slots 10-15)
            if self.agent.global_attack_cd <= 0:
                for i in range(10, 16):
                    if actions[i]:
                        if self.agent.cooldowns[i] <= 0 and self.agent.stamina >= self.stamina_heavy:
                            self.agent.is_attacking = True
                            self.agent.stamina -= self.stamina_heavy
                            self.agent.recovery_timer = RECOVERY_TIMES["heavy"]
                            self.agent.global_attack_cd = 0.8
                            if self._dist_agent_opp() < ATTACK_RANGE * 1.2:
                                damage += ATTACK_DAMAGE["heavy"]
                            self.agent.cooldowns[i] = 1.5
                            break
                        else:
                            spam_count += 1

            # Special attacks (slots 16-21)
            if self.agent.global_attack_cd <= 0:
                for i in range(16, 22):
                    if actions[i]:
                        if self.agent.cooldowns[i] <= 0 and self.agent.stamina >= self.stamina_special:
                            self.agent.is_attacking = True
                            self.agent.stamina -= self.stamina_special
                            self.agent.recovery_timer = RECOVERY_TIMES["special"]
                            self.agent.global_attack_cd = 1.2
                            if self._dist_agent_opp() < ATTACK_RANGE * 2.0:
                                damage += ATTACK_DAMAGE["special"]
                            self.agent.cooldowns[i] = 3.0
                            break
                        else:
                            spam_count += 1

        # Block (slot 22) — blocked by exhaustion and recovery
        if actions[22] and not is_exhausted and not is_recovering:
            if self.agent.stamina >= self.stamina_block:
                self.agent.is_blocking = True
                self.agent.stamina -= self.stamina_block

        # Dodge (slot 23) — blocked by recovery
        if actions[23] and not is_recovering:
            self.agent.is_dodging = True

        # Dash (slot 28)
        if actions[28] and not is_exhausted:
            if self.agent.cooldowns[28] <= 0 and self.agent.stamina >= self.stamina_dash:
                dx = self.opponent.x - self.agent.x
                dy = self.opponent.y - self.agent.y
                d = max(0.1, math.sqrt(dx*dx + dy*dy))
                self.agent.vx += (dx / d) * MOVE_SPEED * 3
                self.agent.vy += (dy / d) * MOVE_SPEED * 3
                self.agent.stamina -= self.stamina_dash
                self.agent.cooldowns[28] = 2.0
            else:
                spam_count += 1

        # Check exhaustion
        if self.agent.stamina <= 0:
            self.agent.stamina = 0
            self.agent.exhausted_timer = 1.0

        # Physics
        self.agent.x += self.agent.vx * 0.1
        self.agent.y += self.agent.vy * 0.1
        self.agent.vx *= 0.8
        self.agent.vy *= 0.8
        h = self.arena_size / 2
        self.agent.x = max(-h, min(h, self.agent.x))
        self.agent.y = max(-h, min(h, self.agent.y))

        return damage, spam_count

    def _process_opponent_custom_actions(self, actions: np.ndarray) -> tuple[float, int]:
        """Process opponent actions with stamina/recovery (Self-Play mode)."""
        damage = 0.0
        spam_count = 0
        self.opponent.is_attacking = False
        self.opponent.is_blocking = False
        self.opponent.is_dodging = False

        is_exhausted = self.opponent.is_exhausted
        is_recovering = self.opponent.is_recovering

        if actions[0]: self.opponent.vy += MOVE_SPEED
        if actions[1]: self.opponent.vy -= MOVE_SPEED
        if actions[2]: self.opponent.vx -= MOVE_SPEED
        if actions[3]: self.opponent.vx += MOVE_SPEED

        if not is_exhausted and self.opponent.global_attack_cd <= 0:
            for i in range(4, 10):
                if actions[i]:
                    if self.opponent.cooldowns[i] <= 0 and self.opponent.stamina >= self.stamina_light:
                        self.opponent.is_attacking = True
                        self.opponent.stamina -= self.stamina_light
                        self.opponent.recovery_timer = RECOVERY_TIMES["light"]
                        self.opponent.global_attack_cd = 0.8
                        self.opponent.stamina_regen_delay = 1.5
                        if self._dist_agent_opp() < ATTACK_RANGE:
                            damage += ATTACK_DAMAGE["light"]
                        self.opponent.cooldowns[i] = 0.5
                        break
                    else:
                        spam_count += 1

            if self.opponent.global_attack_cd <= 0:
                for i in range(10, 16):
                    if actions[i]:
                        if self.opponent.cooldowns[i] <= 0 and self.opponent.stamina >= self.stamina_heavy:
                            self.opponent.is_attacking = True
                            self.opponent.stamina -= self.stamina_heavy
                            self.opponent.recovery_timer = RECOVERY_TIMES["heavy"]
                            self.opponent.global_attack_cd = 0.8
                            if self._dist_agent_opp() < ATTACK_RANGE * 1.2:
                                damage += ATTACK_DAMAGE["heavy"]
                            self.opponent.cooldowns[i] = 1.5
                            break
                        else:
                            spam_count += 1

        if actions[22] and not is_exhausted and not is_recovering:
            if self.opponent.stamina >= self.stamina_block:
                self.opponent.is_blocking = True
                self.opponent.stamina -= self.stamina_block
        if actions[23] and not is_recovering:
            self.opponent.is_dodging = True

        if self.opponent.stamina <= 0:
            self.opponent.stamina = 0
            self.opponent.exhausted_timer = 1.0

        self.opponent.x += self.opponent.vx * 0.1
        self.opponent.y += self.opponent.vy * 0.1
        self.opponent.vx *= 0.8
        self.opponent.vy *= 0.8
        h = self.arena_size / 2
        self.opponent.x = max(-h, min(h, self.opponent.x))
        self.opponent.y = max(-h, min(h, self.opponent.y))

        return damage, spam_count

    def _process_opponent_actions(self) -> float:
        """Rule-based opponent bot AI."""
        damage = 0.0
        self.opponent.is_attacking = False
        self.opponent.is_blocking = False
        self.opponent.is_dodging = False

        dist = self._dist_agent_opp()
        dx = self.agent.x - self.opponent.x
        dy = self.agent.y - self.opponent.y

        behavior = random.random()
        if behavior < 0.6:
            if dist > 0.1:
                self.opponent.vx += (dx / dist) * MOVE_SPEED * 0.7
                self.opponent.vy += (dy / dist) * MOVE_SPEED * 0.7
        elif behavior < 0.8:
            if dist > 0.1:
                self.opponent.vx -= (dx / dist) * MOVE_SPEED * 0.7
                self.opponent.vy -= (dy / dist) * MOVE_SPEED * 0.7

        if dist < ATTACK_RANGE and self.opponent.cooldowns[0] <= 0:
            roll = random.random()
            if roll < 0.3:
                damage = ATTACK_DAMAGE["light"]
                self.opponent.is_attacking = True
                self.opponent.cooldowns[0] = 0.5  # Add a global cooldown
            elif roll < 0.4:
                damage = ATTACK_DAMAGE["heavy"]
                self.opponent.is_attacking = True
                self.opponent.cooldowns[0] = 1.5  # Add a global cooldown

        if random.random() < 0.15:
            self.opponent.is_blocking = True
        if self.agent.is_attacking and random.random() < 0.2:
            self.opponent.is_dodging = True

        self.opponent.x += self.opponent.vx * 0.1
        self.opponent.y += self.opponent.vy * 0.1
        self.opponent.vx *= 0.8
        self.opponent.vy *= 0.8
        h = self.arena_size / 2
        self.opponent.x = max(-h, min(h, self.opponent.x))
        self.opponent.y = max(-h, min(h, self.opponent.y))
        return damage

    # ── Helpers ────────────────────────────────────────────────

    def _dist_agent_opp(self) -> float:
        return _fast_distance(self.agent.x, self.agent.y,
                              self.opponent.x, self.opponent.y)

    def _distance(self, a: Fighter, b: Fighter) -> float:
        return _fast_distance(a.x, a.y, b.x, b.y)

    def _get_observation(self) -> np.ndarray:
        """Build 128-dim observation vector."""
        half = self.arena_size / 2
        obs = np.zeros(self.state_dim, dtype=np.float32)

        obs[0] = self.agent.x / half
        obs[1] = self.agent.y / half
        obs[2] = self.opponent.x / half
        obs[3] = self.opponent.y / half
        obs[4] = self.agent.hp_pct
        obs[5] = self.opponent.hp_pct
        obs[6] = self.agent.vx / (MOVE_SPEED * 3)
        obs[7] = self.agent.vy / (MOVE_SPEED * 3)
        obs[8] = self.opponent.vx / (MOVE_SPEED * 3)
        obs[9] = self.opponent.vy / (MOVE_SPEED * 3)

        dist = self._dist_agent_opp()
        obs[10] = min(1.0, dist / self.arena_size)
        dx = self.opponent.x - self.agent.x
        dy = self.opponent.y - self.agent.y
        obs[11] = math.atan2(dy, dx) / math.pi

        obs[12] = float(self.opponent.is_attacking)
        obs[13] = float(self.opponent.is_blocking)
        obs[14] = float(self.opponent.is_dodging)

        # Cooldowns (15-54)
        for i, cd in enumerate(self.agent.cooldowns):
            if 15 + i < self.state_dim:
                obs[15 + i] = min(1.0, cd / 3.0)

        # Stamina & recovery (55-59)
        obs[55] = self.agent.stamina / self.agent.max_stamina
        obs[56] = min(1.0, self.agent.recovery_timer / 1.5)
        obs[57] = min(1.0, self.agent.exhausted_timer / 1.0)
        obs[58] = self.opponent.stamina / self.opponent.max_stamina
        obs[59] = min(1.0, self.opponent.recovery_timer / 1.5)

        # Opponent actions (60-99)
        for i in range(min(40, len(self.opponent.last_actions))):
            obs[60 + i] = float(self.opponent.last_actions[i])

        # Wall distances (100-103)
        obs[100] = (self.agent.x + half) / self.arena_size
        obs[101] = (half - self.agent.x) / self.arena_size
        obs[102] = (self.agent.y + half) / self.arena_size
        obs[103] = (half - self.agent.y) / self.arena_size

        # Action history (104-119): 4 frames × 4 features
        for frame_i, frame in enumerate(self.opponent_history):
            for feat_i, val in enumerate(frame):
                idx = 104 + frame_i * 4 + feat_i
                if idx < self.state_dim:
                    obs[idx] = val

        return obs

    def get_events(self) -> list[str]:
        """Get notable events for voice pipeline."""
        events = []
        if self.total_damage_dealt > 0 and not self.opponent.alive:
            events.append("got_kill")
        if self.total_damage_taken > 0 and not self.agent.alive:
            events.append("died")
        if self.combo_hits > 0:
            events.append("combo_landed")
        if self.opponent.hp_pct < 0.2:
            events.append("opponent_low_hp")
        if self.agent.hp_pct < 0.2:
            events.append("low_hp")
        return events
