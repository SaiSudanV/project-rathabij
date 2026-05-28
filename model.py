"""
model.py — Combat LNN (v2.1 Predator Rebuild)
==============================================
- REMOVED Emotion Module (Always Cold/Calculated).
- Expanded CfC Backbone to 192 hidden units for higher density.
- Categorical Action Head (12-slot).
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from lnn_cell import LiquidStack

# ── Action Definitions ─────────────────────────────────────────
ACTIONS = [
    "idle",           # 0
    "move_left",      # 1
    "move_right",     # 2
    "jump",           # 3
    "jump_left",      # 4
    "jump_right",     # 5
    "melee",          # 6
    "ki_blast",       # 7
    "ranged_attack",  # 8
    "block",          # 9
    "dash_left",      # 10
    "dash_right",     # 11
]
NUM_ACTIONS = len(ACTIONS)

class StateEncoder(nn.Module):
    """32 -> 192 dense features with residual GELU layers."""
    def __init__(self, state_dim: int, hidden_size: int):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(state)
        return projected + self.net(projected)

class ActionHead(nn.Module):
    def __init__(self, hidden_size: int, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_actions),
        )
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
    def sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist.log_prob(actions)
    def log_prob(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return Categorical(logits=logits).log_prob(actions.long())
    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        return Categorical(logits=logits).entropy()

class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

class CombatLNN(nn.Module):
    """
    Predator Brain: Purely Tactical.
    ~0.9M Params | 192 Hidden | No Emotions.
    """
    def __init__(self, state_dim=32, hidden_size=192, num_actions=NUM_ACTIONS, num_cfc_layers=2, dropout=0.05):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.state_encoder = StateEncoder(state_dim, hidden_size)
        self.backbone = LiquidStack(hidden_size, hidden_size, num_cfc_layers, dropout)
        self.action_head = ActionHead(hidden_size, num_actions)
        self.value_head = ValueHead(hidden_size)
        self._param_count = sum(p.numel() for p in self.parameters())

    @property
    def param_count(self) -> int: return self._param_count
    @property
    def param_count_m(self) -> float: return self._param_count / 1e6

    def forward(self, game_state, hx_list=None, timespans=None, hebbian_mask=None):
        if game_state.dim() == 2: game_state = game_state.unsqueeze(1)
        state_features = self.state_encoder(game_state)
        backbone_out, hx_list_new = self.backbone(state_features, hx_list, timespans)
        last_hidden = backbone_out[:, -1, :]
        
        logits = self.action_head(last_hidden)
        # ── Hebbian Discovery Bias ──────────────────────────────────
        if hebbian_mask is not None:
            logits = logits + hebbian_mask
            
        return {
            "action_logits": logits,
            "value": self.value_head(last_hidden),
            "hx_list": hx_list_new,
            "hidden": last_hidden
        }

    @torch.no_grad()
    def act(self, game_state, hx_list=None, timespans=None, hebbian_mask=None):
        out = self.forward(game_state, hx_list, timespans, hebbian_mask)
        actions, log_probs = self.action_head.sample(out["action_logits"])
        return {
            "actions": actions,
            "action_log_probs": log_probs,
            "action_logits": out["action_logits"],
            "value": out["value"],
            "hx_list": out["hx_list"],
            "hidden": out["hidden"]
        }
