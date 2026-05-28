"""
biological_lnn.py — Spatial Brain (v8.1)
=======================================
FIX: Processes 11x11 Grid (121 dims). Predicts Spatial Map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BiologicalLNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.in_dim = in_dim   # Spatial Grid (121)
        self.out_dim = out_dim # Actions (40)
        
        # 1. LNN Core
        self.W_tau = nn.Parameter(torch.ones(hid_dim))
        self.W_sys = nn.Linear(in_dim + out_dim + hid_dim, hid_dim)
        
        # 2. Hebbian Readout
        self.hebbian_weights = nn.Parameter(torch.randn(hid_dim, out_dim) * 0.1)
        
        # 3. Spatial Forward Model (Grid Prediction)
        self.forward_model = nn.Sequential(
            nn.Linear(in_dim + out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, in_dim) # Predicts 121 grid cells
        )
        
        self.prev_state = torch.zeros(1, hid_dim)

    def forward(self, obs, action_feedback, surprise_signal):
        # x is the 121-dim Spatial Grid
        x = torch.cat([obs, action_feedback], dim=-1)
        combined = torch.cat([x, self.prev_state], dim=-1)
        
        derivative = torch.tanh(self.W_sys(combined))
        tau = torch.sigmoid(self.W_tau)
        self.prev_state = self.prev_state + tau * derivative
        
        logits = self.prev_state @ self.hebbian_weights
        action_probs = torch.softmax(logits, dim=-1)
        
        # One-Shot Hebbian Rewiring based on Spatial Layout
        if surprise_signal > 0.5:
            with torch.no_grad():
                update = torch.ger(self.prev_state.squeeze(), action_probs.squeeze())
                self.hebbian_weights.data += 0.5 * surprise_signal * update
                
        return action_probs

    def predict_next(self, obs, action):
        return self.forward_model(torch.cat([obs, action], dim=-1))
