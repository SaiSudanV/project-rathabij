"""
predator_long_training.py — The 1M Step Evolution Dashboard
============================================================
- 1,000,000 Step Navigation Training.
- Live Reward Meter + Brain Activity Monitor.
- Advanced Hebbian Pruning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import numpy as np
import time
from model import CombatLNN, ACTIONS
from environment_complex import ComplexArenaEnv
from mahoraga_core import HebbianPlasticity

# ── Hyperparameters ──────────────────────────────────────────
LR = 3e-4
BATCH_SIZE = 256
UPDATE_EVERY = 512

class TrainingMonitor:
    def __init__(self):
        self.reward_history = []
        self.avg_reward = 0.0
        self.max_reward = -999
        self.brain_activity = np.zeros(192) # Visualizing 192 hidden units

    def update(self, r, hidden_state):
        self.reward_history.append(r)
        if len(self.reward_history) > 100: self.reward_history.pop(0)
        self.avg_reward = np.mean(self.reward_history)
        self.max_reward = max(self.max_reward, r)
        # Hidden state activity (norm)
        self.brain_activity = hidden_state[0].detach().cpu().numpy()

def run_evolution():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombatLNN(state_dim=32, hidden_size=192).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    env = ComplexArenaEnv(state_dim=32)
    env.current_phase = 1
    plasticity = HebbianPlasticity(num_actions=12)
    monitor = TrainingMonitor()

    pygame.init()
    screen = pygame.display.set_mode((1300, 800))
    pygame.display.set_caption("Predator Evolution — 1M Step Navigation")
    font = pygame.font.SysFont("Consolas", 16)
    bold_font = pygame.font.SysFont("Consolas", 22, bold=True)
    
    obs = env.reset()
    buffer = []
    global_step = 0
    total_reward_epoch = 0

    print("[EVOLUTION] Commencing 1M Step Phase 1 Training...")
    
    running = True
    while running and global_step < 1000000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        # 1. BRAIN INFERENCE
        h_mask = torch.FloatTensor(plasticity.get_weights()).to(device)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        out = model.act(obs_t, hebbian_mask=h_mask)
        
        action = out["actions"].item()
        next_obs, reward, done, _, info = env.step(action)
        total_reward_epoch += reward
        
        # 2. HEBBIAN DISCOVERY
        if reward > 0.4: plasticity.reward(action)
        elif reward < -0.05: plasticity.penalize(action)
        
        # 3. STORE & UPDATE
        buffer.append((obs, action, out["action_log_probs"].item(), reward))
        obs = next_obs
        global_step += 1
        
        if len(buffer) >= BATCH_SIZE:
            # Training logic (Simplified PPO Update)
            b_obs = torch.FloatTensor(np.array([x[0] for x in buffer])).to(device)
            b_acts = torch.LongTensor(np.array([x[1] for x in buffer])).to(device)
            for _ in range(2):
                res = model.forward(b_obs, hebbian_mask=h_mask)
                dist = torch.distributions.Categorical(logits=res["action_logits"])
                loss = -dist.log_prob(b_acts).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            buffer = []

        # 4. MONITORING & DASHBOARD
        if global_step % 5 == 0:
            monitor.update(total_reward_epoch if done else reward, out["hidden"])
            if done: total_reward_epoch = 0
            
            screen.fill((10, 10, 15))
            
            # --- Draw Arena ---
            pygame.draw.rect(screen, (30, 30, 40), (50, 50, 900, 600))
            for p in env.platforms:
                pygame.draw.rect(screen, (60, 60, 70), (50+p.rect.x*0.9, 50+p.rect.y*0.85, p.rect.w*0.9, p.rect.h*0.85))
            pygame.draw.rect(screen, (0, 255, 180), (50+env.agent.x*0.9-10, 50+env.agent.y*0.85-40, 20, 40))
            pygame.draw.rect(screen, (255, 50, 150), (50+env.opponent.x*0.9-10, 50+env.opponent.y*0.85-40, 20, 40))

            # --- Sidebar: Hebbian Discoveries ---
            weights = plasticity.get_weights()
            screen.blit(bold_font.render("BRAIN INTENT (12)", True, (255, 255, 255)), (1000, 30))
            for i, w in enumerate(weights):
                color = (100, 255, 100) if w > 0.5 else (255, 100, 100) if w < -0.5 else (200, 200, 200)
                txt = font.render(f"{ACTIONS[i]:<14}: {w:>5.2f}", True, color)
                screen.blit(txt, (1000, 70 + i*22))

            # --- Physical Body: 40 Unlabeled Slots ---
            screen.blit(bold_font.render("PHYSICAL BODY (40 SLOTS)", True, (255, 255, 255)), (1000, 380))
            mapping = env.dna['mapping']
            # Highlight the slot the AI just pressed
            active_slot = mapping.get(action, -1)
            
            for s in range(40):
                row, col = divmod(s, 5)
                bx, by = 1000 + col*50, 420 + row*40
                # Color based on activity
                b_color = (0, 255, 180) if s == active_slot else (40, 40, 50)
                pygame.draw.rect(screen, b_color, (bx, by, 45, 35))
                # Small label for us
                s_txt = font.render(f"{s}", True, (100, 100, 100))
                screen.blit(s_txt, (bx+5, by+5))

            # --- Bottom: Brain Activity ---
            screen.blit(bold_font.render("LNN BRAIN ACTIVITY (192 NEURONS)", True, (255, 255, 255)), (50, 680))
            for i, val in enumerate(monitor.brain_activity):
                intensity = int(np.clip(abs(val) * 255, 0, 255))
                pygame.draw.rect(screen, (intensity, intensity, intensity), (50 + i*6, 710, 4, 30))

            # --- HUD Stats ---
            stats = [
                f"STEP: {global_step:,}",
                f"AVG REWARD: {monitor.avg_reward:.4f}",
                f"MAX REWARD: {monitor.max_reward:.4f}",
                f"CURRICULUM: Phase 1 (Navigation)"
            ]
            for i, s in enumerate(stats):
                screen.blit(font.render(s, True, (255, 255, 0)), (50, 20 + i*20))

            pygame.display.flip()
        
        if done: obs = env.reset()

    pygame.quit()

if __name__ == "__main__":
    run_evolution()
