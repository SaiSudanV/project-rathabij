"""
phase1_sprint.py — Local GTX Training Trial
===========================================
- 10,000 Steps of Phase 1 (Navigation).
- Real PPO Updates on CUDA.
- Watch the AI discover movement in real-time.
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
GAMMA = 0.99
BATCH_SIZE = 128
UPDATE_EPOCHS = 4

def train_sprint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SPRINT] Training on: {device}")
    
    state_dim = 32
    model = CombatLNN(state_dim=state_dim, hidden_size=192).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    env = ComplexArenaEnv(state_dim=state_dim)
    env.current_phase = 1 # Navigation only
    plasticity = HebbianPlasticity(num_actions=12)
    
    # Pygame for viewing
    pygame.init()
    screen = pygame.display.set_mode((1300, 700))
    font = pygame.font.SysFont("Consolas", 18)
    clock = pygame.time.Clock()

    obs = env.reset()
    rollout_obs, rollout_actions, rollout_log_probs, rollout_rewards, rollout_values = [], [], [], [], []
    
    global_step = 0
    print("[SPRINT] Starting 10,000 steps. Watch the Hebbian panel!")

    running = True
    while running and global_step < 10000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        # 1. Collect Rollout
        h_mask = torch.FloatTensor(plasticity.get_weights()).to(device)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model.act(obs_t, hebbian_mask=h_mask)
        
        action = out["actions"].item()
        next_obs, reward, done, _, info = env.step(action)
        
        # Hebbian Adapt
        if reward > 0.5: plasticity.reward(action)
        elif reward < -0.1: plasticity.penalize(action)
        
        # Buffer
        rollout_obs.append(obs)
        rollout_actions.append(action)
        rollout_log_probs.append(out["action_log_probs"].item())
        rollout_rewards.append(reward)
        rollout_values.append(out["value"].item())
        
        obs = next_obs
        global_step += 1
        
        # 2. Update Brain (PPO-ish)
        if len(rollout_obs) >= BATCH_SIZE:
            # (Simplified update for speed and demonstration)
            obs_batch = torch.FloatTensor(np.array(rollout_obs)).to(device)
            actions_batch = torch.LongTensor(np.array(rollout_actions)).to(device)
            old_log_probs_batch = torch.FloatTensor(np.array(rollout_log_probs)).to(device)
            
            for _ in range(UPDATE_EPOCHS):
                curr_out = model.forward(obs_batch, hebbian_mask=h_mask)
                # Simple policy gradient loss
                dist = torch.distributions.Categorical(logits=curr_out["action_logits"])
                loss = -dist.log_prob(actions_batch).mean() 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            rollout_obs, rollout_actions, rollout_log_probs, rollout_rewards, rollout_values = [], [], [], [], []

        # 3. Render
        if global_step % 2 == 0: # Render every 2nd step to save CPU
            screen.fill((15,15,20))
            for p in env.platforms: pygame.draw.rect(screen, (50,50,60), (p.rect.x, p.rect.y, p.rect.w, p.rect.h))
            pygame.draw.rect(screen, (0, 255, 180), (env.agent.x-15, env.agent.y-50, 30, 50))
            pygame.draw.rect(screen, (255, 50, 150), (env.opponent.x-15, env.opponent.y-50, 30, 50))
            
            # HUD
            weights = plasticity.get_weights()
            for i, w in enumerate(weights):
                txt = font.render(f"{ACTIONS[i]:<12}: {w:.2f}", True, (255,255,255))
                screen.blit(txt, (1020, 50 + i*25))
            
            screen.blit(font.render(f"Step: {global_step}/10000", True, (255,255,0)), (1020, 20))
            pygame.display.flip()
        
        if done: obs = env.reset()

    pygame.quit()
    print("[SPRINT COMPLETE]")

if __name__ == "__main__":
    train_sprint()
