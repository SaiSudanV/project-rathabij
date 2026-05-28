"""
trial_run_gtx.py — The "Deep Debug" Suite
=========================================
- 1 Environment | 150 Steps Total (50 per phase).
- Pygame Visualization (30 FPS).
- Live Hebbian & Action Debugging.
- Uses CUDA (GTX GPU).
"""

import torch
import pygame
import numpy as np
import time
from model import CombatLNN, ACTIONS
from environment_complex import ComplexArenaEnv, ACT_IDLE
from mahoraga_core import HebbianPlasticity

# ── Colors ───────────────────────────────────────────────────
CLR_BG = (15, 15, 20)
CLR_PLAT = (50, 50, 60)
CLR_AGENT = (0, 255, 180)
CLR_OPP = (255, 50, 150)
CLR_TEXT = (220, 220, 220)

def run_trial():
    # 1. Setup GPU & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRIAL] Using Device: {device}")
    
    state_dim = 32
    model = CombatLNN(state_dim=state_dim, hidden_size=192).to(device)
    env = ComplexArenaEnv(state_dim=state_dim)
    plasticity = HebbianPlasticity(num_actions=12)
    
    # 2. Pygame Setup
    pygame.init()
    screen = pygame.display.set_mode((1300, 700)) # Extra width for stats
    pygame.display.set_caption("Mahoraga Predator — GTX Trial Run")
    font = pygame.font.SysFont("Consolas", 18)
    clock = pygame.time.Clock()

    obs = env.reset()
    total_steps = 0
    phase_steps = 0
    current_phase = 1
    
    print("[TRIAL] Starting 150-step loop...")
    
    running = True
    while running and total_steps < 150:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        # 3. Phase Transition Logic
        if phase_steps >= 50:
            current_phase += 1
            phase_steps = 0
            env.current_phase = current_phase
            print(f"\n[TRANSITION] Entering Phase {current_phase}")
        
        # 4. Action Logic (With Hebbian Mask)
        h_mask = torch.FloatTensor(plasticity.get_weights()).to(device)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model.act(obs_t, hebbian_mask=h_mask)
        
        action = out["actions"].item()
        
        # 5. Step Env
        obs, reward, done, _, info = env.step(action)
        
        # 6. Hebbian Adaptation
        if reward > 0.5: plasticity.reward(action)
        elif reward < -0.1: plasticity.penalize(action)
        
        # 7. Render (Custom Pygame)
        screen.fill(CLR_BG)
        
        # Draw Platforms
        for p in env.platforms:
            pygame.draw.rect(screen, CLR_PLAT, (p.rect.x, p.rect.y, p.rect.w, p.rect.h))
        
        # Draw Entities
        pygame.draw.rect(screen, CLR_AGENT, (env.agent.x-15, env.agent.y-50, 30, 50))
        pygame.draw.rect(screen, CLR_OPP, (env.opponent.x-15, env.opponent.y-50, 30, 50))
        
        # Draw Stats (Right Panel)
        pygame.draw.rect(screen, (30, 30, 40), (1000, 0, 300, 700))
        stats = [
            f"Step: {total_steps}",
            f"Phase: {current_phase} ({phase_steps}/50)",
            f"Reward: {reward:.4f}",
            "",
            "HEBBIAN WEIGHTS:",
        ]
        # Show all 12 action weights
        weights = plasticity.get_weights()
        for i, w in enumerate(weights):
            color = (100, 255, 100) if w > 0.7 else (255, 100, 100) if w < 0.3 else (200, 200, 200)
            stats.append(f" {ACTIONS[i]:<12}: {w:.2f}")
        
        for i, text in enumerate(stats):
            line = font.render(text, True, CLR_TEXT if ":" not in text else (255, 255, 255))
            screen.blit(line, (1020, 20 + i*25))

        pygame.display.flip()
        
        # 8. Counter
        total_steps += 1
        phase_steps += 1
        clock.tick(30) # 30 FPS for clear view

        if done: obs = env.reset()

    pygame.quit()
    print("\n[TRIAL COMPLETE]")

if __name__ == "__main__":
    run_trial()
