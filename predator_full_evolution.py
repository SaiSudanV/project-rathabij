"""
predator_full_evolution.py — Progress & Analytics Hub (v9.1)
==========================================================
FIX: Added Intra-Arena (x/3) and Avg Reward to HUD.
"""

import pygame
import torch
import numpy as np
import random
import time
import os
import cv2
from environment_complex import ComplexArenaEnv
from biological_lnn import BiologicalLNN

class Checkbox:
    def __init__(self, x, y, label, active=False, enabled=True):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.label = label
        self.active = active
        self.enabled = enabled

    def draw(self, screen, font):
        color = (0, 255, 150) if self.active else (50, 50, 70)
        if not self.enabled: color = (30, 30, 40)
        pygame.draw.rect(screen, color, self.rect, 2)
        if self.active:
            pygame.draw.rect(screen, color, (self.rect.x + 4, self.rect.y + 4, 12, 12))
        label_color = (255, 255, 255) if self.enabled else (100, 100, 100)
        screen.blit(font.render(self.label, True, label_color), (self.rect.x + 30, self.rect.y + 2))

    def handle_event(self, event):
        if self.enabled and event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
                return True
        return False

def save_checkpoint(brain, episodes, path="checkpoints"):
    if not os.path.exists(path): os.makedirs(path)
    filename = os.path.join(path, f"predator_brain_ep{episodes}.pth")
    torch.save(brain.state_dict(), filename)
    print(f"Checkpoint saved: {filename}")

def load_latest_checkpoint(brain, path="checkpoints"):
    if not os.path.exists(path): return 0
    files = [f for f in os.listdir(path) if f.endswith(".pth")]
    if not files: return 0
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    latest = files[-1]
    try:
        brain.load_state_dict(torch.load(os.path.join(path, latest)))
        print(f"Resumed from checkpoint: {latest}")
        return int(latest.split("ep")[1].split(".")[0])
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return 0

def run_full_evolution():
    pygame.init()
    screen = pygame.display.set_mode((1500, 750))
    pygame.display.set_caption("LNN SPATIAL INTELLIGENCE HUB")
    clock = pygame.time.Clock()
    
    grid_size = 11
    out_dim = 40
    in_dim = grid_size * grid_size
    brain = BiologicalLNN(in_dim=in_dim, hid_dim=128, out_dim=out_dim)
    resumed_ep = load_latest_checkpoint(brain)
    env = ComplexArenaEnv(grid_size=grid_size)
    
    # Curriculum Tiers from Spec
    tier1 = ["PLATFORMER", "TOP-DOWN", "DUNGEON", "SIDESCROLLER"]
    tier2 = ["BEAT-EM-UP", "ISOMETRIC", "FIXED-SCREEN", "FLIP-SCREEN"]
    tier3 = ["AUTO-SCROLLER", "FORCED-SCROLLING", "RUN-AND-GUN", "DIAGONAL-SCROLLING"]
    tier4 = ["METROIDVANIA", "PARALLAX-SCROLLING", "SHOOT-EM-UP", "PSEUDO-3D"]
    
    curriculum = tier1 + tier2 + tier3 + tier4
    current_curriculum_idx = 0
    
    total_steps = 0
    episodes = resumed_ep
    goals_reached = 0
    current_ep_reward = 0.0
    total_reward_history = 0.0 # NEW: For AVG REWARD
    avg_reward = 0.0
    
    last_action = torch.zeros(1, out_dim)
    surprise_meter = 0.0
    curiosity_score = 0.0
    epsilon = 0.2
    recall_active = 0
    
    font = pygame.font.SysFont("Consolas", 12)
    bold_font = pygame.font.SysFont("Consolas", 16, bold=True)
    header_font = pygame.font.SysFont("Consolas", 22, bold=True)

    running = True
    paused = False
    
    # DEBUG UI
    chk_debug = Checkbox(20, 20, "DEBUG MODE")
    chk_control = Checkbox(40, 50, "CONTROL AI BODY", enabled=False)
    chk_pan = Checkbox(40, 80, "PAN CAMERA", enabled=False)
    chk_pick = Checkbox(40, 110, "PICKABLE AI", enabled=False)
    
    custom_cam_x = None
    dragging_agent = False
    drag_offset_x = 0
    drag_offset_y = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p: paused = not paused
                if event.key == pygame.K_n: 
                    current_curriculum_idx = (current_curriculum_idx + 1) % len(curriculum)
                    env.reset(genre=curriculum[current_curriculum_idx].lower(), run_number=1)
            
            # Handle Checkbox Events
            if chk_debug.handle_event(event):
                chk_control.enabled = chk_debug.active
                chk_pan.enabled = chk_debug.active
                chk_pick.enabled = chk_debug.active
                if not chk_debug.active:
                    chk_control.active = chk_pan.active = chk_pick.active = False
            
            chk_control.handle_event(event)
            chk_pan.handle_event(event)
            chk_pick.handle_event(event)

            if chk_pick.active and event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    cam_x = custom_cam_x if custom_cam_x is not None else np.clip(env.agent.x - 500, 0, env.world_width - 1000)
                    ax, ay = env.agent.x - cam_x, env.agent.y
                    dist = np.hypot(event.pos[0] - ax, event.pos[1] - ay)
                    if dist < 30:
                        dragging_agent = True
                        drag_offset_x = env.agent.x - (event.pos[0] + cam_x)
                        drag_offset_y = env.agent.y - event.pos[1]
            
            if event.type == pygame.MOUSEBUTTONUP:
                dragging_agent = False

        if not paused:
            if dragging_agent and chk_pick.active:
                mx, my = pygame.mouse.get_pos()
                cam_x = custom_cam_x if custom_cam_x is not None else np.clip(env.agent.x - 500, 0, env.world_width - 1000)
                env.agent.x = mx + cam_x + drag_offset_x
                env.agent.y = my + drag_offset_y
                env.agent.vx = env.agent.vy = 0
                env.agent.rect.x, env.agent.rect.y = env.agent.x - 15, env.agent.y - 15
            
            if chk_pan.active:
                keys = pygame.key.get_pressed()
                if custom_cam_x is None: custom_cam_x = np.clip(env.agent.x - 500, 0, env.world_width - 1000)
                if keys[pygame.K_a]: custom_cam_x -= 10
                if keys[pygame.K_d]: custom_cam_x += 10
                custom_cam_x = np.clip(custom_cam_x, 0, env.world_width - 1000)
            else:
                custom_cam_x = None

            obs = env._get_obs()
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            
            if chk_control.active:
                keys = pygame.key.get_pressed()
                action_vec = np.zeros(out_dim)
                if keys[pygame.K_LEFT]: action_vec[1] = 1.0
                if keys[pygame.K_RIGHT]: action_vec[2] = 1.0
                if keys[pygame.K_UP]: action_vec[3] = 1.0
                if keys[pygame.K_DOWN]: action_vec[4] = 1.0
                action_idx = np.argmax(action_vec) if np.any(action_vec) else 0
            else:
                action_probs = brain(obs_t, last_action, surprise_meter)
                if random.random() < epsilon:
                    action_idx = random.randint(0, out_dim - 1)
                else:
                    action_idx = torch.argmax(action_probs, dim=-1).item()
                action_vec = np.zeros(out_dim); action_vec[action_idx] = 1.0
            
            last_action = torch.from_numpy(action_vec).unsqueeze(0).float()
            
            next_obs, reward_val, done, _, info = env.step(action_vec)
            
            # Progress reward calculation
            dist_to_target = np.linalg.norm([env.agent.x - env.target_rect.x, env.agent.y - env.target_rect.y])
            step_reward = max(0, (3000 - dist_to_target) / 3000.0)
            current_ep_reward += step_reward
            
            total_steps += 1
            if recall_active > 0: recall_active -= 1
            
            next_obs_t = torch.from_numpy(next_obs).unsqueeze(0)
            predicted_grid = brain.predict_next(obs_t, last_action)
            curiosity_score = torch.mean((predicted_grid - next_obs_t)**2).item() * 500.0
            
            surprise_meter = info.get("surprise", 0.0)
            if curiosity_score > 5.0: surprise_meter = max(surprise_meter, 0.5)
            
            if done: 
                episodes += 1
                
                total_reward_history += current_ep_reward
                avg_reward = total_reward_history / episodes
                
                if not info.get("lost", False) and info.get("surprise", 0.0) < 0.1: 
                    goals_reached += 1
                
                current_ep_reward = 0.0
                
                # Intra-Arena Scaling (Deterministic 3-Run Episodes)
                run_number = (episodes % 3) + 1
                if episodes % 3 == 0:
                    current_curriculum_idx = (current_curriculum_idx + 1) % len(curriculum)
                
                if episodes % 10 == 0:
                    save_checkpoint(brain, episodes)
                
                env.reset(genre=curriculum[current_curriculum_idx].lower(), run_number=run_number)

        # RENDER
        screen.fill((5, 5, 12))
        arena_frame = env.render(custom_cam_x=custom_cam_x)
        arena_frame = cv2.cvtColor(arena_frame, cv2.COLOR_BGR2RGB)
        arena_surf = pygame.surfarray.make_surface(np.transpose(arena_frame, (1, 0, 2)))
        screen.blit(arena_surf, (0, 0))
        
        # Debug Checkboxes
        chk_debug.draw(screen, bold_font)
        if chk_debug.active:
            chk_control.draw(screen, font)
            chk_pan.draw(screen, font)
            chk_pick.draw(screen, font)
        
        # HUD Panel
        sx = 1020
        pygame.draw.rect(screen, (15, 15, 25), (1000, 0, 500, 750))
        pygame.draw.line(screen, (0, 255, 150), (1000, 0), (1000, 750), 2)
        
        screen.blit(header_font.render("SPATIAL MEMORY HUB V9.1", True, (0, 255, 150)), (sx, 20))
        
        # 2D OGM
        y_grid = 70
        screen.blit(bold_font.render("MENTAL OCCUPANCY GRID (OGM)", True, (255, 255, 255)), (sx, y_grid))
        grid_data = obs.reshape(grid_size, grid_size)
        cell_size = 18
        for r in range(grid_size):
            for c in range(grid_size):
                val = grid_data[r, c]
                color = (int(val*255), 100, 255-int(val*255)) if val > 0 else (20, 20, 35)
                if r == grid_size//2 and c == grid_size//2: 
                    pygame.draw.rect(screen, (0, 255, 150), (sx + c*cell_size, y_grid + 25 + r*cell_size, cell_size-2, cell_size-2), 1)
                else:
                    pygame.draw.rect(screen, color, (sx + c*cell_size, y_grid + 25 + r*cell_size, cell_size-2, cell_size-2))

        # STATS BLOCK (UPDATED)
        y_stats = 300
        pygame.draw.rect(screen, (25, 25, 40), (sx, y_stats, 460, 100), border_radius=5)
        
        # Goal Compass (Directional Arrow)
        goal_pos = env.target_rect.x + 30
        dx_to_goal = goal_pos - env.agent.x
        compass_color = (0, 255, 0) if abs(dx_to_goal) < 500 else (255, 255, 0)
        
        # Draw Arrow Base
        pygame.draw.circle(screen, (30, 30, 30), (sx + 125, y_stats + 80), 25)
        # Draw Arrow pointing direction
        arrow_dir = 1 if dx_to_goal > 0 else -1
        arrow_pts = [
            (sx + 125 + arrow_dir*15, y_stats + 80),
            (sx + 125 - arrow_dir*10, y_stats + 80 - 10),
            (sx + 125 - arrow_dir*10, y_stats + 80 + 10)
        ]
        pygame.draw.polygon(screen, compass_color, arrow_pts)
        dist_str = f"DIST: {abs(dx_to_goal):.0f}m"
        screen.blit(bold_font.render(dist_str, True, compass_color), (sx + 160, y_stats + 70))

        # Intra-Arena Progress (x/3)
        run_prog = (episodes % 3) + 1
        screen.blit(bold_font.render(f"RUN PROGRESS: {run_prog}/3", True, (255, 255, 0)), (sx+10, y_stats+10))
        screen.blit(bold_font.render(f"GENRE: {curriculum[current_curriculum_idx]}", True, (0, 255, 255)), (sx+250, y_stats+10))
        
        screen.blit(bold_font.render(f"GOALS REACHED: {goals_reached}", True, (0, 255, 100)), (sx+10, y_stats+35))
        screen.blit(bold_font.render(f"AVG REWARD: {avg_reward:.2f}", True, (255, 150, 0)), (sx+250, y_stats+35))
        
        screen.blit(bold_font.render(f"LIFETIME STEPS: {total_steps:,}", True, (255, 255, 255)), (sx+10, y_stats+60))
        screen.blit(bold_font.render(f"CURIOSITY: {curiosity_score:.4f}", True, (0, 255, 255)), (sx+250, y_stats+60))

        # MOTOR CORTEX
        y_act = 420
        screen.blit(bold_font.render("MOTOR CORTEX (40 BUTTONS)", True, (255, 255, 255)), (sx, y_act))
        for i in range(out_dim):
            col, row = i % 10, i // 10
            val = last_action[0, i].item()
            color = (0, 255, 150) if val > 0.5 else (30, 30, 50)
            pygame.draw.rect(screen, color, (sx + col*45, y_act + 25 + row*15, 40, 10))
            if val > 0.5: screen.blit(font.render(f"B{i}", True, (255, 255, 255)), (sx + col*45, y_act + 25 + row*15 - 12))

        # BRAIN STATE
        y_pl = 620
        p_label = "BRAIN: PLASTIC (SURPRISE)" if surprise_meter > 0.5 else "BRAIN: CRYSTALLIZED"
        screen.blit(bold_font.render(p_label, True, (255, 100, 100) if surprise_meter>0.5 else (0, 255, 150)), (sx, y_pl))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    run_full_evolution()
