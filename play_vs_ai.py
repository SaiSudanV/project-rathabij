"""
play_vs_ai.py — Interactive Pygame Arena (Tactical Predator Edition)
=====================================================================
Play a 1v1 side-scrolling combat game against the trained LNN AI.

Controls:
  W/A/S/D      : Move / Jump
  Left Click   : Light Attack
  Right Click  : Heavy Attack
  Q            : Special Attack
  Shift        : Block
  Space        : Dash / Roll
"""

import os
import math
import numpy as np
import torch
import pygame
import sys

from model import CombatLNN
from environment_wrapper import ArenaEnv

# --- Constants ---
WIDTH, HEIGHT = 900, 650
FPS = 60
ARENA_SIZE = 20.0

# Colors
BG_COLOR = (20, 20, 30)
PLAYER_COLOR = (50, 150, 255)
AI_COLOR = (255, 50, 50)
WHITE = (255, 255, 255)
DARK_GRAY = (50, 50, 65)
GREEN = (50, 200, 80)
YELLOW = (255, 220, 50)
ORANGE = (255, 140, 30)
RED = (255, 40, 40)
CYAN = (80, 220, 255)

# Damage flash state
damage_flashes = []  # [(x, y, text, timer, color)]


def world_to_screen(x: float, y: float) -> tuple[int, int]:
    screen_x = int((x + ARENA_SIZE / 2) / ARENA_SIZE * WIDTH)
    screen_y = int(HEIGHT - 80 - (y + ARENA_SIZE / 2) / ARENA_SIZE * (HEIGHT - 120))
    return screen_x, screen_y


def draw_bar(surface, x, y, w, h, pct, color, bg_color=DARK_GRAY, border=True):
    pygame.draw.rect(surface, bg_color, (x, y, w, h))
    pygame.draw.rect(surface, color, (x, y, int(w * max(0, pct)), h))
    if border:
        pygame.draw.rect(surface, WHITE, (x, y, w, h), 1)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("⚔️ LNN Tactical Predator vs YOU")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("Arial", 18, bold=True)
    font_small = pygame.font.SysFont("Arial", 14)
    font_dmg = pygame.font.SysFont("Arial", 22, bold=True)
    font_title = pygame.font.SysFont("Arial", 12)

    print("Loading 5M Step Tactical Predator Checkpoint...")
    model = CombatLNN(state_dim=128, hidden_size=896, num_action_slots=40, num_cfc_layers=4)

    checkpoint_path = "checkpoints/ppo/combat_lnn_final.pt"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Could not find '{checkpoint_path}'!")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    env = ArenaEnv(state_dim=128)
    obs = env.reset()
    hx_list = None

    player_score = 0
    ai_score = 0
    last_player_hp = 100.0
    last_ai_hp = 100.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pressed()

        # --- Player Input ---
        player_actions = np.zeros(40, dtype=int)
        if keys[pygame.K_w]: player_actions[0] = 1
        if keys[pygame.K_s]: player_actions[1] = 1
        if keys[pygame.K_a]: player_actions[2] = 1
        if keys[pygame.K_d]: player_actions[3] = 1
        if mouse[0]: player_actions[4] = 1       # Left Click -> Light
        if mouse[2]: player_actions[10] = 1      # Right Click -> Heavy
        if keys[pygame.K_q]: player_actions[16] = 1  # Q -> Special
        if keys[pygame.K_LSHIFT]: player_actions[22] = 1  # Block
        if keys[pygame.K_SPACE]: player_actions[28] = 1    # Dash

        # --- AI Inference ---
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            result = model.act(obs_tensor, hx_list=hx_list)
        ai_actions = result["actions"][0].numpy().astype(int)
        hx_list = result["hx_list"]

        # --- Step ---
        obs, reward, done, trunc, info = env.step(ai_actions, opponent_actions=player_actions)

        # Side-scroller gravity
        floor_y = -ARENA_SIZE / 2 + 1.0
        env.agent.vy -= 0.8
        env.opponent.vy -= 0.8
        if env.agent.y <= floor_y:
            env.agent.y = floor_y
            if env.agent.vy < 0: env.agent.vy = 0
        if env.opponent.y <= floor_y:
            env.opponent.y = floor_y
            if env.opponent.vy < 0: env.opponent.vy = 0

        # Damage flash effects
        if env.opponent.hp < last_player_hp and last_player_hp - env.opponent.hp > 0.1:
            px, py = world_to_screen(env.opponent.x, env.opponent.y)
            dmg = last_player_hp - env.opponent.hp
            damage_flashes.append((px, py - 30, f"-{dmg:.0f}", 40, RED))
        if env.agent.hp < last_ai_hp and last_ai_hp - env.agent.hp > 0.1:
            ax, ay = world_to_screen(env.agent.x, env.agent.y)
            dmg = last_ai_hp - env.agent.hp
            damage_flashes.append((ax, ay - 30, f"-{dmg:.0f}", 40, CYAN))

        last_player_hp = env.opponent.hp
        last_ai_hp = env.agent.hp

        if done or trunc:
            if not env.agent.alive:
                player_score += 1
            else:
                ai_score += 1
            obs = env.reset()
            hx_list = None
            last_player_hp = 100.0
            last_ai_hp = 100.0

        # ═══════════════════ RENDERING ═══════════════════
        screen.fill(BG_COLOR)

        # Floor
        floor_screen_y = world_to_screen(0, floor_y)[1]
        pygame.draw.rect(screen, (40, 40, 55), (0, floor_screen_y, WIDTH, HEIGHT - floor_screen_y))
        pygame.draw.line(screen, (80, 80, 100), (0, floor_screen_y), (WIDTH, floor_screen_y), 2)

        # Player character
        px, py = world_to_screen(env.opponent.x, env.opponent.y)
        pygame.draw.circle(screen, PLAYER_COLOR, (px, py), 22)
        if env.opponent.is_blocking:
            pygame.draw.circle(screen, WHITE, (px, py), 28, 3)
        if env.opponent.is_attacking:
            pygame.draw.circle(screen, WHITE, (px, py), int(ARENA_SIZE / 20 * 60), 2)
        if env.opponent.is_exhausted:
            txt = font_small.render("EXHAUSTED", True, ORANGE)
            screen.blit(txt, (px - txt.get_width()//2, py + 30))

        # AI character
        ax, ay = world_to_screen(env.agent.x, env.agent.y)
        pygame.draw.circle(screen, AI_COLOR, (ax, ay), 22)
        if env.agent.is_blocking:
            pygame.draw.circle(screen, WHITE, (ax, ay), 28, 3)
        if env.agent.is_attacking:
            pygame.draw.circle(screen, YELLOW, (ax, ay), int(ARENA_SIZE / 20 * 60), 2)
        if env.agent.is_exhausted:
            txt = font_small.render("EXHAUSTED", True, ORANGE)
            screen.blit(txt, (ax - txt.get_width()//2, ay + 30))
        if env.agent.is_recovering:
            txt = font_small.render("RECOVERING", True, YELLOW)
            screen.blit(txt, (ax - txt.get_width()//2, ay + 45))

        # ── HUD ──
        # Player HP
        label = font_big.render("YOU", True, PLAYER_COLOR)
        screen.blit(label, (30, 15))
        draw_bar(screen, 30, 38, 300, 20, env.opponent.hp_pct, PLAYER_COLOR)
        hp_txt = font_small.render(f"{env.opponent.hp:.0f}/100", True, WHITE)
        screen.blit(hp_txt, (180 - hp_txt.get_width()//2, 39))

        # Player Stamina
        stam_label = font_title.render("STA", True, YELLOW)
        screen.blit(stam_label, (30, 60))
        draw_bar(screen, 60, 62, 200, 10, env.opponent.stamina / 100.0, YELLOW)

        # AI HP
        label = font_big.render(f"LNN AI (Combo: {env.agent.combo_counter})", True, AI_COLOR)
        screen.blit(label, (WIDTH - 330, 15))
        draw_bar(screen, WIDTH - 330, 38, 300, 20, env.agent.hp_pct, AI_COLOR)
        hp_txt = font_small.render(f"{env.agent.hp:.0f}/100", True, WHITE)
        screen.blit(hp_txt, (WIDTH - 180 - hp_txt.get_width()//2, 39))

        # AI Stamina
        stam_label = font_title.render("STA", True, YELLOW)
        screen.blit(stam_label, (WIDTH - 330, 60))
        draw_bar(screen, WIDTH - 300, 62, 200, 10,
                 env.agent.stamina / 100.0, YELLOW)

        # Score
        score_txt = font_big.render(f"YOU {player_score} — {ai_score} AI", True, WHITE)
        screen.blit(score_txt, (WIDTH//2 - score_txt.get_width()//2, 15))

        # Damage flash numbers
        new_flashes = []
        for fx, fy, text, timer, color in damage_flashes:
            if timer > 0:
                alpha = min(255, timer * 8)
                txt_surf = font_dmg.render(text, True, color)
                screen.blit(txt_surf, (fx - txt_surf.get_width()//2, fy - (40 - timer)))
                new_flashes.append((fx, fy, text, timer - 1, color))
        damage_flashes.clear()
        damage_flashes.extend(new_flashes)

        # Controls hint
        hint = font_title.render("WASD:Move | LMB:Light | RMB:Heavy | Q:Special | Shift:Block | Space:Dash", True, (100, 100, 120))
        screen.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT - 25))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
