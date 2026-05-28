"""
play_vs_ai_complex.py — Complex Platformer Arena (Full Rewrite)
================================================================
Features:
  - Full platformer physics with gravity, platforms, wall collision
  - Projectiles (Ki Blast, Dismantle slash, AI ranged)
  - Pickups (Health / Stamina orbs)
  - Hiding pillars that block projectiles
  - Fallback "Bot AI" when no trained model is available

Controls:
  W         : Jump
  A / D     : Move Left / Right
  LMB       : Melee Attack
  RMB       : Shield / Block
  Q         : Ki Blast (ranged projectile)
  E         : DISMANTLE (Sukuna secret attack — fast, high damage)
"""

import os
import math
import random
import numpy as np
import torch
import pygame
import sys
from collections import deque
from mahoraga_core import MahoragaWheel, HebbianPlasticity, PredatorChoke
from debug_logger import DebugLogger

from environment_complex import (
    ComplexArenaEnv, SCREEN_W, SCREEN_H, FLOOR_Y,
    ENTITY_W, ENTITY_H, MELEE_RANGE,
)

# ── Colors ──
BG_TOP = (10, 10, 25)
BG_BOT = (20, 15, 35)
PLAT_COLOR = (55, 55, 70)
PLAT_BORDER = (90, 90, 110)
PILLAR_COLOR = (45, 45, 60)
PLAYER_BODY = (30, 160, 255)
PLAYER_OUTLINE = (80, 200, 255)
AI_BODY = (220, 40, 40)
AI_OUTLINE = (255, 100, 100)
WHITE = (255, 255, 255)
SHIELD_COLOR = (200, 220, 255, 120)
KI_COLOR = (255, 255, 80)
DISMANTLE_COLOR = (255, 20, 120)
AI_PROJ_COLOR = (255, 120, 30)
HP_GREEN = (50, 220, 80)
STAM_YELLOW = (255, 210, 50)
PICKUP_HP = (0, 255, 100)
PICKUP_STAM = (255, 220, 0)
FLOOR_COLOR = (35, 35, 50)
HUD_BG = (20, 20, 30, 180)
DMG_PLAYER = (100, 200, 255)
DMG_AI = (255, 80, 80)

FPS = 60


class BotAI:
    """
    Simple rule-based bot for testing when no trained model exists.
    It chases the player, attacks in range, jumps to reach platforms,
    and sometimes uses ranged attacks and shields.
    """
    def __init__(self):
        self.frame = 0

    def act(self, env):
        self.frame += 1
        actions = np.zeros(40, dtype=int)
        agent = env.agent
        player = env.opponent

        dx = player.x - agent.x
        dy = (player.y - ENTITY_H) - (agent.y - ENTITY_H) # Relative height of heads
        dist = math.hypot(dx, dy)

        # ── 1. Basic Movement: Chase player ──
        if dx < -30:
            actions[2] = 1  # Move left
        elif dx > 30:
            actions[3] = 1  # Move right

        # ── 2. Stuck Detection & Vertical Navigation ──
        moving_intent = actions[2] or actions[3]
        
        # If we are horizontally close but vertically far, we MUST jump to climb
        vertical_gap = abs(dy) > 50
        horizontal_aligned = abs(dx) < 60
        
        if (horizontal_aligned and vertical_gap) or (moving_intent and abs(agent.vx) < 0.5):
            if agent.is_grounded:
                actions[0] = 1 # JUMP!
        
        # ── 3. Platform Navigation (Look Ahead) ──
        if vertical_gap:
            # Find any platform that might help us get higher
            for p in env.platforms:
                pr = p.rect
                # If platform is above us and horizontally relevant
                if pr.top < agent.y - 20 and pr.left - 80 < agent.x < pr.right + 80:
                    if agent.is_grounded:
                        actions[0] = 1
                    break

        # ── 4. Dashing ──
        # Dash if far away and on same level, or to close gap
        if dist > 250 and abs(dy) < 50 and agent.dash_cd <= 0:
            if random.random() < 0.05:
                actions[28] = 1

        # ── 5. Combat ──
        if dist < MELEE_RANGE and self.frame % 35 == 0:
            actions[4] = 1  # Melee

        if 150 < dist < 600 and self.frame % 60 == 0:
            if random.random() < 0.7:
                actions[10] = 1  # Ranged attack

        # ── 6. Defensive Reactions ──
        for p in env.projectiles:
            if not p.active or p.owner == 0:
                continue
            pdx = p.x - agent.x
            if abs(pdx) < 200 and abs(p.y - (agent.y - ENTITY_H / 2)) < 50:
                # Projectile incoming — jump or shield
                if agent.is_grounded and random.random() < 0.6:
                    actions[0] = 1
                else:
                    actions[22] = 1 # Block

        # ── Dash toward player occasionally for aggression ──
        if 100 < dist < 300 and random.random() < 0.02:
            actions[28] = 1  # Dash in

        return actions


def draw_bar(surface, x, y, w, h, pct, fg_color, label="", font=None):
    """Draw a stylish health/stamina bar."""
    pygame.draw.rect(surface, (30, 30, 40), (x, y, w, h), border_radius=3)
    fill_w = int(w * max(0, min(1, pct)))
    if fill_w > 0:
        pygame.draw.rect(surface, fg_color, (x, y, fill_w, h), border_radius=3)
    pygame.draw.rect(surface, (100, 100, 120), (x, y, w, h), 1, border_radius=3)
    if label and font:
        txt = font.render(label, True, WHITE)
        surface.blit(txt, (x + 5, y + (h - txt.get_height()) // 2))


def draw_ai_brain_monitor(surface, env, wheel, plasticity, choke_mods, is_adapted, font, font_hint):
    """
    Render the AI Brain Monitor panel.
    Shows tactical state, active reward signals, and adaptation status.
    """
    import math

    ax, ay = env.agent.x, env.agent.y
    ox, oy = env.opponent.x, env.opponent.y
    dist = math.hypot(ax - ox, ay - oy)
    in_range = dist < 105  # MELEE_RANGE * 1.5
    idle = env.agent.idle_frames
    stam_pct = env.agent.stam_pct
    hp_pct = env.agent.hp_pct
    attack_cd = env.agent.attack_cd

    # ── Determine Primary State ──
    if not env.agent.alive:
        primary = ("DEAD", (200, 50, 50))
    elif choke_mods.get("glow", False):
        primary = ("🔥 FLOW STATE (PREDATOR CHOKE)", (255, 80, 40))
    elif env.agent.is_attacking:
        primary = ("⚔️  STRIKING", (255, 220, 50))
    elif env.agent.is_blocking:
        primary = ("🛡️  BLOCKING", (80, 150, 255))
    elif idle > 90:
        primary = ("⚠️  COWARDICE — COMMIT NOW", (255, 60, 60))
    elif idle > 60:
        primary = ("👁️  WINDOW CLOSING — STRIKE", (255, 180, 30))
    elif idle > 30 and in_range:
        primary = ("☸️  PERFECTLY POISED", (180, 255, 100))
    elif idle > 0 and in_range:
        primary = ("🧊 STALKING", (100, 220, 255))
    elif dist > 350:
        primary = ("🏃 HUNTING (Far)", (255, 120, 50))
    elif dist > 150:
        primary = ("🎯 CLOSING IN", (255, 200, 50))
    elif dist < 80:
        primary = ("🤼 CLINCH RANGE", (200, 100, 200))
    else:
        primary = ("👁️  IN THREAT RADIUS", (120, 220, 255))

    # ── Active Signals (what's rewarding/penalizing right now) ──
    signals = []

    if dist > 150:
        pressure = min(((dist - 150) / 100) * 0.003, 0.02)
        signals.append((f"PRESSURE FUNNEL  -{pressure*1000:.1f}‰/fr", (255, 80, 80)))
    else:
        signals.append(("THREAT RADIUS  +5‰/fr", (80, 255, 80)))

    if env.agent.is_whiffing:
        mult = min(3.0, 1.0 + env.agent.whiff_streak * 0.3)
        signals.append((f"WHIFF ×{mult:.1f}  -{50*mult:.0f}‰/fr", (255, 50, 50)))

    opp_x = env.opponent.x
    if (opp_x < 150 or opp_x > 1000 - 150) and dist < 200:
        signals.append(("CORNERING  +5‰/fr", (255, 200, 50)))

    if env.opponent.is_attacking:
        signals.append(("COUNTER WINDOW OPEN  +0.25 on hit!", (100, 255, 100)))

    if stam_pct < 0.2:
        signals.append(("EXHAUSTED  -5‰/fr", (255, 60, 60)))

    if is_adapted:
        signals.append(("☸️ WHEEL ACTIVE — 0-FRAME REACT", (255, 230, 100)))

    if idle > 30 and in_range:
        if idle <= 60:
            signals.append((f"PATIENCE {idle}/60  +5‰/fr", (100, 255, 150)))
        elif idle <= 90:
            signals.append((f"FADING {idle}/90  +1‰/fr", (200, 200, 80)))
        else:
            signals.append((f"COWARDICE TAX {idle}fr  -8‰/fr", (255, 50, 50)))

    # ── Draw Panel Background ──
    panel_x, panel_y = 5, 70
    panel_w, panel_h = 300, 260
    panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surf.fill((10, 10, 20, 195))
    pygame.draw.rect(panel_surf, (80, 80, 120), (0, 0, panel_w, panel_h), 1, border_radius=6)
    surface.blit(panel_surf, (panel_x, panel_y))

    y_cursor = panel_y + 8

    # Header
    hdr = font.render("☸️  AI BRAIN MONITOR", True, (180, 180, 255))
    surface.blit(hdr, (panel_x + 8, y_cursor))
    y_cursor += 20
    pygame.draw.line(surface, (60, 60, 100), (panel_x + 5, y_cursor), (panel_x + panel_w - 5, y_cursor), 1)
    y_cursor += 6

    # Primary State
    state_txt = font.render(primary[0], True, primary[1])
    surface.blit(state_txt, (panel_x + 8, y_cursor))
    y_cursor += 18

    # Stats row
    stats = [
        f"DIST: {dist:.0f}px",
        f"IDLE: {idle}fr",
        f"CD: {attack_cd:.0f}fr",
        f"WHIFF×{env.agent.whiff_streak}",
    ]
    stat_txt = font_hint.render("  |  ".join(stats), True, (140, 140, 180))
    surface.blit(stat_txt, (panel_x + 8, y_cursor))
    y_cursor += 16

    pygame.draw.line(surface, (40, 40, 70), (panel_x + 5, y_cursor), (panel_x + panel_w - 5, y_cursor), 1)
    y_cursor += 5

    # Signals
    sig_hdr = font_hint.render("ACTIVE SIGNALS:", True, (120, 120, 160))
    surface.blit(sig_hdr, (panel_x + 8, y_cursor))
    y_cursor += 14

    for sig_text, sig_color in signals[:6]:  # max 6 to fit panel
        sig_surf = font_hint.render(f"  • {sig_text}", True, sig_color)
        surface.blit(sig_surf, (panel_x + 8, y_cursor))
        y_cursor += 13

    pygame.draw.line(surface, (40, 40, 70), (panel_x + 5, y_cursor + 2), (panel_x + panel_w - 5, y_cursor + 2), 1)
    y_cursor += 10

    # Hebbian Top Weights
    wgt_hdr = font_hint.render("HEBBIAN (action risk weights):", True, (120, 120, 160))
    surface.blit(wgt_hdr, (panel_x + 8, y_cursor))
    y_cursor += 13

    action_names = {4: "Melee", 16: "Ki Blast", 10: "AI Ranged", 28: "Dash", 22: "Block", 0: "Jump"}
    for idx, name in action_names.items():
        w = plasticity.weights[idx]
        color = (100, 255, 100) if w >= 1.0 else (255, max(0, int(255 * (w - 0.2) / 0.8)), 50)
        bar_w = int((w / 2.0) * 80)
        pygame.draw.rect(surface, (30, 30, 50), (panel_x + 85, y_cursor + 1, 80, 9), border_radius=2)
        pygame.draw.rect(surface, color, (panel_x + 85, y_cursor + 1, bar_w, 9), border_radius=2)
        lbl = font_hint.render(f"  {name}: {w:.2f}", True, color)
        surface.blit(lbl, (panel_x + 8, y_cursor))
        y_cursor += 12

    # TAB hint at bottom
    tab_txt = font_hint.render("[TAB] Toggle Monitor", True, (60, 60, 90))
    surface.blit(tab_txt, (panel_x + 8, panel_y + panel_h - 16))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("LNN Predator - Complex Arena (Sukuna Edition)")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Consolas", 14, bold=True)
    font_big = pygame.font.SysFont("Consolas", 18, bold=True)
    font_dmg = pygame.font.SysFont("Impact", 26, bold=True)
    font_hint = pygame.font.SysFont("Consolas", 12)

    # ── Load Model or Fall Back to Bot ──
    model = None
    bot = BotAI()
    checkpoint_path = "checkpoints/ppo/combat_lnn_complex.pt"
    if os.path.exists(checkpoint_path):
        try:
            from model import CombatLNN
            model = CombatLNN(state_dim=384, hidden_size=1280, num_action_slots=40)
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
            model.load_state_dict(sd)
            model.eval()
            print("Loaded trained model for Complex Arena!")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Running with Bot AI fallback.")
            model = None
    else:
        print("No trained model found. Running with Bot AI fallback.")

    env = ComplexArenaEnv(state_dim=384)
    obs = env.reset()
    hx_list = None
    
    wheel = MahoragaWheel(adaptation_threshold=3)
    plasticity = HebbianPlasticity(num_action_slots=40)
    obs_queue = deque(maxlen=15)
    show_monitor = True  # Toggle with TAB
    
    # Debug Logger
    logger = DebugLogger("config.yaml")
    logger.set_action_map({
        "0": "Jump", "2": "Left", "3": "Right", "4": "Melee", 
        "10": "Dismantle", "16": "Ki Blast", "22": "Shield", "28": "Dash"
    }, "Mahoraga LNN")
    logger.start()
    
    # File Logger for Antigravity Analysis
    behavior_log_path = "ai_behavior_log.txt"
    with open(behavior_log_path, "w") as f:
        f.write("AI Behavior Log Started\n")

    player_score = 0
    ai_score = 0
    damage_flashes = []  # [x, y, text, timer, color]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    show_monitor = not show_monitor

        keys = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pressed()

        # ── Player Input ──
        player_actions = np.zeros(40, dtype=int)
        if keys[pygame.K_w]:      player_actions[0] = 1   # Jump
        if keys[pygame.K_a]:      player_actions[2] = 1   # Left
        if keys[pygame.K_d]:      player_actions[3] = 1   # Right
        if mouse[0]:              player_actions[4] = 1   # Melee (LMB)
        if mouse[2]:              player_actions[22] = 1  # Shield (RMB)
        if keys[pygame.K_q]:      player_actions[16] = 1  # Ki Blast
        if keys[pygame.K_e]:      player_actions[10] = 1  # DISMANTLE
        if keys[pygame.K_SPACE]:  player_actions[28] = 1  # Dash

        # ── Mahoraga Override & Delay Queue ──
        obs_queue.append(obs)
        
        # Predator's Choke (HP-based Flow State)
        choke_mods = PredatorChoke.get_modifiers(env.agent.hp_pct)
        base_delay = 0 if choke_mods["cd_reduction"] > 0 else 10
        
        is_adapted = wheel.is_adapted(env.projectiles, env.agent.x, env.agent.y)
        if is_adapted:
            current_delay = 0
            if env.agent.dash_cd > 0:
                env.agent.dash_cd = max(0, env.agent.dash_cd - 5) # Faster escape
        else:
            current_delay = base_delay
            
        if len(obs_queue) > current_delay:
            delayed_obs = obs_queue[-(current_delay + 1)]
        else:
            delayed_obs = obs_queue[0]

        # ── AI Decision ──
        if model is not None:
            obs_tensor = torch.FloatTensor(delayed_obs).unsqueeze(0)
            with torch.no_grad():
                out = model.forward(obs_tensor, hx_list=hx_list)
                
                # Apply Hebbian Plasticity to logits before sampling
                mod_logits = plasticity.modulate(out["action_logits"].numpy()[0])
                mod_logits_tensor = torch.FloatTensor(mod_logits).unsqueeze(0)
                actions_tensor, log_probs = model.action_head.sample(mod_logits_tensor)
                
                ai_actions = actions_tensor[0].numpy().astype(int)
                hx_list = out["hx_list"]
                
                # Record action for potential Hebbian update
                plasticity.record_action(ai_actions)
                
                # Log frame to Debug Logger
                emo_vals = out["emotions"].cpu().numpy()[0]
                emo_dict = {
                    "aggression": float(emo_vals[0]),
                    "confidence": float(emo_vals[1]),
                    "frustration": float(emo_vals[2]),
                    "focus": float(emo_vals[3])
                }
                
                # Determine mood
                current_mood = "neutral"
                if emo_dict["confidence"] > 0.7: current_mood = "confident"
                if emo_dict["aggression"] > 0.8: current_mood = "cocky"
                if emo_dict["frustration"] > 0.6: current_mood = "tilted"
                
                logger.log_frame(
                    actions=ai_actions.tolist(),
                    action_confidences=torch.sigmoid(out["action_logits"]).cpu().numpy()[0].tolist(),
                    mood=current_mood,
                    emotions=emo_dict,
                    latency_ms=0.0 # Will be tracked by FPS
                )
                
                # Write to behavior log every 30 frames to avoid bloat
                if env.steps % 30 == 0:
                    with open(behavior_log_path, "a") as f:
                        f.write(f"Frame {env.steps} | Mood: {current_mood} | Emo: {emo_dict}\n")
                        f.write(f"  Actions: {ai_actions.tolist()}\n")
                        f.write(f"  Obs (first 16): {delayed_obs[:16].tolist()}\n")
        else:
            ai_actions = bot.act(env)

        # ── Track HP for damage flashes ──
        prev_ai_hp = env.agent.hp
        prev_player_hp = env.opponent.hp

        # ── Step ──
        obs, reward, done, trunc, info = env.step(ai_actions, opponent_actions=player_actions)

        # ── Damage Flashes & Mahoraga Learning ──
        if env.agent.hp < prev_ai_hp:
            dmg = prev_ai_hp - env.agent.hp
            damage_flashes.append([env.agent.x, env.agent.y - ENTITY_H - 20,
                                   f"-{dmg:.0f}", 45, DMG_AI])
            # Live Adaptation Update
            wheel.record_damage(env.projectiles, env.agent.x, env.agent.y)
            plasticity.penalize()
            
        if env.opponent.hp < prev_player_hp:
            dmg = prev_player_hp - env.opponent.hp
            damage_flashes.append([env.opponent.x, env.opponent.y - ENTITY_H - 20,
                                   f"-{dmg:.0f}", 45, DMG_PLAYER])
            plasticity.reward()

        if done:
            if not env.agent.alive:
                player_score += 1
            elif not env.opponent.alive:
                ai_score += 1
            obs = env.reset()
            hx_list = None
            damage_flashes = []

        # ═══════════════════════════ RENDERING ═══════════════════════════

        # ── Background gradient ──
        for y_line in range(SCREEN_H):
            t = y_line / SCREEN_H
            r = int(BG_TOP[0] * (1 - t) + BG_BOT[0] * t)
            g = int(BG_TOP[1] * (1 - t) + BG_BOT[1] * t)
            b = int(BG_TOP[2] * (1 - t) + BG_BOT[2] * t)
            pygame.draw.line(screen, (r, g, b), (0, y_line), (SCREEN_W, y_line))

        # ── Platforms ──
        for plat in env.platforms:
            pr = plat.rect
            is_pillar = pr.w < 60
            color = PILLAR_COLOR if is_pillar else PLAT_COLOR
            border = (80, 80, 100) if is_pillar else PLAT_BORDER
            rect = pygame.Rect(int(pr.x), int(pr.y), int(pr.w), int(pr.h))
            pygame.draw.rect(screen, color, rect, border_radius=4)
            pygame.draw.rect(screen, border, rect, 2, border_radius=4)
            if plat.is_moving:
                # Draw subtle arrows on moving platforms
                cx = int(pr.cx)
                cy = int(pr.cy)
                pygame.draw.polygon(screen, (120, 120, 140),
                                    [(cx - 8, cy), (cx - 16, cy - 5), (cx - 16, cy + 5)])
                pygame.draw.polygon(screen, (120, 120, 140),
                                    [(cx + 8, cy), (cx + 16, cy - 5), (cx + 16, cy + 5)])

        # ── Pickups ──
        for pk in env.pickups:
            if not pk.active:
                continue
            color = PICKUP_HP if pk.pickup_type == 0 else PICKUP_STAM
            pulse = 1.0 + 0.2 * math.sin(env.steps * 0.1)
            r = int(10 * pulse)
            pygame.draw.circle(screen, color, (int(pk.x), int(pk.y)), r)
            pygame.draw.circle(screen, WHITE, (int(pk.x), int(pk.y)), r, 1)
            label = "+" if pk.pickup_type == 0 else "⚡"
            txt = font_hint.render(label, True, WHITE)
            screen.blit(txt, (int(pk.x) - txt.get_width() // 2, int(pk.y) - r - 12))

        # ── Player Character ──
        px, py = int(env.opponent.x), int(env.opponent.y)
        body_rect = pygame.Rect(px - ENTITY_W // 2, py - ENTITY_H, ENTITY_W, ENTITY_H)
        pygame.draw.rect(screen, PLAYER_BODY, body_rect, border_radius=6)
        pygame.draw.rect(screen, PLAYER_OUTLINE, body_rect, 2, border_radius=6)
        # Eyes
        eye_y = py - ENTITY_H + 15
        ex = px + int(env.opponent.facing * 5)
        pygame.draw.circle(screen, WHITE, (ex, eye_y), 4)
        pygame.draw.circle(screen, (0, 0, 0), (ex + int(env.opponent.facing * 2), eye_y), 2)
        # Shield glow
        if env.opponent.is_blocking:
            s = pygame.Surface((ENTITY_W + 20, ENTITY_H + 20), pygame.SRCALPHA)
            pygame.draw.ellipse(s, (100, 150, 255, 60), s.get_rect())
            screen.blit(s, (px - ENTITY_W // 2 - 10, py - ENTITY_H - 10))
        # Label
        lbl = font_hint.render("YOU", True, PLAYER_OUTLINE)
        screen.blit(lbl, (px - lbl.get_width() // 2, py - ENTITY_H - 18))

        # ── AI Character ──
        ax, ay = int(env.agent.x), int(env.agent.y)
        ai_rect = pygame.Rect(ax - ENTITY_W // 2, ay - ENTITY_H, ENTITY_W, ENTITY_H)
        pygame.draw.rect(screen, AI_BODY, ai_rect, border_radius=6)
        pygame.draw.rect(screen, AI_OUTLINE, ai_rect, 2, border_radius=6)
        # Eyes
        eye_y = ay - ENTITY_H + 15
        ex = ax + int(env.agent.facing * 5)
        pygame.draw.circle(screen, WHITE, (ex, eye_y), 4)
        pygame.draw.circle(screen, (0, 0, 0), (ex + int(env.agent.facing * 2), eye_y), 2)
        # Shield glow
        if env.agent.is_blocking:
            s = pygame.Surface((ENTITY_W + 20, ENTITY_H + 20), pygame.SRCALPHA)
            pygame.draw.ellipse(s, (255, 100, 100, 60), s.get_rect())
            screen.blit(s, (ax - ENTITY_W // 2 - 10, ay - ENTITY_H - 10))
        # Label
        mode_label = "LNN" if model else "BOT"
        lbl = font_hint.render(mode_label, True, AI_OUTLINE)
        screen.blit(lbl, (ax - lbl.get_width() // 2, ay - ENTITY_H - 18))
        
        # ── Mahoraga Wheel & Flow State Aura ──
        wheel.draw(screen, ax, ay - ENTITY_H - 45)
        if choke_mods.get("glow", False):
            s = pygame.Surface((ENTITY_W + 30, ENTITY_H + 30), pygame.SRCALPHA)
            pygame.draw.ellipse(s, (200, 50, 50, 40), s.get_rect())
            screen.blit(s, (ax - ENTITY_W // 2 - 15, ay - ENTITY_H - 15))

        # ── Dash trail effects ──
        for ent, color in [(env.opponent, PLAYER_OUTLINE), (env.agent, AI_OUTLINE)]:
            if ent.is_dashing:
                ex, ey = int(ent.x), int(ent.y)
                trail_s = pygame.Surface((60, ENTITY_H), pygame.SRCALPHA)
                trail_s.fill((*color[:3], 40))
                offset_x = -int(ent.facing * 30)
                screen.blit(trail_s, (ex - 30 + offset_x, ey - ENTITY_H))

        # ── Projectiles ──
        for p in env.projectiles:
            if not p.active:
                continue
            sx, sy = int(p.x), int(p.y)
            if p.proj_type == 0:  # Ki Blast
                pygame.draw.circle(screen, KI_COLOR, (sx, sy), 7)
                pygame.draw.circle(screen, WHITE, (sx, sy), 7, 1)
            elif p.proj_type == 1:  # DISMANTLE
                # Vertical crimson slash with trail
                for t in range(4):
                    trail_x = sx - int(p.vx * t * 0.5)
                    alpha = 255 - t * 60
                    color = (255, 20, 120)
                    pygame.draw.line(screen, color, (trail_x, sy - 25), (trail_x, sy + 25), 3)
                pygame.draw.line(screen, DISMANTLE_COLOR, (sx, sy - 30), (sx, sy + 30), 5)
                # Cross slash effect
                pygame.draw.line(screen, DISMANTLE_COLOR, (sx - 15, sy - 15), (sx + 15, sy + 15), 3)
                pygame.draw.line(screen, DISMANTLE_COLOR, (sx + 15, sy - 15), (sx - 15, sy + 15), 3)
            elif p.proj_type == 2:  # AI ranged
                pygame.draw.circle(screen, AI_PROJ_COLOR, (sx, sy), 6)
                pygame.draw.circle(screen, (255, 80, 0), (sx, sy), 6, 1)

        # ── Damage Flashes ──
        new_flashes = []
        for f in damage_flashes:
            txt = font_dmg.render(f[2], True, f[4])
            screen.blit(txt, (int(f[0]) - txt.get_width() // 2, int(f[1])))
            f[1] -= 1.2
            f[3] -= 1
            if f[3] > 0:
                new_flashes.append(f)
        damage_flashes = new_flashes

        # ── AI Brain Monitor (TAB to toggle) ──
        if show_monitor:
            draw_ai_brain_monitor(
                screen, env, wheel, plasticity, choke_mods,
                is_adapted, font, font_hint
            )

        # ── HUD ──
        # Player HUD (left)
        draw_bar(screen, 30, 20, 280, 18, env.opponent.hp_pct, PLAYER_BODY, "HP", font_hint)
        draw_bar(screen, 30, 42, 200, 12, env.opponent.stam_pct, STAM_YELLOW, "STA", font_hint)

        # AI HUD (right)
        draw_bar(screen, SCREEN_W - 310, 20, 280, 18, env.agent.hp_pct, AI_BODY, "HP", font_hint)
        draw_bar(screen, SCREEN_W - 230, 42, 200, 12, env.agent.stam_pct, STAM_YELLOW, "STA", font_hint)

        # Score
        score_txt = font_big.render(f"YOU  {player_score}  —  {ai_score}  AI", True, WHITE)
        screen.blit(score_txt, (SCREEN_W // 2 - score_txt.get_width() // 2, 15))

        # Timer
        remaining = max(0, env.max_steps - env.steps)
        timer_txt = font.render(f"Time: {remaining // 60}s", True, (150, 150, 170))
        screen.blit(timer_txt, (SCREEN_W // 2 - timer_txt.get_width() // 2, 42))

        # Controls
        hint = font_hint.render(
            "W:Jump | A/D:Move | Space:Dash | LMB:Melee | RMB:Shield | Q:Ki Blast | E:DISMANTLE",
            True, (80, 80, 100)
        )
        screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 22))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
