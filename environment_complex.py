"""
environment_complex.py — Solid Platform Collision (v10.2)
========================================================
FIX: Agent now correctly lands on and collides with platforms.
"""

import numpy as np
import random
import math
import cv2
import heapq
from dataclasses import dataclass

@dataclass
class Rect:
    x: float; y: float; w: float; h: float
    def colliderect(self, other):
        return (self.x < other.x + other.w and self.x + self.w > other.x and
                self.y < other.y + other.h and self.y + self.h > other.y)

class ComplexArenaEnv:
    # Target Ratios from Spec
    GENRE_RATIOS = {
        "dungeon": 0.10, "3d": 0.10, "shmup": 0.10,
        "top-down": 0.15, "beat": 0.15, "scroller": 0.15,
        "isometric": 0.20, "oblique": 0.20, "run-and-gun": 0.20,
        "platformer": 0.30, "sidescroller": 0.30, "metroidvania": 0.30
    }

    def __init__(self, num_rays=32, grid_size=11):
        self.num_rays = num_rays
        self.grid_size = grid_size
        self.grid_res = 40
        self.spatial_grid = np.zeros((grid_size, grid_size))
        self.world_width = 3000
        self.total_episodes = 0 # Used for Buffering Logic (epochs < 100)
        self.reset()
        
    def reset(self, genre="platformer", run_number=1):
        self.genre = genre.lower()
        self.run_number = run_number
        self.is_td = any(x in self.genre for x in ["top-down", "dungeon", "3d", "shmup", "isometric", "beat"])
        self.platforms = []
        self.spine_rects = []
        
        # Safety Fallback for UI/Reward tracking
        self.target_rect = Rect(self.world_width - 150, 500, 80, 80)
        
        # 1. Deterministic Scaling
        if self.run_number == 1:
            self.current_modifier_density = 0.0
        elif self.run_number == 2:
            self.current_modifier_density = self.GENRE_RATIOS.get(self.genre, 0.20) * 0.5
        else:
            self.current_modifier_density = self.GENRE_RATIOS.get(self.genre, 0.20)

        self._generate_world()
        self.agent = Entity(150, 500)
        self.spatial_grid.fill(0)
        self.health = 100.0
        self.total_episodes += 1
        return self._get_obs()

    def _generate_world(self):
        # Dispatcher: Strictly Platformer-only Graph Generation
        if any(x in self.genre for x in ["platformer", "sidescroller", "metroidvania", "scroller"]):
            self._generate_platformer_graph_layout()
        else:
            # Fallback for empty worlds (Clean Slate)
            self.platforms = [Platform(Rect(50, 600, 300, 30), 0)]
            self.platforms.append(Platform(self.target_rect, 9))
            self.platforms.append(Platform(Rect(self.target_rect.x - 60, self.target_rect.y + 100, 200, 30), 0))

    def _can_physically_reach(self, p1, p2):
        # BUG FIX 1 & 2: No VY drag + Key-holding simulation
        ACCEL = 1.3
        DRAG_VX = 0.82
        GRAVITY = 0.7
        JUMP_FORCE = -14.0
        SAFETY = 0.85
        MAX_VX = 7.22

        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        
        # Test 1: Jumping (Simulate holding RIGHT + JUMP)
        vx, vy = 0.0, JUMP_FORCE
        x, y = 0.0, 0.0
        for _ in range(120):
            vx = min(vx + ACCEL, MAX_VX * SAFETY)
            vy += GRAVITY
            vx *= DRAG_VX
            x += vx; y += vy
            if abs(x - dx) < 70 and abs(y - dy) < 30: return True
            if x > dx + 100 or y > dy + 300: break
            
        # Test 2: Dropping (Holding RIGHT only)
        vx, vy = 0.0, 0.0
        x, y = 0.0, 0.0
        for _ in range(120):
            vx = min(vx + ACCEL, MAX_VX * SAFETY)
            vy += GRAVITY
            vx *= DRAG_VX
            x += vx; y += vy
            if abs(x - dx) < 70 and abs(y - dy) < 30: return True
            if x > dx + 100 or y > dy + 300: break
            
        return False

    def _generate_platformer_graph_layout(self):
        # 5-PHASE GRAPH GENERATOR (TRUE RGE + SPAWN TETHER)
        for attempt in range(10): # Retry loop if spawn is isolated
            success = self._attempt_platformer_generation()
            if success: break

    def _attempt_platformer_generation(self):
        spawn_node = (150, 600)
        goal_x = self.world_width - 250
        goal_pos = (goal_x, 500)
        target_density = 40
        
        nodes = {spawn_node}
        graph = {spawn_node: []}; in_edges = {spawn_node: []}
        
        # Force the first hop (conservative range)
        bridge_found = False
        for _ in range(100):
            dx, dy = random.randint(120, 220), random.randint(-40, 60)
            candidate = (spawn_node[0] + dx, spawn_node[1] + dy)
            if self._can_physically_reach(spawn_node, candidate):
                nodes.add(candidate); graph[candidate] = []; in_edges[candidate] = [spawn_node]
                graph[spawn_node].append(candidate)
                bridge_found = True; break
        
        if not bridge_found: return False
        
        frontier = []
        for n in nodes:
            if n == spawn_node: continue
            dg = math.sqrt((n[0]-goal_pos[0])**2 + (n[1]-goal_pos[1])**2)
            ds = math.sqrt((n[0]-spawn_node[0])**2 + (n[1]-spawn_node[1])**2)
            heapq.heappush(frontier, (0.7 * dg - 0.3 * ds, n))
        
        goal_node = None
        while frontier:
            # Stop only if we've reached the goal AND met minimum density
            if goal_node and len(nodes) >= target_density:
                break
                
            _, p = heapq.heappop(frontier)
            
            candidates = []
            for _ in range(12): # More samples to ensure progress
                dx = random.randint(120, 260)
                dy = random.randint(-140, 140)
                nx, ny = min(p[0] + dx, self.world_width - 150), np.clip(p[1] + dy, 200, 650)
                c = (nx, ny)
                if self._can_physically_reach(p, c):
                    candidates.append(c)

            # Add all valid candidates to maintain graph connectivity
            for c in candidates:
                if c not in nodes:
                    nodes.add(c)
                    graph[c] = []; in_edges[c] = []
                    dg = math.sqrt((c[0]-goal_pos[0])**2 + (c[1]-goal_pos[1])**2)
                    ds = math.sqrt((c[0]-spawn_node[0])**2 + (c[1]-spawn_node[1])**2)
                    heapq.heappush(frontier, (0.7 * dg - 0.3 * ds, c))
                
                if c not in graph[p]:
                    graph[p].append(c); in_edges[c].append(p)
                
                # Check for Goal Reach
                if c[0] >= goal_x - 100 and not goal_node:
                    goal_node = c

        if not goal_node: 
            return False # Fail and retry RGE if no natural path to goal exists

        reachable_to_goal = {goal_node}
        q = [goal_node]
        while q:
            curr = q.pop(0)
            for parent in in_edges.get(curr, []):
                if parent not in reachable_to_goal:
                    reachable_to_goal.add(parent); q.append(parent)
        
        # Isolation Detection
        if spawn_node not in reachable_to_goal: return False 
        
        final_nodes = [n for n in nodes if n in reachable_to_goal]
        self.platforms = []; self.spine_rects = []
        for node in final_nodes:
            px, py = node
            p_width = 150
            for child in graph.get(node, []):
                if child in reachable_to_goal and (child[0] - px) > 200: p_width = 380; break
            
            if node == goal_node:
                self.target_rect = Rect(px, py - 100, 80, 80)
                self.platforms.append(Platform(self.target_rect, 9))
                self.platforms.append(Platform(Rect(px-100, py, 200, 30), 0))
            else:
                type_id = random.randint(1, 5) if random.random() < self.current_modifier_density else 0
                if node == spawn_node: type_id = 0
                plat_rect = Rect(px - p_width//2, py, p_width, 30)
                self.platforms.append(Platform(plat_rect, type_id))
                self.spine_rects.append(plat_rect)
        return True

    def step(self, action):
        self._apply_physics(self.agent, action)
        self._update_spatial_grid(self.agent.vx, self.agent.vy)
        surprise = 0.0; done = False; lost = False
        
        # Fall into void condition (Platformer only)
        if not self.is_td and self.agent.y >= 675:
            done = True; lost = True

        for p in self.platforms:
            if self.agent.rect.colliderect(p.rect):
                if p.type_id == 4: self.health -= 5.0; surprise = 1.0
                if p.type_id == 9: done = True
        if self.health <= 0: done = True; lost = True
        return self._get_obs(), 0.0, done, False, {"surprise": surprise, "lost": lost}

    def _apply_physics(self, ent, action):
        accel = 1.3
        if action[1]: ent.vx -= accel
        if action[2]: ent.vx += accel
        
        if self.is_td:
            if action[3]: ent.vy -= accel
            if action[4]: ent.vy += accel
        else:
            if action[3] and ent.is_grounded: ent.vy = -14.0; ent.is_grounded = False
            ent.vy += 0.7 # Gravity
            
        ent.vx *= 0.82 # Horizontal drag
        # ent.vy drag REMOVED
        
        # X movement and collision
        ent.x += ent.vx
        ent.rect.x = ent.x - 15
        for p in self.platforms:
            if p.type_id == 9: continue # Goal is not solid
            if ent.rect.colliderect(p.rect):
                if ent.vx > 0: ent.x = p.rect.x - 15; ent.vx = 0
                elif ent.vx < 0: ent.x = p.rect.x + p.rect.w + 15; ent.vx = 0
        
        # Y movement and collision
        ent.y += ent.vy
        ent.rect.y = ent.y - 15
        ent.is_grounded = False
        for p in self.platforms:
            if p.type_id == 9: continue
            if ent.rect.colliderect(p.rect):
                if ent.vy > 0: # Landing
                    ent.y = p.rect.y - 15
                    ent.vy = 0
                    ent.is_grounded = True
                elif ent.vy < 0: # Head bonk
                    ent.y = p.rect.y + p.rect.h + 15
                    ent.vy = 0
        
        # World Bounds
        ent.x = np.clip(ent.x, 30, self.world_width - 30)
        ent.y = np.clip(ent.y, 30, 680)
        ent.rect.x, ent.rect.y = ent.x-15, ent.y-15

    def _update_spatial_grid(self, vx, vy):
        shift_x, shift_y = -vx / self.grid_res, -vy / self.grid_res
        self.spatial_grid = self._shift_2d(self.spatial_grid, shift_x, shift_y)
        self.spatial_grid *= 0.95
        for i in range(self.num_rays):
            angle = (i / self.num_rays) * 2 * math.pi
            dx, dy = math.cos(angle), math.sin(angle)
            for d in range(5, 400, 20):
                rx, ry = self.agent.x + dx*d, self.agent.y + dy*d
                gx = int((dx*d) / self.grid_res) + (self.grid_size // 2)
                gy = int((dy*d) / self.grid_res) + (self.grid_size // 2)
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    for p in self.platforms:
                        if p.rect.x < rx < p.rect.x+p.rect.w and p.rect.y < ry < p.rect.y+p.rect.h:
                            self.spatial_grid[gy, gx] = p.signal; break
                    if self.spatial_grid[gy, gx] > 0: break

    def _shift_2d(self, arr, dx, dy):
        idx, idy = int(round(dx)), int(round(dy))
        res = np.zeros_like(arr); h, w = arr.shape
        for y in range(h):
            for x in range(w):
                nx, ny = x + idx, y + idy
                if 0 <= nx < w and 0 <= ny < h: res[ny, nx] = arr[y, x]
        return res

    def _get_obs(self): return self.spatial_grid.flatten().astype(np.float32)

    def render(self, custom_cam_x=None):
        img = np.zeros((700, 1000, 3), dtype=np.uint8); img[:] = (10, 10, 15)
        if custom_cam_x is None:
            cam_x = np.clip(self.agent.x - 500, 0, self.world_width - 1000)
        else:
            cam_x = np.clip(custom_cam_x, 0, self.world_width - 1000)
            
        for p in self.platforms:
            rx, ry = int(p.rect.x - cam_x), int(p.rect.y)
            # Standard Platform
            cv2.rectangle(img, (rx, ry), (rx+int(p.rect.w), ry+int(p.rect.h)), p.color, -1)
            
            # Special Rendering for Goal (ID 9)
            if p.type_id == 9:
                # Add a glowing border
                cv2.rectangle(img, (rx-5, ry-5), (rx+int(p.rect.w)+5, ry+int(p.rect.h)+5), (0, 255, 100), 2)
                # Add "GOAL" Label
                cv2.putText(img, "TARGET", (rx, ry-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Add Beacon/Energy Beam (Vertical line to top of screen)
                cv2.line(img, (rx + int(p.rect.w)//2, ry), (rx + int(p.rect.w)//2, 0), (0, 100, 0), 1)
                # Add inner pulse (simulated)
                cv2.rectangle(img, (rx+10, ry+10), (rx+int(p.rect.w)-10, ry+int(p.rect.h)-10), (255, 255, 255), 1)

        # Draw Agent
        cv2.circle(img, (int(self.agent.x - cam_x), int(self.agent.y)), 12, (0, 255, 150), -1)
        return img

class Platform:
    # Spec Signals & ID Mappings
    SIGNALS = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0, 5: 0.5, 9: 1.0}
    def __init__(self, rect, type_id=0):
        self.rect = rect; self.type_id = type_id
        self.signal = self.SIGNALS.get(type_id, 0.0)
        # Spec Color Mappings
        self.debug_colors = {
            0: (128, 128, 128), # Normal Gray #808080
            1: (209, 206, 0),   # Ice Cyan #00CED1 (BGR: 209, 206, 0 approx)
            2: (0, 215, 255),   # Motion Yellow #FFD700 (BGR: 0, 215, 255 approx)
            3: (226, 43, 138),  # Unstable Purple #8A2BE2
            4: (0, 36, 255),    # Hazard Red #FF2400 (BGR: 0, 36, 255)
            5: (57, 255, 20),   # Logic Green #39FF14 (BGR: 20, 255, 57)
            9: (0, 255, 0)      # Goal
        }
        self.color = self.debug_colors.get(type_id, (128,128,128))

class Entity:
    def __init__(self, x, y):
        self.x, self.y = x, y; self.vx, self.vy = 0, 0
        self.is_grounded = False; self.rect = Rect(x-15, y-15, 30, 30)
