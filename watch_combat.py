import time
import os
import torch
import numpy as np
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

from model import CombatLNN
from environment_wrapper import ArenaEnv

def create_arena_view(env):
    # Create a 20x20 grid
    grid_size = 20
    arena = [[" " for _ in range(grid_size * 2)] for _ in range(grid_size)]
    
    half = env.arena_size / 2
    
    # Map coordinates to grid
    def to_grid(x, y):
        gx = int((x + half) / env.arena_size * (grid_size * 2 - 1))
        gy = int((y + half) / env.arena_size * (grid_size - 1))
        return max(0, min(grid_size * 2 - 1, gx)), max(0, min(grid_size - 1, gy))
    
    # Draw Player (Opponent)
    px, py = to_grid(env.opponent.x, env.opponent.y)
    arena[py][px] = "[blue]P[/blue]"
    
    # Draw LNN Agent
    ax, ay = to_grid(env.agent.x, env.agent.y)
    arena[ay][ax] = "[red]A[/red]"
    
    # Add attack visualizers
    if env.agent.is_attacking:
        arena[max(0, ay-1)][ax] = "[yellow]*[/yellow]"
    if env.opponent.is_attacking:
        arena[max(0, py-1)][px] = "[cyan]*[/cyan]"

    # Convert grid to string
    lines = []
    for row in arena:
        lines.append("".join(row))
    return Panel("\n".join(lines), title="Arena (A=AI, P=Player Dummy)", expand=False)

def watch():
    print("Loading 128-hidden Quick Checkpoint...")
    model = CombatLNN(state_dim=64, hidden_size=128, num_action_slots=40, num_cfc_layers=2)
    model.load_state_dict(torch.load("checkpoints/ppo/combat_lnn_quick.pt", map_location="cpu", weights_only=True))
    model.eval()
    
    env = ArenaEnv()
    obs = env.reset()
    hx_list = None
    
    with Live(refresh_per_second=10) as live:
        for _ in range(500):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to("cpu")
            with torch.no_grad():
                result = model.act(obs_tensor, hx_list=hx_list)
            
            actions = result["actions"][0].numpy().astype(int)
            hx_list = result["hx_list"]
            
            obs, reward, done, trunc, info = env.step(actions)
            
            # UI layout
            table = Table(show_header=False, box=None)
            
            # Stats panel
            stats = f"""
[red]AI Agent (LNN)[/red]
HP: {info['agent_hp']:.0f}/100
Combo: {env.agent.combo_counter}
Attacking: {env.agent.is_attacking}

[blue]Dummy Opponent[/blue]
HP: {info['opponent_hp']:.0f}/100
Attacking: {env.opponent.is_attacking}

Steps: {info['steps']}
Damage Dealt: {info['damage_dealt']:.0f}
Damage Taken: {info['damage_taken']:.0f}
            """
            
            table.add_row(create_arena_view(env), Panel(stats, title="Match Stats"))
            live.update(table)
            
            time.sleep(0.1) # Slow down so you can see it
            
            if done or trunc:
                time.sleep(1)
                obs = env.reset()
                hx_list = None

if __name__ == "__main__":
    watch()
