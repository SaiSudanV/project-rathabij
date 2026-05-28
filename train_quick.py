"""
train_quick.py — Quick PPO Training with Self-Play & Live Visualization
======================================================================
This script trains the CombatLNN on CPU.
Features:
- Self-Play (AI fights its past clones)
- Live Terminal Visualization
- 128-dim state space (40-slot player awareness)
"""

import time
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rich.live import Live
from rich.table import Table
from rich.panel import Panel

from model import CombatLNN
from environment_wrapper import ArenaEnv
from trainer_rl import compute_gae, RolloutBuffer

def create_arena_view(env):
    """Create a 20x20 text grid representing the arena."""
    grid_size = 20
    arena = [[" " for _ in range(grid_size * 2)] for _ in range(grid_size)]
    half = env.arena_size / 2
    
    def to_grid(x, y):
        gx = int((x + half) / env.arena_size * (grid_size * 2 - 1))
        gy = int((y + half) / env.arena_size * (grid_size - 1))
        return max(0, min(grid_size * 2 - 1, gx)), max(0, min(grid_size - 1, gy))
    
    # Opponent
    px, py = to_grid(env.opponent.x, env.opponent.y)
    arena[py][px] = "[blue]P[/blue]"
    
    # Agent
    ax, ay = to_grid(env.agent.x, env.agent.y)
    arena[ay][ax] = "[red]A[/red]"
    
    if env.agent.is_attacking:
        arena[max(0, ay-1)][ax] = "[yellow]*[/yellow]"
    if env.opponent.is_attacking:
        arena[max(0, py-1)][px] = "[cyan]*[/cyan]"

    lines = ["".join(row) for row in arena]
    return Panel("\n".join(lines), title="Arena", expand=False)

def train_quick():
    device = torch.device("cpu")
    visualize = "--visualize" in sys.argv or "-v" in sys.argv
    
    print("[Quick Train] Starting 128-dim model with Self-Play...")
    model = CombatLNN(state_dim=128, hidden_size=128, num_action_slots=40, num_cfc_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    env = ArenaEnv(state_dim=128)

    total_timesteps = 20_000
    rollout_length = 256
    ppo_epochs = 4
    batch_size = 64
    
    # Self-Play pool
    past_models = []
    
    obs = env.reset()
    hx_list = None
    opponent_hx_list = None
    
    episode_reward = 0.0
    episode_count = 0
    wins = 0
    losses = 0
    episode_rewards = []
    global_step = 0
    buffer = RolloutBuffer()
    start_time = time.time()
    
    # Live visualization context
    live = Live(refresh_per_second=10) if visualize else None
    if live: live.start()

    try:
        while global_step < total_timesteps:
            model.eval()
            buffer.clear()
            
            # 100% chance to do self-play vs a past model (dummy bot is just for bootstrapping step 0)
            is_self_play = len(past_models) > 0
            opponent_model = random.choice(past_models) if is_self_play else None

            for _ in range(rollout_length):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                result = model.act(obs_tensor, hx_list=hx_list)
                actions = result["actions"][0].cpu().numpy().astype(int)
                log_prob = result["action_log_probs"][0].cpu().item()
                value = result["value"][0].cpu().item()
                hx_list = result["hx_list"]

                opponent_actions = None
                if is_self_play:
                    # Invert positions for the opponent's perspective
                    opp_obs = obs.copy()
                    opp_obs[0:2], opp_obs[2:4] = opp_obs[2:4], opp_obs[0:2] # Swap positions
                    opp_obs[4], opp_obs[5] = opp_obs[5], opp_obs[4]         # Swap HP
                    opp_obs[6:8], opp_obs[8:10] = opp_obs[8:10], opp_obs[6:8] # Swap Vel
                    opp_obs_tensor = torch.FloatTensor(opp_obs).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        opp_result = opponent_model.act(opp_obs_tensor, hx_list=opponent_hx_list)
                        opponent_actions = opp_result["actions"][0].cpu().numpy().astype(int)
                        opponent_hx_list = opp_result["hx_list"]

                next_obs, reward, done, truncated, info = env.step(actions, opponent_actions=opponent_actions)
                buffer.add(obs, actions, log_prob, reward, value, done or truncated)

                episode_reward += reward
                global_step += 1
                
                # Update UI
                if live:
                    mode_str = "[magenta]Self-Play (AI Clone)[/magenta]" if is_self_play else "[blue]Rule-based Dummy[/blue]"
                    stats = f"""
[red]AI Agent (LNN)[/red]
HP: {info['agent_hp']:.0f}/100 | Combo: {env.agent.combo_counter}

{mode_str}
HP: {info['opponent_hp']:.0f}/100

Step: {global_step}/{total_timesteps} | FPS: {global_step / max(1, time.time() - start_time):.0f}
Win Rate: {wins/max(1, wins+losses):.0%} | Avg R: {np.mean(episode_rewards[-10:]) if episode_rewards else 0:.2f}
"""
                    table = Table(show_header=False, box=None)
                    table.add_row(create_arena_view(env), Panel(stats, title="Match Stats"))
                    live.update(table)
                    time.sleep(0.02) # Cap render speed

                if done or truncated:
                    episode_rewards.append(episode_reward)
                    episode_count += 1
                    if info.get("opponent_hp", 1) <= 0:
                        wins += 1
                    else:
                        losses += 1
                    
                    obs = env.reset()
                    hx_list = None
                    opponent_hx_list = None
                    episode_reward = 0.0
                    
                    # Refresh opponent mode per episode
                    is_self_play = len(past_models) > 0
                    opponent_model = random.choice(past_models) if is_self_play else None
                else:
                    obs = next_obs

            # Save clone for self-play every 8 rollouts (2048 steps)
            if (global_step // rollout_length) % 8 == 0 and len(past_models) < 10:
                clone = CombatLNN(state_dim=128, hidden_size=128, num_action_slots=40, num_cfc_layers=2).to(device)
                clone.load_state_dict(model.state_dict())
                clone.eval()
                # Check if we already added a model this step to prevent dupes
                if not past_models or global_step != getattr(past_models[-1], '_saved_at', -1):
                    clone._saved_at = global_step
                    past_models.append(clone)
                    if len(past_models) > 5:
                        past_models.pop(0)

            # Compute GAE & Train
            data = buffer.get_tensors(device)
            with torch.no_grad():
                last_obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
                last_result = model.forward(last_obs, hx_list=hx_list)
                next_value = last_result["value"][0].cpu()

            advantages, returns = compute_gae(
                data["rewards"].cpu(), data["values"].cpu(),
                data["dones"].cpu(), next_value,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            model.train()
            total_loss = 0.0

            for epoch in range(ppo_epochs):
                indices = np.arange(len(buffer))
                np.random.shuffle(indices)

                for start in range(0, len(buffer), batch_size):
                    batch_idx = indices[start:start + batch_size]
                    batch_obs = data["observations"][batch_idx]
                    batch_actions = data["actions"][batch_idx]
                    batch_old_lp = data["log_probs"][batch_idx]
                    batch_adv = advantages[batch_idx]
                    batch_ret = returns[batch_idx]

                    result = model.forward(batch_obs)
                    new_lp = model.action_head.log_prob(result["action_logits"], batch_actions)
                    entropy = model.action_head.entropy(result["action_logits"])
                    values = result["value"].squeeze(-1)

                    ratio = torch.exp(new_lp - batch_old_lp)
                    surr1 = ratio * batch_adv
                    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * batch_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.functional.mse_loss(values, batch_ret)
                    entropy_loss = -entropy.mean()
                    loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    total_loss += loss.item()

            if not visualize and episode_rewards:
                print(f"[PPO] Step {global_step:>6} | WinRate: {wins/max(1,wins+losses):.0%} | Loss: {total_loss/ppo_epochs:.4f} | FPS: {global_step / max(1, time.time() - start_time):.0f}")

    except KeyboardInterrupt:
        pass
    finally:
        if live: live.stop()

    # Save checkpoint
    save_path = "checkpoints/ppo/combat_lnn_quick_128dim.pt"
    import os
    os.makedirs("checkpoints/ppo", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n[PPO] Checkpoint saved: {save_path}")
    print(f"[PPO] Done! 128-Dim Model ready.")


if __name__ == "__main__":
    train_quick()
