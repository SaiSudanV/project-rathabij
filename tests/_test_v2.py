"""Quick verification of the v2 rebuild."""
import sys
import numpy as np
sys.path.insert(0, ".")

# 1. Model check
print("=" * 50)
print("TEST 1: Model instantiation")
import torch
from model import CombatLNN, ACTIONS, NUM_ACTIONS

model = CombatLNN(state_dim=32, hidden_size=128, num_actions=12, num_cfc_layers=2)
print(f"  Actions: {NUM_ACTIONS} -> {ACTIONS}")
print(f"  Params: {model.param_count_m:.2f}M ({model.param_count:,})")
print(f"  PASS: <2M? {'YES' if model.param_count < 2_000_000 else 'NO'}")

# Test forward
x = torch.randn(4, 32)
out = model.act(x)
print(f"  Action shape: {out['actions'].shape} dtype={out['actions'].dtype}")
print(f"  Actions: {out['actions'].tolist()}")
print(f"  Value shape: {out['value'].shape}")
print(f"  Emotions shape: {out['emotions'].shape}")
assert out["actions"].shape == (4,), f"Expected (4,), got {out['actions'].shape}"
assert all(0 <= a < 12 for a in out["actions"].tolist()), "Actions out of range!"
print("  ✅ Model OK\n")

# 2. Environment check
print("=" * 50)
print("TEST 2: Environment")
from environment_complex import ComplexArenaEnv

env = ComplexArenaEnv(state_dim=32)
obs = env.reset()
print(f"  Obs shape: {obs.shape}")
assert obs.shape == (32,), f"Expected (32,), got {obs.shape}"

# Step with categorical action
for action_id in range(12):
    obs, reward, done, trunc, info = env.step(action_id)
    if done:
        obs = env.reset()
print(f"  All 12 actions executed OK")
print(f"  Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
print(f"  Non-zero: {np.count_nonzero(obs)}/{len(obs)} ({np.count_nonzero(obs)/len(obs)*100:.0f}%)")
print("  ✅ Environment OK\n")

# 3. Episode test
print("=" * 50)
print("TEST 3: Random episodes")
wins = 0
total_rewards = []
for ep in range(10):
    obs = env.reset()
    ep_reward = 0
    for step in range(500):
        action = np.random.randint(0, 12)
        obs, reward, done, trunc, info = env.step(action, opponent_actions=np.random.randint(0, 12))
        ep_reward += reward
        if done:
            if info.get("opponent_hp", 1) <= 0:
                wins += 1
            break
    total_rewards.append(ep_reward)

print(f"  10 episodes: avg_reward={np.mean(total_rewards):.2f}")
print(f"  Wins: {wins}/10")
print("  ✅ Episodes OK\n")

print("=" * 50)
print("ALL TESTS PASSED ✅")
