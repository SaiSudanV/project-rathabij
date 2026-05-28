from environment_wrapper import ArenaEnv
import numpy as np

env = ArenaEnv()
obs = env.reset()
print("Obs shape:", obs.shape)

total_r = 0
steps = 0
done = False
trunc = False

while not done and not trunc:
    actions = np.random.randint(0, 2, size=40)
    obs, r, done, trunc, info = env.step(actions)
    total_r += r
    steps += 1

agent_hp = info["agent_hp"]
opp_hp = info["opponent_hp"]
print(f"Episode done in {steps} steps")
print(f"Reward: {total_r:.2f}")
print(f"Agent HP: {agent_hp:.0f} | Opponent HP: {opp_hp:.0f}")
print("ENV TEST PASSED")
