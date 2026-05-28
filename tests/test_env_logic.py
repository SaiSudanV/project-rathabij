from environment_complex import ComplexArenaEnv
import numpy as np

try:
    print("Initializing Env...")
    env = ComplexArenaEnv()
    obs = env.reset()
    print(f"Initial Obs Shape: {obs.shape}")
    print(f"Current Theme: {env.dna['theme']}")
    
    for i in range(100):
        action = np.random.randint(0, 12)
        next_obs, reward, done, _, info = env.step(action)
        if i % 20 == 0:
            print(f"Step {i} | Reward: {reward:.4f} | Dist: {env._min_dist:.2f}")
        if done:
            print("Done reached!")
            env.reset()
    print("Test Successful!")
except Exception as e:
    import traceback
    traceback.print_exc()
