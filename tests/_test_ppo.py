from trainer_rl import PPOTrainer
import torch

trainer = PPOTrainer("config.yaml")
print(f"Model: {trainer.model.param_count_m:.1f}M params")
print(f"Device: {trainer.device}")
print(f"Env obs shape: {trainer.env.reset().shape}")

# Run a quick 3-step rollout to verify the pipeline
obs = trainer.env.reset()
obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
result = trainer.model.act(obs_t)
actions = result["actions"][0].cpu().numpy().astype(int)
obs2, r, done, trunc, info = trainer.env.step(actions)
print(f"Step 1: reward={r:.3f}, agent_hp={info['agent_hp']:.0f}")

print("PPO TRAINER TEST PASSED")
