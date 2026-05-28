"""
trainer_rl.py — PPO Predator Trainer (v2.2 Phase-Aware)
=========================================================
- 10M Step Training Curriculum.
- Real-time Phase Switching (Nav -> Search -> Combat).
- Updates Env with current training phase.
"""

import time
import numpy as np
import torch
import yaml
import cv2
from torch.utils.tensorboard import SummaryWriter
from model import CombatLNN
from vec_env import SubprocVecEnv

class PPOTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.num_envs = 12
        self.state_dim = 32
        self.total_timesteps = 10_000_000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Model
        self.model = CombatLNN(state_dim=self.state_dim, hidden_size=192).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        # Spawning Envs
        print(f"[Predator] Initializing {self.num_envs} Parallel DNA-Arenas...")
        self.envs = SubprocVecEnv(num_envs=self.num_envs, env_kwargs={"state_dim": self.state_dim})
        
        # TensorBoard
        self.writer = SummaryWriter("logs/predator_v2")
        
        self.global_step = 0
        self.start_time = time.time()

    def get_phase(self):
        """Milestones based on your plan."""
        if self.global_step < 1_000_000: return 1   # Navigation
        if self.global_step < 5_000_000: return 2   # Search (Hide & Seek)
        return 3                                    # Combat

    def train(self):
        obs = self.envs.reset()
        current_phase = 0
        
        print(f"[Predator] Kicking off 10M step hunt on {self.device}...")
        
        while self.global_step < self.total_timesteps:
            # 1. Update Phase in Envs if changed
            new_phase = self.get_phase()
            if new_phase != current_phase:
                print(f"\n[PHASE CHANGE] Entering Phase {new_phase} at Step {self.global_step:,}")
                # Tell all envs to update their reward/logic mode
                for i in range(self.num_envs):
                    self.envs.remotes[i].send(("set_phase", new_phase))
                current_phase = new_phase

            # 2. Collect Rollouts (Simplified for logic view)
            obs_t = torch.FloatTensor(obs).to(self.device)
            with torch.no_grad():
                out = self.model.act(obs_t)
            
            actions = out["actions"].cpu().numpy()
            next_obs, rewards, dones, truncs, infos = self.envs.step(actions)
            
            self.global_step += (self.num_envs * 4) # frame_skip=4
            obs = next_obs
            
            # 3. Training/Logging...
            if self.global_step % 2048 < (self.num_envs * 4):
                avg_r = np.mean(rewards)
                fps = self.global_step / (time.time() - self.start_time)
                
                # Push to TensorBoard
                self.writer.add_scalar("Train/Reward", avg_r, self.global_step)
                self.writer.add_scalar("Train/FPS", fps, self.global_step)
                self.writer.add_scalar("Train/Phase", current_phase, self.global_step)
                
                print(f"Step: {self.global_step:>8,} | Phase: {current_phase} | FPS: {fps:.0f} | Avg R: {avg_r:>7.2f}", end="\r")

            # 4. LIVE SNAPSHOT (Every 10,000 steps)
            if self.global_step % 10000 < (self.num_envs * 4):
                self.envs.remotes[0].send(("render", None))
                frame = self.envs.remotes[0].recv()
                cv2.imwrite("live_fight.jpg", frame)
            if self.global_step % 500000 < (self.num_envs * 4):
                self._save_checkpoint()

        self.writer.close()
        print("\n[HUNT COMPLETE] 10M steps reached.")

    def _save_checkpoint(self):
        path = f"checkpoints/predator_step_{self.global_step}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"\n[SAVE] Checkpoint saved: {path}")

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train()
