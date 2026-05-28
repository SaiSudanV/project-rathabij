"""
vec_env.py — Vectorized Parallel Environments (v2.2 Predator)
==============================================================
- Handles "set_phase" command for curriculum switching.
- Syncs training phase with Mahoraga/Hebbian adaptation.
"""

import multiprocessing as mp
import numpy as np

def _worker(remote, parent_remote, env_kwargs, frame_skip=4):
    parent_remote.close()
    from environment_complex import ComplexArenaEnv
    from mahoraga_core import MahoragaWheel, HebbianPlasticity

    env = ComplexArenaEnv(**env_kwargs)
    wheel = MahoragaWheel()
    plasticity = HebbianPlasticity(num_actions=12)
    obs = env.reset()

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError: break

        if cmd == "step":
            action, opp_action = data
            total_r = 0.0
            for _ in range(frame_skip):
                # Apply Hebbian Modulation before stepping? 
                # (Neural network does this, but we track penalties here)
                obs, r, d, t, info = env.step(action, opponent_actions=opp_action)
                total_r += r
                
                # Hebbian adaptation based on damage
                if r < -0.1: plasticity.penalize(action)
                if r > 0.5: plasticity.reward(action)
                
                if d or t: 
                    obs = env.reset()
                    break
            remote.send((obs, total_r, d, t, info))

        elif cmd == "set_phase":
            env.current_phase = data
            print(f"  [Env Worker] Phase updated to {data}")

        elif cmd == "render":
            remote.send(env.render_frame())

        elif cmd == "reset":
            remote.send(env.reset())

        elif cmd == "close":
            remote.close(); break

class SubprocVecEnv:
    def __init__(self, num_envs, env_kwargs=None):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []
        for w, r in zip(self.work_remotes, self.remotes):
            p = mp.Process(target=_worker, args=(w, r, env_kwargs or {}), daemon=True)
            p.start()
            w.close()
            self.processes.append(p)

    def step(self, actions):
        for i, r in enumerate(self.remotes):
            r.send(("step", (actions[i], None)))
        res = [r.recv() for r in self.remotes]
        o, r, d, t, i = zip(*res)
        return np.stack(o), np.array(r), np.array(d), np.array(t), i

    def reset(self):
        for r in self.remotes: r.send(("reset", None))
        return np.stack([r.recv() for r in self.remotes])
