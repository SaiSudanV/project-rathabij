"""
visual_test_v2.py — Local Predator Visualizer
=============================================
- Single-env real-time rendering.
- Shows AI intent vs. Randomized Body Mapping.
- Displays Phase 1 Navigation logic.
"""

import torch
import cv2
import numpy as np
import time
from model import CombatLNN
from environment_complex import ComplexArenaEnv, ACT_IDLE

def run_visual_test():
    # 1. Setup
    device = torch.device("cpu")
    state_dim = 32
    model = CombatLNN(state_dim=state_dim, hidden_size=192).to(device)
    env = ComplexArenaEnv(state_dim=state_dim)
    env.current_phase = 2
    
    # 2. Show the "Body" Discovery
    print("\n" + "="*50)
    print("PREDATOR INITIALIZED: RANDOMIZED BODY MAPPING")
    print("="*50)
    mapping = env.dna['mapping']
    for act_id, slot in mapping.items():
        print(f"Action ID {act_id:<2} -> Raw Button Slot {slot}")
    print("="*50)
    print("Note: In Phase 1, only movement actions (1-5) are active.")
    
    obs = env.reset()
    done = False
    
    print("\n[LIVE] Opening Arena View... (Press 'q' to quit)")
    
    while not done:
        # AI Logic
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.act(obs_t)
        
        action = out["actions"].item()
        
        # Step Env
        obs, reward, done, _, info = env.step(action)
        
        # Render
        frame = env.render_frame()
        
        # Add a local debug overlay
        cv2.putText(frame, f"Action: {action} (Slot {mapping.get(action, 'N/A')})", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"Reward: {reward:.2f}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Show Window
        cv2.imshow("Mahoraga Predator v2.1 — Local Arena", frame)
        
        # Slow down for human eyes (30 FPS)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\n[TEST ENDED]")

if __name__ == "__main__":
    run_visual_test()
