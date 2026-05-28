import cv2
import numpy as np
import time
from environment_complex import ComplexArenaEnv

def play_test():
    env = ComplexArenaEnv()
    # Using the GENRE_MAP keys to ensure all 20 are accessible
    genres = list(env.GENRE_MAP.keys())
    genre_idx = 0
    
    cv2.namedWindow("Overhaul Playground", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Overhaul Playground", 1000, 700)
    
    env.reset(genre=genres[genre_idx])
    
    print("\n" + "="*30)
    print("  OVERHAUL PLAYGROUND ACTIVE")
    print("="*30)
    print("W, A, S, D : Move / Turn")
    print("N          : Next Genre")
    print("R          : Reset Arena")
    print("Q          : Quit")
    print("="*30)
    
    while True:
        btns = [False]*12
        key = cv2.waitKey(20) & 0xFF # Faster polling
        
        # Mapping common keys
        if key == ord('w'): btns[3] = True
        if key == ord('s'): btns[4] = True
        if key == ord('a'): btns[1] = True
        if key == ord('d'): btns[2] = True
        
        if key == ord('n'):
            genre_idx = (genre_idx + 1) % len(genres)
            env.reset(genre=genres[genre_idx])
            print(f"[{genre_idx+1}/20] Switching to: {genres[genre_idx]}")
            
        if key == ord('r'):
            env.reset(genre=genres[genre_idx])
            
        if key == ord('q'):
            break
            
        env.step(btns)
        img = env.render()
        cv2.imshow("Overhaul Playground", img)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_test()
