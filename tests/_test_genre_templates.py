import cv2
import numpy as np
import time
import os
from environment_complex import ComplexArenaEnv

def test_all_genres():
    env = ComplexArenaEnv()
    genres = list(env.GENRE_MAP.keys())
    
    os.makedirs("genre_tests", exist_ok=True)
    print(f"Starting automated capture for {len(genres)} genres...")
    
    for genre in genres:
        print(f"Capturing: {genre}")
        env.reset(genre=genre)
        
        # Run for 5 frames to stabilize
        img = None
        for _ in range(5):
            env.step(0)
            img = env.render()
        
        filename = f"genre_tests/{genre.replace('-', '_')}.png"
        cv2.imwrite(filename, img)
        print(f"  - Saved to {filename}")

    print("All 20 genres captured in 'genre_tests/' directory.")

if __name__ == "__main__":
    test_all_genres()
