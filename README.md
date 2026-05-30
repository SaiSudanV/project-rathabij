# Project Rathabij

> 🚧 **Work in Progress:** This project is currently in active development. Features, neural network architectures, and game integrations are incomplete and subject to change.

An experimental, endlessly evolving Game AI named after the mythological demon Raktabija-who spawned a clone of himself every time a drop of his blood hit the ground. Built using Liquid Neural Networks (LNN) and Reinforcement Learning (PPO), Rathabij learns from every defeat, adapting dynamically to the player's strategy. It also features an integrated Language Foundation Model (LFM) for real-time, contextual trashtalking.

## Features

- **Liquid Neural Networks (CfC):** Uses compact, continuous-time recurrent networks for the combat brain.
- **Reinforcement Learning (PPO):** Trained using Proximal Policy Optimization with custom curriculum learning.
- **Dynamic Language Brain (LFM2):** Integrates Liquid AI's LFM2-700M model to generate adult-level, contextual trash talk based on game state, emotions, and player speech.
- **Real-Time Voice Pipeline:** Uses Moonshine STT for listening to the player and Kokoro TTS for verbal responses.
- **Godot Integration:** Communicates with a Godot game client over UDP.

## Project Structure

- `model.py` / `lnn_cell.py`: Neural network architecture and Liquid Neural Network (CfC) implementation.
- `trainer_rl.py` / `train_quick.py`: PPO reinforcement learning training scripts.
- `lfm2_handler.py`: Interface for the Liquid AI language model.
- `voice_handler.py`: Full STT -> LFM2 -> TTS pipeline.
- `emotion_engine.py` / `player_profiler.py`: Manages the AI's "mood" and builds a profile of the player's behavior.
- `environment_complex.py` / `environment_wrapper.py`: RL environment definitions.
- `config.yaml`: Centralized configuration for model architecture, training, and language features.
- `godot/`: Contains the Godot game client (if exported).
- `tests/`: Various test scripts for different components.

## Setup & Installation

1. Install Python requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) For the voice pipeline, install Moonshine and Kokoro dependencies.
3. Update `config.yaml` with your preferred settings.

## Running the AI

Run the main game loop / AI bridge:
```bash
python main.py
```

To run a quick training sprint:
```bash
python train_quick.py
```
