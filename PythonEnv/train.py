import numpy as np

# Stable Baselines3 algorithms
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_wrappers.gym_wrapper_general import GymWrapperGeneral

"""
train.py

Single-file training script:
Defines constants for hyperparameters and environment configuration at the top.

Usage:
  Just run: python train.py
  Modify the constants at the top to suit your needs.
"""

# -------------------- CONSTANTS -------------------- #

ENV_IP = "127.0.0.1"        # IP address of the Unreal TCP server
ENV_PORT = 7777             # Port for the Unreal TCP server

TOTAL_TIMESTEPS = 100000    # How long we train
LEARNING_RATE = 3e-4        # Learning rate for the RL algorithm
GAMMA = 0.99                # Discount factor
N_STEPS = 1024              # Batch size per update (PPO or On-Policy Algos)

MODEL_NAME = "fighter_model"  # Filename prefix for saving the model

# ------------------------------------------------------ #

def make_env():
    """
    Returns a Gym environment instance that connects to Unreal via TCP.
    Modify this if you want to use a different environment or pass extra parameters.
    """
    env = GymWrapperGeneral(ip=ENV_IP, port=ENV_PORT)
    return env

# Initialize the environment
env = DummyVecEnv([make_env])

# Initalize algo (can choose between PPO SAC TD3)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    n_steps=N_STEPS
    # Add other PPO hyperparams here if desired
)

# Train the model
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Save the model
model.save(MODEL_NAME)
print(f"Model saved as '{MODEL_NAME}'")

env.envs[0].send_data("TRAINING_COMPLETE")
print("Training done")

env.close()
