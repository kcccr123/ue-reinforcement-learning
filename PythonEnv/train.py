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

TOTAL_TIMESTEPS = 1000000    # Total training timesteps
LEARNING_RATE = 3e-4         # Reduced learning rate for stability
GAMMA = 0.99                 # Discount factor
N_STEPS = 2048               # Increased batch size per update
BATCH_SIZE = 128             # Mini-batch size for PPO updates
ENT_COEF = 0.01              # Entropy coefficient for improved exploration
GAE_LAMBDA = 0.95            # Generalized Advantage Estimation lambda
CLIP_RANGE = 0.2             # PPO clipping parameter
VF_COEF = 0.5                # Value function loss coefficient
MAX_GRAD_NORM = 0.5          # Gradient clipping

MODEL_NAME = "model"  # Filename prefix for saving the model

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

# Initialize PPO algorithm with detailed hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    ent_coef=ENT_COEF,
    gae_lambda=GAE_LAMBDA,
    clip_range=CLIP_RANGE,
    vf_coef=VF_COEF,
    max_grad_norm=MAX_GRAD_NORM
)

# Train the model
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Save the model
model.save(MODEL_NAME)
print(f"Model saved as '{MODEL_NAME}'")

env.envs[0].send_data("TRAINING_COMPLETE")
print("Training done")

env.close()