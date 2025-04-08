import numpy as np
import socket
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces

# Import only GymWrapperGeneral.
from gym_wrappers.gym_wrapper_general import GymWrapperGeneral

# -------------------- CONSTANTS -------------------- #
ENV_IP = "127.0.0.1"        # Unreal TCP server IP
ENV_PORT = 7777             # Unreal TCP server port

TOTAL_TIMESTEPS = 1000000   # Total training timesteps
LEARNING_RATE = 3e-4        # Learning rate
GAMMA = 0.99                # Discount factor
N_STEPS = 2048              # Batch size per update
BATCH_SIZE = 128            # Mini-batch size for PPO updates
ENT_COEF = 0.01             # Entropy coefficient
GAE_LAMBDA = 0.95           # GAE lambda
CLIP_RANGE = 0.2            # PPO clipping parameter
VF_COEF = 0.5              # Value function loss coefficient
MAX_GRAD_NORM = 0.5         # Gradient clipping

MODEL_NAME = "model"        # Model filename prefix

# -------------------- FACTORY FUNCTION -------------------- #
def create_gym_env(ip=ENV_IP, port=ENV_PORT):
    # Create a socket and connect.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((ip, port))
        print(f"[Factory] Connected to {ip}:{port}")
    except Exception as e:
        print(f"[Factory] Failed to connect: {e}")
        return None

    # Receive handshake message until the "STEP" delimiter is encountered.
    recv_buffer = ""
    bufsize = 1024
    while "STEP" not in recv_buffer:
        data = sock.recv(bufsize)
        if not data:
            break
        recv_buffer += data.decode('utf-8')
    if "STEP" in recv_buffer:
        handshake, remaining = recv_buffer.split("STEP", 1)
        handshake = handshake.strip()
        print(f"[Factory] Handshake received: {handshake}")
    else:
        print("[Factory] No valid handshake received.")
        sock.close()
        return None

    # Parse the handshake. Expected format:
    # "CONFIG:OBS=<obs_shape>;ACT=<act_shape>;[ENV_TYPE=<type>;ENV_COUNT=<n>]"
    # In this reverted version, we ignore any multi-env configuration.
    try:
        config_body = handshake.split("CONFIG:")[1]
        parts = config_body.split(";")
        obs_part = parts[0]  # e.g., "OBS=7"
        act_part = parts[1]  # e.g., "ACT=6"
        obs_shape = int(obs_part.split("=")[1])
        act_shape = int(act_part.split("=")[1])
        print(f"[Factory] Parsed handshake: obs_shape={obs_shape}, act_shape={act_shape}")
    except Exception as e:
        print(f"[Factory] Error parsing handshake: {e}")
        sock.close()
        return None

    # Always use GymWrapperGeneral.
    env = GymWrapperGeneral(ip=ip, port=port, sock=sock, use_external_handshake=True)
    env.obs_shape = obs_shape
    env.act_shape = act_shape
    env.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
    env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_shape,), dtype=np.float32)
    print("[Factory] Using GymWrapperGeneral.")
    return env

def make_env():
    return create_gym_env()

# -------------------- TRAINING SCRIPT -------------------- #
env = DummyVecEnv([make_env])

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

model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(MODEL_NAME)
print(f"Model saved as '{MODEL_NAME}'")

env.envs[0].send_data("TRAINING_COMPLETE")
print("Training done")
env.close()
