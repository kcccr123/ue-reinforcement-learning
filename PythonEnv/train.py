import argparse as ap

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from sockets.admin_manager import AdminManager
from sockets.socket_factory import create_unreal_socket

from gym_wrappers.gym_wrapper_rl_base import GymWrapperRLBase
from gym_wrappers.gym_wrapper_single_env import GymWrapperSingleEnv
from gym_wrappers.gym_wrapper_multi_env import GymWrapperMultiEnv

# -------------------- CONSTANTS -------------------- #
ENV_IP = "127.0.0.1"        # Unreal TCP server IP
ENV_PORT = 7777             # Unreal TCP server port

TOTAL_TIMESTEPS = 128   # Total training timesteps
LEARNING_RATE = 3e-3        # Learning rate
GAMMA = 0.99                # Discount factor
N_STEPS = 1024              # Batch size per update
BATCH_SIZE = 128            # Mini-batch size for PPO updates
ENT_COEF = 0.01             # Entropy coefficient
GAE_LAMBDA = 0.95           # GAE lambda
CLIP_RANGE = 0.2            # PPO clipping parameter
VF_COEF = 0.5               # Value function loss coefficient
MAX_GRAD_NORM = 0.5         # Gradient clipping

MODEL_NAME = "model"        # Model filename prefix


# -------------------- TRAINING SCRIPT -------------------- #
if __name__ == "__main__":

# ____Training Options ____ #
    parser = ap.ArgumentParser()
    parser.add_argument("--load_cp", type=str, help="Path to checkpoint zip file if it is required to continue training from last checkpoint.")

    args = parser.parse_args()

    load_checkpoint = False
    checkpoint_path = ""

    if args.load_cp is not None:
        load_checkpoint = True
        checkpoint_path = args.load_cp

    # Initialize and connect the AdminManager
    admin = AdminManager(ip=ENV_IP, port=ENV_PORT)
    admin.connect()

    # Block until we receive a valid handshake (ENV_TYPE, OBS, ACT, ENV_COUNT).
    # Meanwhile, any other commands that come in will be processed in dispatch.
    env_type, obs_shape, act_shape, env_count = admin.wait_for_handshake()

    # Check for handshake errors
    if env_type is None:
        print("[Training] Could not complete handshake")
        admin.close()
        exit(1) 

    # if observation space or action space is size 0 raise error
    if obs_shape == 0 or act_shape == 0:
        admin.close()
        raise ValueError("Handshake error: obs_shape and act_shape must be non-zero"
                         f"Received OBS={obs_shape}, ACT={act_shape}")

    # Create the environment(s) based on env_type
    if env_type == "RLBASE":
        print("[Training] RLBASE => reusing AdminManager socket for single environment")
        # Construct single environment with RLBASE socket
        # RLBaseBridge uses the admin socket directly for simplicity
        def make_env():
            return GymWrapperRLBase(sock=admin.sock, obs_shape=obs_shape, act_shape=act_shape)
        vec_env = DummyVecEnv([make_env])

    elif env_type == "SINGLE":
        print("[Training] SINGLE => create new socket for single environment")
        def make_env():
            sock = create_unreal_socket(ENV_IP, ENV_PORT)
            return GymWrapperSingleEnv(sock=sock, obs_shape=obs_shape, act_shape=act_shape)
        vec_env = DummyVecEnv([make_env])

    elif env_type == "MULTI":
        print(f"[Training] MULTI => create {env_count} sub-environments")

        def make_env(i):
            sock = create_unreal_socket(ENV_IP, ENV_PORT)
            return GymWrapperMultiEnv(sock=sock, obs_shape=obs_shape, act_shape=act_shape, env_id=i)
        
        env_fns = [lambda i=i: make_env(i) for i in range(env_count)]
        vec_env = SubprocVecEnv(env_fns)

    else:
        raise ValueError("Handshake error: Unknown enviornment type")
    
    # Create the model (by default ppo, add your own if you want)
    if load_checkpoint:
        print("\n =====================Continue training from previous checkpoint======================\n")
        model = PPO.load(checkpoint_path, env=vec_env, force_reset=False)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
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

    # Periodic Imaging
    imaging = CheckpointCallback(
    save_freq=100_000,
    save_path="checkpoints",
    name_prefix="ppo_unreal",
    )

    # Train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps= not load_checkpoint, callback=imaging)
        
    model.save(MODEL_NAME)
    print(f"[Training] Model saved as '{MODEL_NAME}'")

    # TODO: Notify Unreal that training is complete through admin connection:
    # NEED TO ADD SEND COMMAND FOR ADMIN SOCKET, CURRENTLY USER JUST MANUALLY CLOSES UNREAL
    
    admin.close()
    vec_env.close()
    print("[Training] Done")
