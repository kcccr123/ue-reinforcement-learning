import argparse as ap
import yaml, sys, pathlib

from stable_baselines3 import PPO, A2C, SAC, TD3, DQN, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from sockets.admin_manager import AdminManager
from sockets.socket_factory import create_unreal_socket

from gym_wrappers.gym_wrapper_rl_base import GymWrapperRLBase
from gym_wrappers.gym_wrapper_single_env import GymWrapperSingleEnv
from gym_wrappers.gym_wrapper_multi_env import GymWrapperMultiEnv

# -------------------- ENV CONSTANTS -------------------- #
ENV_IP = "127.0.0.1"        # Unreal TCP server IP
ENV_PORT = 7777             # Unreal TCP server port
# -------------------- TRAINING SCRIPT -------------------- #

def load_config(cfg_file):
    try:
        with cfg_file.open("r") as f:
            cfg = yaml.safe_load(f) 
    except yaml.YAMLError as e:
        print(f"YAML parse error.\n")
        sys.exit(1)

    algo = cfg["model"]

    common = cfg["common_parameters"]
    hps = dict(
        device         = common["device"],
        learning_rate  = common["learning_rate"],
        gamma          = common["gamma"],
        verbose        = common["verbose"],
    )

    if algo != "a2c":
        hps.update(batch_size     = common["batch_size"],)

    if algo in ("ppo", "a2c"):
        policy = cfg["on_policy"]
        hps.update(
            n_steps    = policy["n_steps"],
            gae_lambda = policy["gae_lambda"],
            ent_coef   = policy["ent_coef"],
        )
        hps.update(policy.get(algo, {})) 
    else:
        policy = cfg["off_policy"]
        hps.update(
            buffer_size     = policy["buffer_size"],
            tau             = policy["tau"],
            learning_starts = policy["learning_starts"],
        )
        hps.update(policy.get(algo, {})) 
    paths = cfg["paths"]
    resume = cfg.get("load_from_checkpoint", False)
    total_timesteps = cfg["total_timesteps"]

    pathlib.Path(paths["save_dir"]).mkdir(parents=True, exist_ok=True)


    return algo, hps, paths, resume, total_timesteps

if __name__ == "__main__":

# ____Training Options ____ #
    parser = ap.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")

    args = parser.parse_args()
    if args.config is None:
        print("Path to config file not found.\n")
        sys.exit(1)

    cfg_file = pathlib.Path(args.config)
    if not cfg_file.is_file():
        print(f"Config file not found: {cfg_file}")
        sys.exit(1)


    ALGOS = {
    "ppo":  PPO,  "a2c": A2C,
    "sac":  SAC,  "td3": TD3,  "ddpg": DDPG,
    }

    algo, hparams, paths, load_checkpoint, total_timesteps = load_config(cfg_file)
    try:
        model_class = ALGOS[algo]
        print(algo+"\n\n\n\n")
    except KeyError:
        raise ValueError(f"Algo '{algo}' not in {list(ALGOS)}")


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
        model = model_class.load(paths["previous_model"], env=vec_env, force_reset=False, device=hparams["device"])
    else:
        model = model_class("MlpPolicy", vec_env, **hparams)

    # Periodic Imaging
    imaging = CheckpointCallback(
    save_freq=100_000,
    save_path=paths["save_dir"],
    name_prefix="checkpoint"
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps= not load_checkpoint, callback=imaging)
        
    model.save(path=paths["save_dir"])
    print(f"[Training] Model saved to '{paths['save_dir']}'")

    # TODO: Notify Unreal that training is complete through admin connection:
    # NEED TO ADD SEND COMMAND FOR ADMIN SOCKET, CURRENTLY USER JUST MANUALLY CLOSES UNREAL
    
    admin.close()
    vec_env.close()
    print("[Training] Done")
