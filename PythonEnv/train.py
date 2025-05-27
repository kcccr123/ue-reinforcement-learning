import argparse as ap
import sys, pathlib, queue, yaml
from bridge import AdminTHD, envTHD

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG

ENV_IP  = "127.0.0.1"        # Unreal TCP server IP
ENV_PORT = 7777         # Unreal TCP server port

def load_config(cfg_file):
    try:
        with cfg_file.open("r") as f:
            cfg = yaml.safe_load(f) 
    except yaml.YAMLError as e:
        print(f"YAML parse error.\n")
        sys.exit(1)

    ALGOS = {
    "ppo":  PPO,  "a2c": A2C,
    "sac":  SAC,  "td3": TD3,  "ddpg": DDPG,
    }

    algo_name = cfg["model"]
    try:
        algo = ALGOS[algo_name]
    except KeyError as e:
        raise ValueError(f"Unsupported algo '{algo_name}'. Allowed: {list(ALGOS)}") from e
    


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
    checkpt_freq =cfg["checkpoint_freq"]

    pathlib.Path(paths["save_dir"]).mkdir(parents=True, exist_ok=True)


    return algo, hps, paths, resume, total_timesteps, checkpt_freq

# -------------------- TRAINING SCRIPT -------------------- #
def main():
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

    algo, hps, paths, resume, total_timesteps, checkpt_freq = load_config(cfg_file)

    print("HERE")
    meta_q = queue.Queue(maxsize=1)
    print("a")
    master = AdminTHD(ENV_IP, ENV_PORT, meta_q)
    print("b")
    master.start()

    print("HERE2")
    meta_data = meta_q.get() #blocking for master
    print("HERE3")
    if meta_data["env_type"] is None or meta_data["obs_shape"] == 0 or meta_data["act_shape"] == 0:
        print("[ERROR] Invalid handshake – aborting.")
        sys.exit(1)
    
    vec_env = envTHD(ENV_IP, ENV_PORT, meta_data).build()

    if resume:
        print("\n =====================Continue training from previous checkpoint======================\n")
        model = algo.load(paths["previous_model"], env=vec_env, force_reset=False, device=hps["device"])
    else:
        model = algo("MlpPolicy", vec_env, **hps)

    # Periodic Imaging
    imaging = CheckpointCallback(
    save_freq=checkpt_freq,
    save_path=paths["save_dir"],
    name_prefix="checkpoint"
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps= not resume, callback=imaging)
    model.save(path=paths["save_dir"])
    print(f"[Training] Model saved to '{paths['save_dir']}'")

    meta_data["admin"].close()
    vec_env.close()

    
if __name__ == "__main__":
    main()
