import argparse as ap
import sys, pathlib, queue, signal
from bridge import AdminTHD, envTHD
import utils as help

from stable_baselines3.common.callbacks import CheckpointCallback

ENV_IP  = "127.0.0.1"        # Unreal TCP server IP
ENV_PORT = 7777         # Unreal TCP server port

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

    algo, hps, paths, resume, total_timesteps, checkpt_freq = help.load_config(cfg_file)

    model_cls = help.get_model_class(algo)
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
        model = model_cls.load(paths["previous_model"], env=vec_env, force_reset=False, device=hps["device"])
    else:
        model = model_cls("MlpPolicy", vec_env, **hps)

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
