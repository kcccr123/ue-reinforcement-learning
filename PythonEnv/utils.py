
from __future__ import annotations

import argparse as ap
import sys, pathlib, yaml, queue, threading, signal
from functools import partial
from typing import Dict, Any

from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG

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
    checkpt_freq =cfg["checkpoint_freq"]

    pathlib.Path(paths["save_dir"]).mkdir(parents=True, exist_ok=True)


    return algo, hps, paths, resume, total_timesteps, checkpt_freq

def get_model_class(algo):
    ALGOS = {
    "ppo":  PPO,  "a2c": A2C,
    "sac":  SAC,  "td3": TD3,  "ddpg": DDPG,
    }

    try:
        ans = ALGOS[algo]
    except KeyError as e:
        raise ValueError(f"Unsupported algo '{algo}'. Allowed: {list(ALGOS)}") from e
    
    return ans

