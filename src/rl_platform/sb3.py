from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv

ALGOS = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
}


class SB3Learner:
    def __init__(self, params: dict) -> None:
        params = dict(params)
        self.algo_name = params.pop("algorithm")
        self.algo_class = ALGOS[self.algo_name]
        self.hps = params
        self.model = None

    def train(
        self,
        vec_env: VecEnv,
        total_steps: int,
        checkpoint_dir: Path,
        save_freq: int,
    ) -> dict[str, float | int]:
        self.model = self.algo_class("MlpPolicy", vec_env, **self.hps)

        callback = None
        if save_freq and save_freq > 0:
            callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=str(checkpoint_dir),
                name_prefix="model",
            )

        try:
            self.model.learn(total_timesteps=total_steps, callback=callback)
        finally:
            vec_env.close()

        results: dict[str, float | int] = {"num_timesteps": self.model.num_timesteps}
        if self.model.ep_info_buffer:
            rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            results["mean_reward"] = float(np.mean(rewards))
        return results

    def save(self, path: Path) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("save() called before train()")
        self.model.save(path)
        return {"algorithm": self.algo_name, "step": self.model.num_timesteps}
