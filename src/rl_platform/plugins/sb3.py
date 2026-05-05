from rl_platform.core.protocols import TrainingCallback, InferencePolicy
from rl_platform.core.specifications import EnvSpec, TaskSpec
from typing import Callable, Any
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
import numpy as np
from pathlib import Path

ALGOS = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
}


class SB3Plugin:
    def configure(self, env_spec: EnvSpec, config: dict) -> None:
        if env_spec.is_multi_agent:
            raise NotImplementedError("multi-agent training is M1+ only")
        self.env_spec = env_spec
        params = dict(config)
        algo_name = params.pop("algorithm")
        self.algo_class = ALGOS[algo_name]
        self.algo_name = algo_name
        self.hps = params
        self._trained_model = None 

    def train(
        self,
        worker_addrs: list[tuple[str, int]] | None,
        env_fns: list[Callable[[], gym.Env]] | None,
        task: TaskSpec | None,
        total_steps: int,
        callbacks: list[TrainingCallback],
    ) -> dict[str, float]:

        if env_fns is None:
            raise ValueError("env_fns is required (worker_addrs not supported yet)")
        if task is not None:
            raise NotImplementedError("task-parameterized reset is not supported until M4")
        # Monitor wraps each env so SB3 populates info["episode"] on each
        # episode termination, which is required for the callback bridge to
        # forward episode_reward / episode_length metrics via on_step().
        vec_env = DummyVecEnv([lambda fn=fn: Monitor(fn()) for fn in env_fns])

        self._trained_model = self.algo_class("MlpPolicy", vec_env, **self.hps)
        callback_bridge = SB3CallbackBridge(callbacks)
        try:
            self._trained_model.learn(total_timesteps=total_steps, callback=callback_bridge)
        finally:
            vec_env.close()

        results = {"num_timesteps": self._trained_model.num_timesteps}
        if callback_bridge.episode_rewards:
            results["mean_reward"] = float(np.mean(callback_bridge.episode_rewards))
            results["mean_length"] = float(np.mean(callback_bridge.episode_lengths))
            results["last_100_mean_reward"] = float(
                np.mean(callback_bridge.episode_rewards[-100:])
            )
            results["total_episodes"] = len(callback_bridge.episode_rewards)
        return results

    def load(self, path: Path) -> None:
        #need to set up the env again if want to train with this
        self._trained_model = self.algo_class.load(path, force_reset=False)

    def save(self, path: Path) -> dict[str, Any]:
        if self._trained_model is None:
            raise RuntimeError("save() called before train() or load()")
        self._trained_model.save(path)
        return {"algorithm": self.algo_name, "step": self._trained_model.num_timesteps}

    def get_policy(self) -> InferencePolicy:
        model = self._trained_model

        if model is None:
            raise RuntimeError("get_policy() called before train() or load()")
        class _SB3Policy:
            def get_actions(self, obs_batch: np.ndarray) -> np.ndarray:
                actions, _ = model.predict(obs_batch, deterministic=True)
                return actions

        return _SB3Policy()


class SB3CallbackBridge(BaseCallback):
    # NOTE: TrainingCallback.on_update_end has no direct SB3 source
    def __init__(self, platform_callbacks: list[TrainingCallback]):
        super().__init__()
        self.platform_callbacks = platform_callbacks
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        cur_step = self.num_timesteps
        metrics = {}
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                r = float(info["episode"]["r"])
                l = int(info["episode"]["l"])
                metrics["episode_reward"] = r
                metrics["episode_length"] = float(l)
                self.episode_rewards.append(r)
                self.episode_lengths.append(l)
        for cb in self.platform_callbacks:
            if not cb.on_step(cur_step, metrics):
                return False
        return True

    def _on_rollout_end(self) -> None:
        cur_step = self.num_timesteps
        metrics = {}
        for cb in self.platform_callbacks:
            cb.on_batch_end(cur_step, metrics)

    def _on_training_end(self) -> None:
        for cb in self.platform_callbacks:
            cb.on_training_end()
