from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import structlog

from rl_platform.core.protocols import InferencePolicy
from rl_platform.artifacts.database import Database

log = structlog.get_logger()


@dataclass(frozen=True)
class EvalScenario:
    name: str
    num_episodes: int = 10
    env_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    scenario_name: str
    mean_reward: float
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    per_agent_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_episodes(self) -> int:
        return len(self.episode_rewards)

    @property
    def mean_length(self) -> float:
        if not self.episode_lengths:
            return 0.0
        return sum(self.episode_lengths) / len(self.episode_lengths)


class EvalDriver:

    def __init__(self, database: Database) -> None:
        self._db = database

    def evaluate(
        self,
        policy: InferencePolicy,
        env_fn: Callable[[], gym.Env],
        scenario: EvalScenario,
    ) -> EvalResult:
        # TODO(M4): scenario.env_params is currently unused. Wire it through
        # env.reset(options={"task": ...}) once task-parameterised reset is
        # supported (curriculum task forwarding, M4).
        env = env_fn()
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        try:
            for ep in range(scenario.num_episodes):
                obs, _ = env.reset()
                done = False
                ep_reward = 0.0
                ep_length = 0

                # TODO: add a max_steps guard here to prevent infinite loops
                # against TCP envs that never signal done (e.g. hung UE5 worker).
                while not done:
                    action = policy.get_actions(np.expand_dims(obs, axis=0))
                    action = action[0]
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += float(reward)
                    ep_length += 1
                    done = terminated or truncated

                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
        finally:
            env.close()

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0

        result = EvalResult(
            scenario_name=scenario.name,
            mean_reward=mean_reward,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )

        log.info(
            "evaluation_complete",
            scenario=scenario.name,
            num_episodes=result.num_episodes,
            mean_reward=result.mean_reward,
            mean_length=result.mean_length,
        )
        return result

    def evaluate_and_store(
        self,
        run_id: str,
        checkpoint_id: str | None,
        policy: InferencePolicy,
        env_fn: Callable[[], gym.Env],
        scenario: EvalScenario,
    ) -> EvalResult:
        result = self.evaluate(policy, env_fn, scenario)
        self._db.add_evaluation(
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            scenario=scenario.name,
            results=asdict(result),
        )
        return result
