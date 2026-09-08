from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import (
    InteractionType,
    TensorDictModule,
    TensorDictSequential,
    set_interaction_type,
)
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from gymnasium.spaces.utils import flatdim

from rl_platform.specs import EnvSpec

log = structlog.get_logger()

CHECKPOINT_FORMAT = 1


@dataclass
class PPOConfig:
    hidden_dim: int = 256
    depth: int = 2
    lr: float = 3e-4
    gamma: float = 0.99
    lmbda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    critic_coeff: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 10
    minibatch_size: int = 256
    normalize_advantage: bool = True
    share_params: bool = True
    centralised_critic: bool = False
    device: str = "cpu"


@dataclass
class TorchRLConfig:
    frames_per_batch: int = 2048
    ppo: PPOConfig = field(default_factory=PPOConfig)


def build_team_map(env_spec: EnvSpec) -> dict[str, list[str]]:
    teams: dict[str, list[str]] = {}

    for agent_id in env_spec.agent_ids:
        obs_space = env_spec.observation_spaces[agent_id]
        act_space = env_spec.action_spaces[agent_id]
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))

        key = f"g{obs_dim}x{act_dim}"
        team = env_spec.teams.get(agent_id)
        if team is not None:
            key = f"{team}_{key}"

        teams.setdefault(key, []).append(agent_id)

    return teams


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.dim() == mask.dim() + 1 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.shape != mask.shape:
        return value.mean()
    selected = value[mask]
    if selected.numel() == 0:
        return value.sum() * 0.0
    return selected.mean()


class TeamPPO:
    def __init__(
        self,
        team: str,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        act_low: torch.Tensor,
        act_high: torch.Tensor,
        config: PPOConfig,
    ):
        self.team = team
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        self.device = torch.device(config.device)
        self.act_low = torch.as_tensor(act_low, device=self.device)
        self.act_high = torch.as_tensor(act_high, device=self.device)

        actor_net = nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=obs_dim,
                n_agent_outputs=2 * act_dim,
                n_agents=n_agents,
                centralised=False,
                share_params=config.share_params,
                device=self.device,
                depth=config.depth,
                num_cells=config.hidden_dim,
                activation_class=nn.Tanh,
            ),
            NormalParamExtractor(),
        )

        self.actor = ProbabilisticActor(
            module=TensorDictModule(
                actor_net,
                in_keys=[(team, "observation")],
                out_keys=[(team, "loc"), (team, "scale")],
            ),
            in_keys=[(team, "loc"), (team, "scale")],
            out_keys=[(team, "action")],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": self.act_low, "high": self.act_high},
            return_log_prob=True,
            log_prob_key=(team, "action_log_prob"),
            default_interaction_type=InteractionType.RANDOM,
        )

        self.critic = TensorDictModule(
            MultiAgentMLP(
                n_agent_inputs=obs_dim,
                n_agent_outputs=1,
                n_agents=n_agents,
                centralised=config.centralised_critic,
                share_params=config.share_params,
                device=self.device,
                depth=config.depth,
                num_cells=config.hidden_dim,
                activation_class=nn.Tanh,
            ),
            in_keys=[(team, "observation")],
            out_keys=[(team, "state_value")],
        )

        self.advantage_module = GAE(
            gamma=config.gamma,
            lmbda=config.lmbda,
            value_network=self.critic,
            average_gae=False,
        )
        self.advantage_module.set_keys(
            advantage=(team, "advantage"),
            value_target=(team, "value_target"),
            value=(team, "state_value"),
            reward=(team, "reward"),
            done=(team, "done"),
            terminated=(team, "terminated"),
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=config.clip_epsilon,
            entropy_coeff=config.entropy_coeff,
            critic_coeff=config.critic_coeff,
            normalize_advantage=False,
            reduction="none",
        )
        self.loss_module.set_keys(
            advantage=(team, "advantage"),
            value_target=(team, "value_target"),
            value=(team, "state_value"),
            action=(team, "action"),
            sample_log_prob=(team, "action_log_prob"),
            reward=(team, "reward"),
            done=(team, "done"),
            terminated=(team, "terminated"),
        )

        self.optimizer = torch.optim.Adam(
            self.loss_module.parameters(), lr=config.lr
        )

    def compute_advantage(self, data: TensorDict) -> None:
        with torch.no_grad():
            self.advantage_module(data)

        if not self.config.normalize_advantage:
            return

        advantage = data[self.team, "advantage"]
        mask = data[self.team, "mask"].unsqueeze(-1).expand_as(advantage)
        valid = advantage[mask]
        if valid.numel() > 1:
            data[self.team, "advantage"] = (advantage - valid.mean()) / valid.std().clamp_min(1e-6)

    def update_step(self, batch: TensorDict) -> dict[str, float]:
        out = self.loss_module(batch)
        mask = batch[self.team, "mask"]

        loss_objective = masked_mean(out["loss_objective"], mask)
        loss_critic = masked_mean(out["loss_critic"], mask)
        loss_entropy = masked_mean(out["loss_entropy"], mask)
        loss = loss_objective + loss_critic + loss_entropy

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        return {
            "loss_objective": float(loss_objective.detach()),
            "loss_critic": float(loss_critic.detach()),
            "loss_entropy": float(loss_entropy.detach()),
            "grad_norm": float(grad_norm.detach()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])


class TorchRLPlatform:
    def __init__(self, env, config: TorchRLConfig | None = None):
        self.config = config or TorchRLConfig()
        self.env_spec = env.env_spec
        self.team_map = build_team_map(self.env_spec)

        self.env = PettingZooWrapper(
            env,
            use_mask=True,
            return_state=False,
            done_on_any=False,
            group_map=self.team_map,
            device=self.config.ppo.device,
        )

        self.teams = {}
        for team, agent_ids in self.team_map.items():
            action_spec = self.env.full_action_spec[team, "action"]
            self.teams[team] = TeamPPO(
                team=team,
                n_agents=len(agent_ids),
                obs_dim=self.env.observation_spec[team, "observation"].shape[-1],
                act_dim=action_spec.shape[-1],
                act_low=action_spec.space.low,
                act_high=action_spec.space.high,
                config=self.config.ppo,
            )

        self.policy = TensorDictSequential(
            *[team_ppo.actor for team_ppo in self.teams.values()]
        )

        self.buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                self.config.frames_per_batch, device=self.config.ppo.device
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.config.ppo.minibatch_size,
        )

        self.collector = Collector(
            self.env,
            self.policy,
            frames_per_batch=self.config.frames_per_batch,
            total_frames=-1,
            device=self.config.ppo.device,
        )
        self.collector_iter = iter(self.collector)
        self.total_steps = 0

        log.info(
            "torchrl_platform_init",
            env_id=self.env_spec.env_id,
            teams=self.team_map,
        )

    def collect(self) -> TensorDict:
        data = next(self.collector_iter) #should iterate over an entire batch of data not 1 frame
        self.total_steps += int(data.batch_size.numel())
        return data

    def learn(self, data: TensorDict) -> dict[str, dict[str, float]]:
        for team_ppo in self.teams.values():
            team_ppo.compute_advantage(data)

        self.buffer.extend(data.reshape(-1))

        totals = {team: {} for team in self.teams}
        updates = 0
        minibatches = max(
            1, self.config.frames_per_batch // self.config.ppo.minibatch_size
        )

        for _ in range(self.config.ppo.epochs):
            for _ in range(minibatches):
                batch = self.buffer.sample()
                for team, team_ppo in self.teams.items():
                    stats = team_ppo.update_step(batch)
                    for key, value in stats.items():
                        totals[team][key] = totals[team].get(key, 0.0) + value
                updates += 1

        self.buffer.empty()

        results: dict[str, dict[str, float]] = {}
        for team, stats in totals.items():
            results[team] = {
                key: value / max(updates, 1) for key, value in stats.items()
            }
            reward = data["next", team, "reward"]
            mask = data[team, "mask"]
            results[team]["mean_step_reward"] = float(
                masked_mean(reward, mask).detach()
            )

        return results

    def train(self, total_steps: int) -> dict[str, Any]:
        history = []
        while self.total_steps < total_steps:
            data = self.collect()
            metrics = self.learn(data)
            history.append(metrics)
            log.info("torchrl_update", step=self.total_steps, metrics=metrics)
        return {"num_timesteps": self.total_steps, "history": history}

    @torch.no_grad()
    def evaluate(self, episodes: int = 1) -> dict[str, float]:
        rollouts = []
        with set_interaction_type(InteractionType.DETERMINISTIC):
            for _ in range(episodes):
                rollouts.append(self.env.rollout(1000, policy=self.policy))

        results = {}
        for team in self.teams:
            totals = [
                float(
                    masked_mean(
                        rollout["next", team, "reward"],
                        rollout[team, "mask"],
                    )
                )
                for rollout in rollouts
            ]
            results[team] = float(np.mean(totals))
        return results

    def save(self, path) -> dict[str, Any]:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "algorithm": "PPO",
            "env_id": self.env_spec.env_id,
            "total_steps": self.total_steps,
            "team_map": self.team_map,
            "config": asdict(self.config),
            "specs": {
                team: {
                    "n_agents": gp.n_agents,
                    "obs_dim": gp.obs_dim,
                    "act_dim": gp.act_dim,
                    "act_low": gp.act_low.cpu(),
                    "act_high": gp.act_high.cpu(),
                }
                for team, gp in self.teams.items()
            },
            "teams": {
                team: gp.state_dict() for team, gp in self.teams.items()
            },
        }
        torch.save(payload, path)
        return {
            "algorithm": "PPO",
            "step": self.total_steps,
            "teams": list(self.teams),
        }

    def load(self, path) -> None:
        state = torch.load(path, weights_only=False)
        teams = state["teams"] if "teams" in state else state

        saved_map = None
        if isinstance(state, dict):
            saved_map = state.get("team_map", state.get("group_map"))
        if saved_map is not None and saved_map != self.team_map:
            raise ValueError(
                f"checkpoint team map {saved_map} does not match the connected "
                f"environment {self.team_map}"
            )

        unknown = [team for team in teams if team not in self.teams]
        if unknown:
            raise ValueError(
                f"checkpoint contains teams {unknown} absent from the "
                f"connected environment {list(self.teams)}"
            )

        for team, team_state in teams.items():
            self.teams[team].load_state_dict(team_state)

        self.total_steps = state.get("total_steps", self.total_steps)
        log.info(
            "torchrl_checkpoint_loaded",
            path=str(path),
            step=self.total_steps,
            teams=list(teams),
        )

    @torch.no_grad()
    def act(
        self, observations: dict[str, np.ndarray], deterministic: bool = True
    ) -> dict[str, np.ndarray]:
        mode = (
            InteractionType.DETERMINISTIC if deterministic else InteractionType.RANDOM
        )
        actions: dict[str, np.ndarray] = {}

        for team, agent_ids in self.team_map.items():
            if not any(agent_id in observations for agent_id in agent_ids):
                continue

            team_ppo = self.teams[team]
            batch = torch.zeros(
                1, len(agent_ids), team_ppo.obs_dim, device=team_ppo.device
            )
            for index, agent_id in enumerate(agent_ids):
                if agent_id in observations:
                    batch[0, index] = torch.as_tensor(
                        observations[agent_id],
                        dtype=torch.float32,
                        device=team_ppo.device,
                    )

            data = TensorDict(
                {
                    team: TensorDict(
                        {"observation": batch}, batch_size=[1, len(agent_ids)]
                    )
                },
                batch_size=[1],
            )
            with set_interaction_type(mode):
                team_ppo.actor(data)

            selected = data[team, "action"][0].cpu().numpy()
            for index, agent_id in enumerate(agent_ids):
                if agent_id in observations:
                    actions[agent_id] = selected[index]

        return actions

    def close(self) -> None:
        self.collector.shutdown()
        self.env.close()
