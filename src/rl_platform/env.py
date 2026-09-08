import gymnasium as gym
import pettingzoo as pz
from rl_platform.protocol import TCPClient, Handshake
from typing import Any
import numpy as np


class TCPEnvBase:
    def __init__(self, ip: str, port: int):
        self.tcp_client = TCPClient(ip, port)
        self.handshake_client = Handshake(self.tcp_client)

        self.env_spec = self.handshake_client.wait_for_handshake()
        if not self.env_spec:
            raise ValueError(f"Failed to get environment spec from {ip}:{port}")

    def encode_action(self, action: Any) -> Any:
        return action.tolist() if hasattr(action, "tolist") else action

    def parse_reset_data(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        message_type = data.get("type")
        obs = {}
        info = {}
        if message_type == "reset_result":
            agent_results = data.get("agents")
            for agent_id, agent_result in agent_results.items():
                obs[agent_id] = np.array(agent_result.get("obs"), dtype=np.float32)
                info[agent_id] = agent_result.get("info", {})
            return obs, info
        else:
            raise ValueError(f"data type not reset result: {message_type}")

    def parse_step_data(self, data: dict[str, Any]) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
        dict[str, Any],
    ]:
        message_type = data.get("type")
        obs = {}
        reward = {}
        terminated = {}
        truncated = {}
        info = {}
        episode_info = {}
        if message_type == "step_result":
            agent_results = data.get("agents")
            for agent_id, agent_result in agent_results.items():
                obs[agent_id] = np.array(agent_result.get("obs"), dtype=np.float32)
                reward[agent_id] = agent_result.get("reward")
                terminated[agent_id] = agent_result.get("done")
                truncated[agent_id] = False
                info[agent_id] = agent_result.get("info", {})
            episode_info = data.get("global", {})
            return obs, reward, terminated, truncated, info, episode_info
        else:
            raise ValueError(f"data type not step result: {message_type}")

    def all_agents_present(
        self, results: dict[str, Any], expected: list[str], message_type: str
    ) -> None:
        missing = [agent_id for agent_id in expected if agent_id not in results]
        if missing:
            raise ValueError(
                f"{message_type} from {self.tcp_client.ip}:{self.tcp_client.port} "
                f"missing agents {missing}"
            )

    def close(self):
        try:
            self.tcp_client.send_data({"type": "close"})
        except ConnectionError:
            pass
        self.tcp_client.close()


class TCPGymEnv(TCPEnvBase, gym.Env):
    def __init__(self, ip: str, port: int):
        super().__init__(ip, port)

        if len(self.env_spec.agent_ids) != 1:
            self.close()
            raise ValueError(
                f"TCPGymEnv supports single-agent environments only, but "
                f"{ip}:{port} declared {len(self.env_spec.agent_ids)} agents "
                f"({self.env_spec.agent_ids}); use TCPPZParallelEnv instead"
            )

        self.agent_id = self.env_spec.agent_ids[0]

        self.observation_space = self.env_spec.observation_spaces[self.agent_id]
        self.action_space = self.env_spec.action_spaces[self.agent_id]

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        self.tcp_client.send_data({"type": "reset"})
        data = self.tcp_client.receive_data()
        obs, info = self.parse_reset_data(data)

        self.all_agents_present(obs, [self.agent_id], "reset_result")

        return obs[self.agent_id], info[self.agent_id]

    def step(self, actions) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.tcp_client.send_data(
            {"type": "step", "actions": {self.agent_id: self.encode_action(actions)}}
        )
        data = self.tcp_client.receive_data()
        obs, reward, terminated, truncated, info, episode_info = self.parse_step_data(
            data
        )

        self.all_agents_present(obs, [self.agent_id], "step_result")

        episode_done = episode_info.get("done", False)

        return (
            obs[self.agent_id],
            reward[self.agent_id],
            terminated[self.agent_id] or episode_done,
            truncated[self.agent_id],
            info[self.agent_id],
        )


class TCPPZParallelEnv(TCPEnvBase, pz.ParallelEnv):
    metadata = {"render_modes": [], "name": "tcp_pz_parallel_env"}

    def __init__(self, ip: str, port: int):
        super().__init__(ip, port)

        self.possible_agents = list(self.env_spec.agent_ids)
        self.agents = []
        self.observation_spaces = dict(self.env_spec.observation_spaces)
        self.action_spaces = dict(self.env_spec.action_spaces)
        self.render_mode = None

    def observation_space(self, agent: str) -> gym.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]

    def reset(
        self, seed=None, options=None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        self.tcp_client.send_data({"type": "reset"})
        data = self.tcp_client.receive_data()
        obs, info = self.parse_reset_data(data)

        self.all_agents_present(obs, self.possible_agents, "reset_result")
        self.agents = list(self.possible_agents)

        observations = {agent_id: obs[agent_id] for agent_id in self.agents}
        infos = {agent_id: info[agent_id] for agent_id in self.agents}

        return observations, infos

    def step(self, actions: dict[str, Any]) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        unknown = [
            agent_id for agent_id in actions if agent_id not in self.possible_agents
        ]
        if unknown:
            raise ValueError(f"received actions for unknown agents {unknown}")

        actions = {
            agent_id: action
            for agent_id, action in actions.items()
            if agent_id in self.agents
        }

        missing = [agent_id for agent_id in self.agents if agent_id not in actions]
        if missing:
            raise ValueError(f"missing actions for active agents {missing}")

        payload = {
            agent_id: self.encode_action(action)
            for agent_id, action in actions.items()
        }
        self.tcp_client.send_data({"type": "step", "actions": payload})
        data = self.tcp_client.receive_data()
        obs, reward, terminated, truncated, info, episode_info = self.parse_step_data(
            data
        )

        stepped = list(self.agents)
        self.all_agents_present(obs, stepped, "step_result")

        episode_done = episode_info.get("done", False)

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        for agent_id in stepped:
            observations[agent_id] = obs[agent_id]
            rewards[agent_id] = reward[agent_id]
            terminations[agent_id] = terminated[agent_id] or episode_done
            truncations[agent_id] = truncated[agent_id]
            infos[agent_id] = info[agent_id]

        self.agents = [
            agent_id
            for agent_id in stepped
            if not (terminations[agent_id] or truncations[agent_id])
        ]

        return observations, rewards, terminations, truncations, infos

    def close(self):
        self.agents = []
        super().close()
