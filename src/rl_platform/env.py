import gymnasium as gym
from rl_platform.protocol import TCPClient, Handshake
from typing import Any
import numpy as np


class TCPGymEnv(gym.Env):
    def __init__(self, ip: str, port: int):
        super().__init__()
        self.tcp_client = TCPClient(ip, port)
        self.handshake_client = Handshake(self.tcp_client)

        self.env_spec = self.handshake_client.wait_for_handshake()
        if not self.env_spec:
            raise ValueError(f"Failed to get environment spec from {ip}:{port}")

        # M0 Patch
        self.agent_id = self.env_spec.agent_ids[0]

        self.observation_space = self.env_spec.observation_spaces[self.agent_id]
        self.action_space = self.env_spec.action_spaces[self.agent_id]

    def parse_reset_data(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        message_type = data.get("type")
        obs = {}
        info = {}
        if message_type == "reset_result":
            agent_results = data.get("agents")  # {id: {obs: [float]}}}
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
            agent_results = data.get("agents")  # {id: {obs: [float]}}}
            for agent_id, agent_result in agent_results.items():
                obs[agent_id] = np.array(agent_result.get("obs"), dtype=np.float32)
                reward[agent_id] = agent_result.get("reward")
                terminated[agent_id] = agent_result.get("done")
                truncated[agent_id] = (
                    False  # M0 Patch maybe use info into furture to determine if truncated
                )
                info[agent_id] = agent_result.get("info", {})
            episode_info = data.get("global", {})
            return obs, reward, terminated, truncated, info, episode_info
        else:
            raise ValueError(f"data type not step result: {message_type}")

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        self.tcp_client.send_data({"type": "reset"})
        # need to send task spec in the furture Milestone
        data = self.tcp_client.receive_data()
        obs, info = self.parse_reset_data(data)

        # M0 Patch
        obs = obs[self.agent_id]
        info = info[self.agent_id]

        return obs, info

    def step(self, actions) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_value = actions.tolist() if hasattr(actions, 'tolist') else actions
        self.tcp_client.send_data(
            {"type": "step", "actions": {self.agent_id: action_value}}
        )
        data = self.tcp_client.receive_data()
        obs, reward, terminated, truncated, info, episode_info = self.parse_step_data(
            data
        )

        episode_done = episode_info.get("done", False)

        # M0 Patch
        obs = obs[self.agent_id]
        reward = reward[self.agent_id]
        terminated = terminated[self.agent_id] or episode_done  # ONLY FOR M0
        truncated = (
            False  # M0 Patch maybe use info into furture to determine if truncated
        )
        info = info[self.agent_id]

        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self.tcp_client.send_data({"type": "close"})
        except ConnectionError:
            pass
        self.tcp_client.close()
