import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .gym_wrapper_base import GymWrapperBase

class GymWrapperGeneral(GymWrapperBase):
    """
    A dynamic Gym environment implementation for general use.
    Inherits TCP logic from GymWrapperBase and the gym.Env interface.
    """
    def __init__(self, ip='127.0.0.1', port=7777, sock=None, use_external_handshake=False):
        # Initialize gym.Env.
        gym.Env.__init__(self)
        # Initialize TCP connection using the modified GymWrapperBase.
        GymWrapperBase.__init__(self, ip, port, sock, use_external_handshake)
        
        # Define observation and action spaces based on handshake values.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_shape,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment by sending a 'RESET' command to Unreal and parsing the initial state.
        Returns (observation, info).
        """
        self.send_data("RESET")
        data = self.receive_data()
        obs, reward, done = self._parse_state(data)
        return obs, {}
    
    def step(self, action):
        """
        Sends the action to Unreal and receives the new state.
        Returns (observation, reward, done, truncated, info).
        """
        action_str = ",".join([f"{a:.2f}" for a in action])
        self.send_data(action_str)
        data = self.receive_data()
        obs, reward, done = self._parse_state(data)
        info = {}
        return obs, reward, done, False, info
    
    def _parse_state(self, data: str) -> tuple:
        """
        Parse the state string from Unreal.
        Expected format: "1.0,2.0,3.0,0.0,0.0,0.0,1.0;10.0;1"
        where the first part (comma-separated) is the observation,
        the second part is the reward, and the third part is the done flag.
        """
        try:
            parts = data.split(";")
            parsed_values = []
            for part in parts:
                part = part.strip()
                if ',' in part:
                    parsed_values.extend([float(x) for x in part.split(",") if x.strip() != ""])
                else:
                    parsed_values.append(float(part))
            if len(parsed_values) < 3:
                raise ValueError("Not enough data in state string.")
            done_value = int(parsed_values.pop())
            reward = parsed_values.pop()
            obs = np.array(parsed_values, dtype=np.float32)
            done = bool(done_value)
            return obs, reward, done
        except Exception as e:
            print("[GymWrapperGeneral] Error parsing state:", e)
            default_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return default_obs, 0.0, True
