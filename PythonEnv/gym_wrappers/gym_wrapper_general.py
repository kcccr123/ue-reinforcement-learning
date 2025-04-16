import numpy as np
from gymnasium import spaces
from .gym_wrapper_base import GymWrapperBase

class GymWrapperGeneral(GymWrapperBase):
    """
    A generic Gym environment that uses an existing TCP socket to
    communicate with Unreal. No handshake or connection logic here—
    those tasks are done externally.
    """

    def __init__(self, sock, obs_shape=0, act_shape=0):
        """
        :param sock:       A pre-connected TCP socket
        :param obs_shape:  Number of observation dimensions
        :param act_shape:  Number of action dimensions
        """
        super().__init__(sock=sock, obs_shape=obs_shape, act_shape=act_shape)

        # Define observation/action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_shape,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_shape,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment by sending 'RESET' to Unreal and parsing the initial state.
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
        action_str = ",".join([f"{a:.2f}" for a in action]) if action is not None else "0.0"
        self.send_data(action_str)
        data = self.receive_data()
        obs, reward, done = self._parse_state(data)
        info = {}
        return obs, reward, done, False, info

    def _parse_state(self, data: str):
        """
        Parse the state string from Unreal, expected format:
          "obs0,obs1,...,obsN;reward;done"
        If there's an error, default to zeros, reward=0, done=True.
        """
        try:
            parts = data.split(";")
            parsed_values = []
            for part in parts:
                part = part.strip()
                if "," in part:
                    parsed_values.extend(float(x) for x in part.split(",") if x.strip())
                else:
                    parsed_values.append(float(part))

            if len(parsed_values) < 3:
                raise ValueError("Not enough data in state string.")

            done_val = int(parsed_values.pop())
            reward = parsed_values.pop()
            obs = np.array(parsed_values, dtype=np.float32)
            done = bool(done_val)
            return obs, reward, done

        except Exception as e:
            print("[GymWrapperGeneral] Error parsing state:", e)
            default_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return default_obs, 0.0, True
