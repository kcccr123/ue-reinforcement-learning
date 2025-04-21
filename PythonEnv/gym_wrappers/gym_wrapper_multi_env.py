# gym_wrapper_multi_env.py

import numpy as np
from gymnasium import spaces
from .gym_wrapper_base import GymWrapperBase

class GymWrapperMultiEnv(GymWrapperBase):
    """
    A Gym environment wrapper for the MultiEnvBridge protocol:
      - On reset: sends "ACT=RESET;ENV=<id>" to Unreal
      - On step:   sends "ACT=<a0>,<a1>,...;ENV=<id>" to Unreal
      - Expects responses of the form:
          "OBS=<obs0>,<obs1>,...;REW=<reward>;DONE=<0|1>;ENV=<id>"
      - Uses the base class’s send_data / receive_data to handle framing.
    """

    def __init__(self, sock, obs_shape=0, act_shape=0, env_id=0):
        """
        :param sock:       A pre-connected TCP socket to the MultiTcpConnection server
        :param obs_shape:  Number of observation dimensions per environment
        :param act_shape:  Number of action dimensions per environment
        :param env_id:     Integer index (0 ≤ env_id < ENV_COUNT) for this sub-environment
        """
        super().__init__(sock=sock, obs_shape=obs_shape, act_shape=act_shape)
        self.env_id = env_id

        # Define Box spaces for vector observations & actions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_shape,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.act_shape,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset this environment instance.
        Sends "ACT=RESET;ENV=<id>" and parses the returned state.
        Returns:
            obs (np.ndarray), info (dict)
        """
        # tell UE to reset this specific env
        self.send_data(f"ACT=RESET;ENV={self.env_id}")
        data = self.receive_data()
        obs, reward, done = self._parse_state(data)
        return obs, {}

    def step(self, action):
        """
        Take one step for this env instance.
        Sends "ACT=<...>;ENV=<id>" and parses the returned state.
        Returns:
            obs (np.ndarray),
            reward (float),
            done (bool),
            truncated (False),
            info (dict)
        """
        # format the action vector
        vals = ",".join(f"{a:.2f}" for a in action) if action is not None else ""
        # include the env index
        self.send_data(f"ACT={vals};ENV={self.env_id}")

        data = self.receive_data()
        obs, reward, done = self._parse_state(data)
        return obs, reward, done, False, {}

    def _parse_state(self, data: str):
        """
        Parse a state string from MultiEnvBridge:
            "OBS=<o0>,...;REW=<r>;DONE=<0|1>;ENV=<id>"
        Returns:
            obs   (np.ndarray),
            reward(float),
            done  (bool)
        """
        try:
            parts = [seg.strip() for seg in data.split(";")]
            kv = {k: v for k, v in (p.split("=", 1) for p in parts if "=" in p)}

            obs_str  = kv.get("OBS", "")
            rew_str  = kv.get("REW", "0")
            done_str = kv.get("DONE", "1")

            obs    = (np.fromstring(obs_str, sep=",", dtype=np.float32)
                      if obs_str else np.zeros(self.obs_shape, dtype=np.float32))
            reward = float(rew_str)
            done   = bool(int(done_str))

            return obs, reward, done

        except Exception as e:
            print(f"[GymWrapperMultiEnv] Error parsing '{data}': {e}")
            default_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return default_obs, 0.0, True
