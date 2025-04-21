import numpy as np
from gymnasium import spaces
from .gym_wrapper_base import GymWrapperBase

class GymWrapperSingleEnv(GymWrapperBase):
    """
    A Gym environment wrapper that speaks the SingleEnvBridge protocol:
      - On reset: sends "ACT=RESET" to Unreal, then waits for a keyed response
      - On step: sends "ACT=<a0>,<a1>,..." to Unreal, then waits for a keyed response
      - Expects responses of the form:
          "OBS=<obs0>,<obs1>,...;REW=<reward>;DONE=<0|1>"
      - All I/O uses the base class’s send_data / receive_data methods
    """

    def __init__(self, sock, obs_shape=0, act_shape=0):
        """
        :param sock:       A pre-connected TCP socket
        :param obs_shape:  Number of observation dimensions
        :param act_shape:  Number of action dimensions
        """
        super().__init__(sock=sock, obs_shape=obs_shape, act_shape=act_shape)

        # Define observation and action spaces
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
        Reset the environment by telling UE to reset and then parsing
        the first state message.
        Returns:
            observation (np.ndarray), info (dict)
        """
        self.send_data("ACT=RESET")
        data = self.receive_data()
        obs, reward, done = self._parse_state(data)
        return obs, {}

    def step(self, action):
        """
        Send the chosen action to UE as "ACT=<a0>,<a1>,..."
        Then parse back the resulting state.
        Returns:
            observation (np.ndarray),
            reward (float),
            done (bool),
            truncated (False),
            info (dict)
        """
        # format comma‑separated floats
        action_vals = ",".join(f"{a:.2f}" for a in action) if action is not None else ""
        # prefix with ACT=
        self.send_data(f"ACT={action_vals}")

        data = self.receive_data()
        obs, reward, done = self._parse_state(data)
        return obs, reward, done, False, {}

    def _parse_state(self, data: str):
        """
        Parse a SingleEnvBridge state string of the form:
            "OBS=<obs0>,<obs1>,...;REW=<reward>;DONE=<0|1>"
        Returns:
            obs (np.ndarray), reward (float), done (bool)
        """
        try:
            parts = [seg.strip() for seg in data.split(";")]
            kv = dict(part.split("=", 1) for part in parts if "=" in part)

            obs_str  = kv.get("OBS", "")
            rew_str  = kv.get("REW", "0")
            done_str = kv.get("DONE", "1")

            obs    = np.fromstring(obs_str, sep=",", dtype=np.float32) \
                     if obs_str else np.zeros(self.obs_shape, dtype=np.float32)
            reward = float(rew_str)
            done   = bool(int(done_str))

            return obs, reward, done

        except Exception as e:
            print(f"[GymWrapperSingleEnv] Error parsing state '{data}': {e}")
            default_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return default_obs, 0.0, True
