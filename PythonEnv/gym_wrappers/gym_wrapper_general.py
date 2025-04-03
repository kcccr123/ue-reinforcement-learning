import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .gym_wrapper_base import GymWrapperBase

class GymWrapperGeneral(gym.Env, GymWrapperBase):
    """
    A dynamic Gym environment implementation for general use with RLBaseBridge subclasses.
    Inherits TCP logic from GymWrapperBase and the Gym Env interface from gym.Env.
    """
    def __init__(self, ip='127.0.0.1', port=7777):
        """
        Initialize the Pawn environment:
        - Connect to Unreal via TCP.
        - The handshake in the base class sets self.obs_shape and self.act_shape.
        - Define observation and action spaces dynamically.
        """
        # Initialize Gym's Env
        gym.Env.__init__(self)
        # Initialize our base TCP connection (which also processes the handshake)
        GymWrapperBase.__init__(self, ip, port)
        
        # Create action and observation spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_shape,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment by sending a 'RESET' command to Unreal
        and parsing the initial state. Returns (observation, info).
        """
        self.send_data("RESET")
        data = self.receive_data()
        obs, reward, done = self.parse_state(data)
        return obs, {}
            
    def step(self, action):
        """
        Sends the action to Unreal, then receives the new state.
        Returns (observation, reward, done, info).
        """
        # Convert the action array to a comma-separated string, e.g. "0.50,0.25"
        action_str = ",".join([f"{a:.2f}" for a in action])
        self.send_data(action_str)
        
        # Receive new state
        data = self.receive_data()
        obs, reward, done = self.parse_state(data)

        
        info = {}
        
        return obs, reward, done, False, info

    
    def parse_state(self, data: str) -> tuple:
        """
        Dynamically parse the state string from Unreal.
        The state string is expected to be semicolon separated. The last two values are
        interpreted as reward and done respectively, while all preceding numbers form the observation.
        
        For example, a state string might look like:
        "1.0,2.0,3.0,0.0,0.0,0.0,1.0;10.0;1"
        which would parse into:
        obs = [1.0,2.0,3.0,0.0,0.0,0.0,1.0] (as a numpy array),
        reward = 10.0,
        done = True
        """
        try:
            # Split the state string by semicolons
            parts = data.split(";")
            parsed_values = []
            
            # Iterate over all parts and split by comma if applicable.
            for part in parts:
                # Remove any extraneous whitespace
                part = part.strip()
                if ',' in part:
                    # Extend the list with each parsed float value
                    parsed_values.extend([float(x) for x in part.split(",") if x.strip() != ""])
                else:
                    # Try to parse a single float value (could be reward or done flag)
                    parsed_values.append(float(part))
            
            # Ensure we have at least three numbers (obs + reward + done)
            if len(parsed_values) < 3:
                raise ValueError("Not enough data in state string.")

            # Assume that the last two values are reward and done.
            done_value = int(parsed_values.pop())  # Remove and convert last element to int then bool
            reward = parsed_values.pop()  # Remove reward
            
            # Convert the remaining values to a numpy array for the observation.
            obs = np.array(parsed_values, dtype=np.float32)
            done = bool(done_value)
            
            return obs, reward, done

        except Exception as e:
            print("[GymWrapperPawn] Error parsing state:", e)
            default_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return default_obs, 0.0, True
