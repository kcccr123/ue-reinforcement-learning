import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .gym_wrapper_base import GymWrapperBase

class GymWrapperPawn(gym.Env, GymWrapperBase):
    """
    A Gym environment implementation for controlling a single pawn in Unreal.
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
        
        # obs mapping:
        # 0 -> position
        # 1 -> quat
        
        info = {}
        
        return obs, reward, done, False, info

    
    def parse_state(self, data: str) -> tuple:
        """
        Parse the state string from Unreal.
        Expected format: "x,y,z;qw,qx,qy,qz;reward;done"
        """
        try:
            parts = data.split(";")
            # pos_str => "x,y,z"
            pos_str = parts[0]
            # quat_str => "qw,qx,qy,qz"
            quat_str = parts[1]
            # reward
            reward_str = parts[2]
            # done or not
            done_str = parts[3]
            
            # Parse floats
            pos = [float(v) for v in pos_str.split(",")]
            quat = [float(v) for v in quat_str.split(",")]
            reward = float(reward_str)
            done = bool(int(done_str))
            # Combine into a single array
            obs = np.array(pos + quat, dtype=np.float32)
            return obs, reward, done
        except Exception as e:
            print("[GymWrapperPawn] Error parsing state:", e)
            # Return a default tuple (observation, reward, done)
            default_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return default_obs, 0.0, True
