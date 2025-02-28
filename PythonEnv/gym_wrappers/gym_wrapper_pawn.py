import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_wrapper_base import GymWrapperBase

class GymWrapperPawn(gym.Env, GymWrapperBase):
    """
    A Gym environment for controlling a single pawn in Unreal.
    Inherits TCP logic from GymWrapperBase and the Gym Env interface from gym.Env.
    """
    def __init__(self, ip='127.0.0.1', port=7777):
        """
        Initialize the Pawn environment:
        - Connect to Unreal via TCP.
        - Define observation and action spaces.
        """
        # Initialize Gym's Env
        gym.Env.__init__(self)
        # Initialize our base TCP connection
        GymWrapperBase.__init__(self, ip, port)
        
        # determines number of data points coming in from unreal
        # shape=(7,) 
        # => 7 floats coming in from unreal => 3 for position (x,y,z) + 4 for quaternion (w,x,y,z)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        # shape(2, ) => sends two floats back to unreal
        # 2 floats, one for thurst one for torque to apply to pawn.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    
    def reset(self):
        """
        Resets the environment by sending a 'RESET' command to Unreal
        and parsing the initial state. Returns the initial observation.
        """
        self.send_data("RESET")
        data = self.receive_data()
        obs = self.parse_state(data)
        return obs
    
    def step(self, action):
        """
        The main interaction point: sends the action to Unreal,
        then receives the new state. Returns (observation, reward, done, info).
        For now, reward is 0.0 and done is False by default, if we're deferring
        reward logic to the Unreal side.
        """
        # Convert the action array to a comma-separated string, e.g. "0.50,0.25"
        action_str = ",".join([f"{a:.2f}" for a in action])
        self.send_data(action_str)
        
        # Receive new state
        data = self.receive_data()
        obs = self.parse_state(data)
        
        # If rewards are computed in Unreal, you'd parse them here too.
        # For now, we assume the reward is 0 and done is False.
        reward = 0.0
        done = False
        info = {}
        
        return obs, reward, done, info

    
    def parse_state(self, data: str) -> np.ndarray:
        """
        Parse the state string from Unreal.
        Example format: "x,y,z;qw,qx,qy,qz"
        Returns a NumPy array of shape (7,)
        """
        try:
            parts = data.split(";")
            # pos_str => "x,y,z"
            pos_str = parts[0]
            # quat_str => "qw,qx,qy,qz"
            quat_str = parts[1]
            
            # Parse floats
            pos = [float(v) for v in pos_str.split(",")]
            quat = [float(v) for v in quat_str.split(",")]
            
            # Combine into a single (7,) array
            obs = np.array(pos + quat, dtype=np.float32)
            return obs
        except Exception as e:
            print("[GymWrapperPawn] Error parsing state:", e)
            # Return a default array if parsing fails
            return np.zeros(self.observation_space.shape, dtype=np.float32)
