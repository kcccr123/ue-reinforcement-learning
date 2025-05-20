import yaml, sys, pathlib

from sockets.admin_manager import AdminManager
from sockets.socket_factory import create_unreal_socket

from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_wrappers.gym_wrapper_rl_base import GymWrapperRLBase
from gym_wrappers.gym_wrapper_single_env import GymWrapperSingleEnv
from gym_wrappers.gym_wrapper_multi_env import GymWrapperMultiEnv

class sb3_trainer:
    def __init__(self, config_path):
        self.config_path = config_path
        self.prev_model = None
        self.save_path = None

        self.admin = None
        self.vec_env = None
        self.env_ip = None
        self.env_port = None

        self.algo = None
        self.model = None
        self.total_time_steps = 0
        self.checkpoint_freq = 100_000
        self.hyp_params = None
        self.resume_train = False

        self.load_config()


    def load_config(self):

        cfg_file = pathlib.Path(self.config_path)
        if not cfg_file.is_file():
            print(f"Config file not found: {cfg_file}")
            sys.exit(1)

        try:
            with cfg_file.open("r") as f:
                cfg = yaml.safe_load(f) 
        except yaml.YAMLError as e:
            print(f"YAML parse error.\n")
            sys.exit(1)

        self.algo = cfg["model"]

        self.env_ip = cfg["env"]["ip"]
        self.env_port = cfg["env"]["port"]

        self.save_path = cfg["paths"]["save_dir"]
        self.prev_model = cfg["paths"]["previous_model"]

        self.resume_train = cfg.get("load_from_checkpoint", False)
        self.total_time_steps = cfg["total_timesteps"]
        self.checkpoint_freq =cfg["checkpoint_freq"]

        common = cfg["common_parameters"]
        self.hyp_params = dict(
            device         = common["device"],
            learning_rate  = common["learning_rate"],
            gamma          = common["gamma"],
            verbose        = common["verbose"],
        )

        if self.algo != "a2c":
            self.hyp_params.update(batch_size     = common["batch_size"],)

        if self.algo in ("ppo", "a2c"):
            policy = cfg["on_policy"]
            self.hyp_params.update(
                n_steps    = policy["n_steps"],
                gae_lambda = policy["gae_lambda"],
                ent_coef   = policy["ent_coef"],
            )
            self.hyp_params.update(policy.get(self.algo, {})) 
        else:
            policy = cfg["off_policy"]
            self.hyp_params.update(
                buffer_size     = policy["buffer_size"],
                tau             = policy["tau"],
                learning_starts = policy["learning_starts"],
            )
            self.hyp_params.update(policy.get(self.algo, {})) 

    def connect_bridge(self):
        self.admin = AdminManager(ip=self.env_ip, port=self.env_port)
        self.admin.connect()

        # Block until we receive a valid handshake (ENV_TYPE, OBS, ACT, ENV_COUNT).
        # Meanwhile, any other commands that come in will be processed in dispatch.
        env_type, obs_shape, act_shape, env_count = self.admin.wait_for_handshake()
        self.env_setup(env_type, obs_shape, act_shape, env_count)
    
    def env_setup(self, env_type, obs_shape, act_shape, env_count):
        # Check for handshake errors
        if env_type is None:
            print("[Training] Could not complete handshake")
            self.admin.close()
            exit(1) 

        # if observation space or action space is size 0 raise error
        if obs_shape == 0 or act_shape == 0:
            self.admin.close()
            raise ValueError("Handshake error: obs_shape and act_shape must be non-zero"
                            f"Received OBS={obs_shape}, ACT={act_shape}")

        # Create the environment(s) based on env_type
        if env_type == "RLBASE":
            print("[Training] RLBASE => reusing AdminManager socket for single environment")
            # Construct single environment with RLBASE socket
            # RLBaseBridge uses the admin socket directly for simplicity
            def make_env():
                return GymWrapperRLBase(sock=self.admin.sock, obs_shape=obs_shape, act_shape=act_shape)
            self.vec_env = DummyVecEnv([make_env])

        elif env_type == "SINGLE":
            print("[Training] SINGLE => create new socket for single environment")
            def make_env():
                sock = create_unreal_socket(self.env_ip, self.env_port)
                return GymWrapperSingleEnv(sock=sock, obs_shape=obs_shape, act_shape=act_shape)
            self.vec_env = DummyVecEnv([make_env])

        elif env_type == "MULTI":
            print(f"[Training] MULTI => create {env_count} sub-environments")

            def make_env(i):
                sock = create_unreal_socket(self.env_ip, self.env_port)
                return GymWrapperMultiEnv(sock=sock, obs_shape=obs_shape, act_shape=act_shape, env_id=i)
            
            env_fns = [lambda i=i: make_env(i) for i in range(env_count)]
            self.vec_env = SubprocVecEnv(env_fns)

        else:
            raise ValueError("Handshake error: Unknown enviornment type")

    def init_model(self):
        ALGOS = {
            "ppo":  PPO,  "a2c": A2C,
            "sac":  SAC,  "td3": TD3,  "ddpg": DDPG,
            }
        model_class = ALGOS[self.algo]
        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

        if self.resume_train:
            print("\n ===================== Continue training from previous checkpoint ======================\n")
            self.model = model_class.load(self.prev_model, env=self.vec_env, force_reset=False, device=self.hyp_params["device"])
        else:
            print("\n ===================== New training session ======================\n")
            self.model = model_class("MlpPolicy", self.vec_env, **self.hyp_params)

    def train_model(self):
        # Periodic Imaging
        imaging = CheckpointCallback(
                save_freq=self.checkpoint_freq,
                save_path=self.save_path,
                name_prefix="checkpoint"
                )

        # Train the model
        self.model.learn(total_timesteps=self.total_timesteps, reset_num_timesteps= not self.resume_train, callback=imaging)
        
    def save_model(self):
        self.model.save(path=self.save_path)
        print(f"[Training] Model saved to '{self.save_path}'")
    
    def disconnect_bridge(self):
        self.admin.close()
        self.vec_env.close()    
        print("[Training] Done, Disconnected From Bridge")
    

