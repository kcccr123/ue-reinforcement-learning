import threading, time

from sockets.admin_manager import AdminManager
from sockets.socket_factory import create_unreal_socket

from gym_wrappers.gym_wrapper_rl_base import GymWrapperRLBase
from gym_wrappers.gym_wrapper_single_env import GymWrapperSingleEnv
from gym_wrappers.gym_wrapper_multi_env import GymWrapperMultiEnv

class AdminTHD(threading.Thread):

    def __init__(self, ip, port, out_q):
        super().__init__(daemon=True, name= "AdminThread")
        self.ip, self.port = ip, port
        self.q = out_q
        self.admin = None

    def run(self):
        self.admin = AdminManager(ip=self.ip, port=self.port)
        self.admin.connect()
        env_type, obs_shape, act_shape, env_count = self.admin.wait_for_handshake()
        self.q.put({
            "env_type": env_type,
            "obs_shape": obs_shape,
            "act_shape": act_shape,
            "env_count": env_count,
            "admin": self.admin, 
        })

        while True:
            try:
                _ = self.admin.get_cmd_nonblocking()
            except Exception:
                time.sleep(0.3) #sleep for 2 seconds 
                pass
class envTHD:
    def __init__(self, ip, port, meta):
        self.ip, self.port = ip, port
        self.meta = meta
        self.obs_shape = meta["obs_shape"]
        self.act_shape = meta["act_shape"]
        self.env_type  = meta["env_type"]
        self.n_envs    = meta["env_count"] if meta["env_type"] == "MULTI" else 1
    
    def init_single_env(self):
        obs_shape, act_shape = self.obs_shape, self.act_shape
        ip, port = self.ip, self.port
        admin_sock           = self.meta["admin"].sock
        env_type             = self.env_type

        def  _init():
            if env_type == "RLBASE":
                print("[Training] RLBASE => reusing Admin socket")
                return GymWrapperRLBase(
                    sock=admin_sock,
                    obs_shape=obs_shape,
                    act_shape=act_shape,
                )
            elif env_type == "SINGLE":  # SINGLE
                print("[Training] SINGLE => new socket")
                sock = create_unreal_socket(ip, port)
                return GymWrapperSingleEnv(
                    sock=sock,
                    obs_shape=obs_shape,
                    act_shape=act_shape,
                )
        return _init

    def init_multi_env(self, idx):
        ip, port          = self.ip, self.port
        obs_shape         = self.obs_shape
        act_shape         = self.act_shape

        def _init():
            print(f"[Training] MULTI => create {idx} sub-environments")
            sock = create_unreal_socket(ip, port)
            return GymWrapperMultiEnv(
                sock=sock,
                obs_shape=obs_shape,
                act_shape=act_shape,
                env_id=idx,
            )
        return _init

    def build(self):
        is_multi = False
        if self.env_type == "MULTI":
            is_multi = True
            env_fns = [self.init_multi_env(i) for i in range(self.n_envs)]
        else:
            env_fns = [self.init_single_env()]
        
        return env_fns, is_multi

        
