import abc
import socket
import gymnasium as gym

class GymWrapperBase(gym.Env, metaclass=abc.ABCMeta):
    """
    An abstract base class for handling TCP communication with Unreal.
    Subclasses must implement step() and reset() from OpenAI Gymnasium.

    If using this class with custom training script and not via the framework factory function,
    initalize with use_external_handshake set to true.
    """
    def __init__(self, ip='127.0.0.1', port=7777, sock=None, use_external_handshake=False):
        self.ip = ip
        self.port = port
        self.sock = None
        self.recv_buffer = ""
        self.obs_shape = 0
        self.act_shape = 0
        
        if use_external_handshake:
            # Use the externally provided socket.
            if sock is None:
                raise ValueError("External handshake requested but no socket provided.")
            self.sock = sock
            print(f"[GymWrapperBase] Using external handshake; handshake values must be set externally.")
        else:
            self.connect()
            self._receive_handshake()

    def connect(self):
        """Establish a TCP connection to the Unreal simulation."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.ip, self.port))
            print(f"[GymWrapperBase] Connected to {self.ip}:{self.port}")
        except Exception as e:
            print(f"[GymWrapperBase] Failed to connect: {e}")

    def _receive_handshake(self):
        """
        Receive and parse the handshake message from Unreal.
        Expected format: "CONFIG:OBS=<obs_shape>;ACT=<act_shape>" ending with "STEP".

        Can be extended in subclasses to include other configs from handshake message.
        """
        handshake = self.receive_data()
        if handshake.startswith("CONFIG:"):
            try:
                config_body = handshake.split("CONFIG:")[1]
                parts = config_body.split(";")
                obs_part = parts[0]  # e.g., "OBS=7"
                act_part = parts[1]  # e.g., "ACT=6"
                self.obs_shape = int(obs_part.split("=")[1])
                self.act_shape = int(act_part.split("=")[1])
                print(f"[GymWrapperBase] Handshake received: obs_shape = {self.obs_shape}, act_shape = {self.act_shape}")
            except Exception as e:
                print(f"[GymWrapperBase] Error parsing handshake: {e}")
                self.disconnect()
        else:
            print("[GymWrapperBase] No valid handshake received; closing connection.")
            self.disconnect()

    def disconnect(self):
        """Close the TCP connection."""
        if self.sock:
            self.sock.close()
            self.sock = None
            print("[GymWrapperBase] Disconnected.")

    def close(self):
        """Shut down the training environment."""
        self.disconnect()
    
    def send_data(self, data: str):
        """Send a UTF-8 encoded string over the TCP connection."""
        if self.sock:
            try:
                self.sock.sendall(data.encode('utf-8'))
                print(f"[GymWrapperBase] Sent data: {data}")
            except Exception as e:
                print(f"[GymWrapperBase] Error sending data: {e}")
    
    def receive_data(self, bufsize=1024) -> str:
        """
        Receive data from the Unreal environment until the "STEP" delimiter is encountered.
        Returns the complete message (delimiter removed).
        """
        if self.sock:
            try:
                while "STEP" not in self.recv_buffer:
                    data = self.sock.recv(bufsize)
                    if not data:
                        break
                    self.recv_buffer += data.decode('utf-8')
                if "STEP" in self.recv_buffer:
                    message, self.recv_buffer = self.recv_buffer.split("STEP", 1)
                    message = message.strip()
                    print(f"[GymWrapperBase] Received data: {message}")
                    return message
            except Exception as e:
                print(f"[GymWrapperBase] Error receiving data: {e}")
        return ""
    
    @abc.abstractmethod
    def step(self, action):
        """Execute one time step in the environment."""
        pass

    @abc.abstractmethod
    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        pass
