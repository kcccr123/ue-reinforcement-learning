import abc
import gymnasium as gym
import socket

class GymWrapperBase(gym.Env, metaclass=abc.ABCMeta):
    """
    An abstract base class for handling TCP communication with Unreal.
    This class expects:
      - A pre-connected socket (sock)
      - Known obs_shape and act_shape
    No handshake or network connection logic is contained here.
    TCP uses and expects "\n" (newline char) as delimiter.
    """

    def __init__(self, sock, obs_shape=0, act_shape=0):
        """
        :param sock:       A pre-connected socket for sending/receiving data.
        :param obs_shape:  Number of observation dimensions
        :param act_shape:  Number of action dimensions
        """
        super().__init__()

        if not sock or not isinstance(sock, socket.socket):
            raise ValueError("A valid, pre-connected socket must be provided.")

        self.sock = sock
        self.recv_buffer = ""
        self.obs_shape = obs_shape
        self.act_shape = act_shape

    def disconnect(self):
        """Close the TCP connection."""
        if self.sock:
            try:
                self.sock.close()
                print("[GymWrapperBase] Disconnected socket.")
            except Exception as e:
                print(f"[GymWrapperBase] Error closing socket: {e}")
        self.sock = None

    def close(self):
        """Shut down the training environment."""
        self.disconnect()

    def send_data(self, data: str):
        """Send a UTF-8 encoded string over the TCP connection."""
        data += "\n"
        if self.sock:
            try:
                self.sock.sendall(data.encode('utf-8'))
                print(f"[GymWrapperBase] Sent data: {data}")
            except Exception as e:
                print(f"[GymWrapperBase] Error sending data: {e}")

    def receive_data(self, bufsize=1024) -> str:
        """
        Receive data from the Unreal environment until the newline delimiter is encountered.
        Returns the complete message (delimiter removed).
        """
        if not self.sock:
            return ""

        try:
            while "\n" not in self.recv_buffer:
                data = self.sock.recv(bufsize)
                if not data:
                    break
                self.recv_buffer += data.decode('utf-8')

            if "\n" in self.recv_buffer:
                message, self.recv_buffer = self.recv_buffer.split("\n", 1)
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
