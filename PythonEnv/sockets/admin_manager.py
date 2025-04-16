import socket
from collections import deque

class AdminManager:
    """
    Manages the 'admin' TCP connection to Unreal.
    Intended for future extensibility to handle user and 
    system requests while training is underway.
    """

    def __init__(self, ip="127.0.0.1", port=7777, bufsize=1024):
        self.ip = ip
        self.port = port
        self.bufsize = bufsize
        self.sock = None

        self._recv_buffer = ""
        self._msg_queue = deque() 

        # Handshake-related state
        self.env_type = None
        self.obs_shape = 0
        self.act_shape = 0
        self.env_count = 1   

        self.handshake_completed = False

    def connect(self):
        """Establish a TCP connection to the Unreal server."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.ip, self.port))
            print(f"[AdminManager] Connected to {self.ip}:{self.port}")
        except Exception as e:
            print(f"[AdminManager] Connection error: {e}")
            self.sock = None

    def close(self):
        """Close the admin TCP connection."""
        if self.sock:
            self.sock.close()
            self.sock = None
            print("[AdminManager] Connection closed.")

    def receive_msg(self):
        """
        BLOCK until at least one full message (delimited by 'STEP') is received.
        Return that message as a string (with 'STEP' removed).
        If the socket closes prematurely, raises ConnectionError.
        """
        if not self.sock:
            raise ConnectionError("[AdminManager] No socket available.")

        while True:
            # If there's already a queued message, return it immediately:
            if self._msg_queue:
                return self._msg_queue.popleft()

            # Otherwise, read more data from the socket.
            data = self.sock.recv(self.bufsize)
            if not data:
                raise ConnectionError("[AdminManager] Socket closed unexpectedly.")
            self._recv_buffer += data.decode('utf-8')

            # Split out completed messages
            while "STEP" in self._recv_buffer:
                msg, remainder = self._recv_buffer.split("STEP", 1)
                msg = msg.strip()
                self._recv_buffer = remainder
                self._msg_queue.append(msg)

            # If we appended at least one message, we'll return it on the next loop iteration.

    def process_message(self, msg):
        """
        Decide how to handle an incoming message. 
        Possible message types:
          - 'CONFIG:' => handshake
          - others => unknown or logs

        This is a 'dispatch' approach: each type is delegated to a separate method.
        """
        if msg.startswith("CONFIG:"):
            self._handle_config(msg)

        else:
            print(f"[AdminManager] Received unrecognized message: {msg}")
            
        # add more branches to handle more commands in future

    def _handle_config(self, config_msg):
        """
        Handle the handshake message, e.g.:
          "CONFIG:OBS=7;ACT=6;ENV_TYPE=RLBASE;ENV_COUNT=3"
        Once parsed, we store these values in the AdminManager instance.
        Then we mark handshake_completed = True.
        """
        print(f"[AdminManager] Handling handshake: {config_msg}")
        try:
            # everything after 'CONFIG:'
            config_body = config_msg.split("CONFIG:", 1)[1]
            parts = [p.strip() for p in config_body.split(";") if p.strip()]

            # defaults
            self.env_type = "RLBASE"
            self.obs_shape = 0
            self.act_shape = 0
            self.env_count = 1

            for part in parts:
                if part.startswith("OBS="):
                    self.obs_shape = int(part.split("=")[1])
                elif part.startswith("ACT="):
                    self.act_shape = int(part.split("=")[1])
                elif part.startswith("ENV_TYPE="):
                    self.env_type = part.split("=")[1].upper()
                elif part.startswith("ENV_COUNT="):
                    try:
                        self.env_count = int(part.split("=")[1])
                    except:
                        pass

            self.handshake_completed = True
            print(f"[AdminManager] Parsed handshake -> ENV_TYPE={self.env_type}, "
                  f"OBS={self.obs_shape}, ACT={self.act_shape}, ENV_COUNT={self.env_count}")
        except Exception as e:
            print(f"[AdminManager] Error parsing CONFIG: {e}")

    def run_command(self):
        """
        Check if command from Unreal was received, then runs it.
        """
        # CURRENTLY BLOCKS UNTIL A MESSAGE FROM UNREAL WAS RECEIVED
        # IN FUTURE, ADMIN SHOULD BE MOVED TO ITS OWN THREAD, AND THIS SHOULD BE RUN CONTINOUSLY
        # TO CHECK IF ANY COMMANDS WERE RECEIVED
        # FOR NOW AVOID USING OR ITLL JUST BLOCK EVERYTHING
        self.process_message(self.receive_msg())

    def wait_for_handshake(self):
        """
        Blocks until handshake_completed is True. 
        Training cannot start until handshake is received to initalize loop.
        """
        print("[AdminManager] Waiting for handshake...")
        while not self.handshake_completed:
            msg = self.receive_msg()
            self.process_message(msg)

        return (self.env_type, self.obs_shape, self.act_shape, self.env_count)
    

