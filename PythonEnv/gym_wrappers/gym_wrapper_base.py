import socket

class GymWrapperBase:
    """
    A base class that handles TCP communication with Unreal.
    """
    def __init__(self, ip='127.0.0.1', port=7777):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connect()
    
    def connect(self):
        """
        Establish a TCP connection to the Unreal simulation.
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.ip, self.port))
            print(f"[GymWrapperBase] Connected to {self.ip}:{self.port}")
        except Exception as e:
            print(f"[GymWrapperBase] Failed to connect: {e}")

    def disconnect(self):
        """
        Close the TCP connection.
        """
        if self.sock:
            self.sock.close()
            self.sock = None
            print("[GymWrapperBase] Disconnected.")

    def close(self):
        """
        Shuts down the training enviornment.
        """
        # add more enviornment shutdown stuff here in the future
        self.disconnect()
    
    def send_data(self, data: str):
        """
        Send a UTF-8 encoded string over the TCP connection.
        Sends back to Unreal Side
        """
        if self.sock:
            try:
                self.sock.sendall(data.encode('utf-8'))
                print(f"[GymWrapperBase] Sent data: {data}")
            except Exception as e:
                print(f"[GymWrapperBase] Error sending data: {e}")
    
    def receive_data(self, bufsize=1024) -> str:
        """
        Receive data from the Unreal environment through TCP connection and return it as a string.
        """
        if self.sock:
            try:
                data = self.sock.recv(bufsize)
                if data:
                    received = data.decode('utf-8').strip()
                    print(f"[GymWrapperBase] Received data: {received}")
                    return received
            except Exception as e:
                print(f"[GymWrapperBase] Error receiving data: {e}")
        return ""
