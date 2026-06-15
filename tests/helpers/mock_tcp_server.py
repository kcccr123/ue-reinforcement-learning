import socket
from typing import Any
import struct
from rl_platform.infra.communication import MsgSerializer
import gymnasium as gym
import numpy as np

class MockTcpServer:
    def __init__(self, ip: str, port: int):
        self.env = gym.make("CartPole-v1")
        self.ip = ip
        self.port = port
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listen_sock.bind((self.ip, self.port))
        self.listen_sock.listen(1)
        self.response_sock = None
        self.serializer = MsgSerializer()
        self.handshake_sent = False
    
    def start(self):
        """Accept and serve connections in a loop until the listen socket closes."""
        while True:
            try:
                self.response_sock, addr = self.listen_sock.accept()
            except OSError:
                # listen_sock was closed (shutdown() called)
                break
            self.handshake_sent = False
            self.send_handshake()
            while self.process_data():
                pass
    
    def send_data(self, data: dict[str, Any]) -> None:
        payload = self.serializer.encode(data)
        if self.response_sock:
            try:
                self.response_sock.sendall(payload)
                print(f"Sent data to {self.ip}:{self.port}")
            except Exception as e:
                raise ConnectionError(
                    f"Error sending data to {self.ip}:{self.port}: {e}, data_type: {data.get('type')}"
                )
        else:
            raise ConnectionError(
                f"Cannot send: not connected to {self.ip}:{self.port}"
            )

    def receive_data(self) -> dict[str, Any]:
        header_bytes = 4
        buffer1 = b""
        if not self.response_sock:
            raise ConnectionError(
                f"Cannot receive: no socket connected to {self.ip}:{self.port}"
            )

        while len(buffer1) < header_bytes:
            try:
                data1 = self.response_sock.recv(header_bytes - len(buffer1))
            except socket.timeout:
                raise ConnectionError(f"Timed out waiting for data from {self.ip}:{self.port}")
            if not data1:
                raise ConnectionError(f"no data received from {self.ip}:{self.port}")
            buffer1 += data1

        message_length = struct.unpack(">I", buffer1)[0]
        buffer2 = b""

        while len(buffer2) < message_length:
            try:
                data2 = self.response_sock.recv(message_length - len(buffer2))
            except socket.timeout:
                raise ConnectionError(f"Timed out waiting for data from {self.ip}:{self.port}")
            if not data2:
                raise ConnectionError(f"no data received from {self.ip}:{self.port}")
            buffer2 += data2

        return self.serializer.decode_data(buffer2)

    def send_handshake(self):
        if not self.handshake_sent:
            handshake_data = {
                "type": "handshake",
                "env_id": "test_env",
                "agents": [
                    {
                        "id": "agent_1",
                        "obs_shape": [4],
                        "act_shape": [1],
                        "is_scripted": False,
                    }
                ],
                "metadata": {
                },
            }
            handshake_data = self.serializer.encode(handshake_data)
            self.response_sock.sendall(handshake_data)
            self.handshake_sent = True

    def process_data(self):
        data = self.receive_data()
        if data.get("type") == "reset":
            self.send_reset()
            return True
        elif data.get("type") == "step":
            self.send_step(data.get("actions").get("agent_1"))
            return True
        elif data.get("type") == "close":
            self.close()
            return False
        else:
            raise ValueError(f"Unknown data type: {data.get('type')}")
    
    def send_reset(self):
        obs, info= self.env.reset()
        obs = obs.tolist()
        self.send_data({"type": "reset_result", "agents": {"agent_1": {"obs": obs, "info": info}}})
    
    def send_step(self, action: np.ndarray):
        discrete_action = 0 if action[0] < 0.5 else 1
        obs, reward, terminated, truncated, info = self.env.step(discrete_action)
        obs = obs.tolist()
        reward = float(reward)
        done = terminated or truncated
        self.send_data({"type": "step_result", "agents": {"agent_1": {"obs": obs, "reward": reward, "done": done, "info": info}}, "global": {"done": done}})

    def close(self):
        """Close the current client connection; keep the listen socket open for the next client."""
        if self.response_sock:
            self.response_sock.close()
            self.response_sock = None

    def shutdown(self):
        """Fully shut down the server (close listen socket and any active connection)."""
        self.close()
        try:
            self.listen_sock.close()
        except OSError:
            pass