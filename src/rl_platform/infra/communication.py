import msgpack
from typing import Any
import struct
import socket
from rl_platform.core.specifications import EnvSpec
import gymnasium as gym
import numpy as np


class MsgSerializer:
    def encode(self, data: dict[str, Any]) -> bytes:
        payload = msgpack.packb(data, use_bin_type=True)
        header = struct.pack(">I", len(payload))
        return header + payload

    def decode_data(self, data: bytes) -> dict[str, Any]:
        return msgpack.unpackb(data, raw=False)


class TCPClient:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.ip, self.port))
        except Exception as e:
            self.sock.close()
            self.sock = None
            raise ConnectionError(
                f"Error initializing socket to {self.ip}:{self.port}: {e}"
            )
        self.serializer = MsgSerializer()
        self.sock.settimeout(90.0)
        print(f"Created a socket to {self.ip}:{self.port}")

    def send_data(self, data: dict[str, Any]) -> None:
        payload = self.serializer.encode(data)
        if self.sock:
            try:
                self.sock.sendall(payload)
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
        if not self.sock:
            raise ConnectionError(
                f"Cannot receive: no socket connected to {self.ip}:{self.port}"
            )

        while len(buffer1) < header_bytes:
            try:
                data1 = self.sock.recv(header_bytes - len(buffer1))
            except socket.timeout:
                raise ConnectionError(f"Timed out waiting for data from {self.ip}:{self.port}")
            if not data1:
                raise ConnectionError(f"no data received from {self.ip}:{self.port}")
            buffer1 += data1

        message_length = struct.unpack(">I", buffer1)[0]
        buffer2 = b""

        while len(buffer2) < message_length:
            try:
                data2 = self.sock.recv(message_length - len(buffer2))
            except socket.timeout:
                raise ConnectionError(f"Timed out waiting for data from {self.ip}:{self.port}")
            if not data2:
                raise ConnectionError(f"no data received from {self.ip}:{self.port}")
            buffer2 += data2

        return self.serializer.decode_data(buffer2)

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
                print(f"Disconnected socket to {self.ip}:{self.port}")
            except Exception as e:
                print(f"Error closing socket to {self.ip}:{self.port}: {e}")
        self.sock = None


class Handshake:
    def __init__(self, client: TCPClient):
        self.tcp_client = client
        self.handshake_received = False
        self.env_spec = None

    def generate_environment_config(self, data: dict[str, Any]) -> EnvSpec:
        env_id = data.get("env_id")
        agent_data = data.get("agents")
        agent_ids = []
        observation_spaces = {}
        action_spaces = {}
        if agent_data is None:
            raise ValueError(
                f"Handshake missing 'agents' field in {self.tcp_client.ip}:{self.tcp_client.port}"
            )
        if env_id is None:
            raise ValueError(
                f"Handshake missing 'env_id' field in {self.tcp_client.ip}:{self.tcp_client.port}"
            )

        for a in agent_data:
            scripted = a.get("is_scripted")
            if scripted is True:
                print(f"Skipping scripted agent {a.get('id')}")
                continue
            agent_id = a.get("id")
            observation_shape = a.get("obs_shape")
            action_shape = a.get("act_shape")
            agent_ids.append(agent_id)
            observation_spaces[agent_id] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=tuple(observation_shape)
            )
            act_low = a.get("act_low", -1.0)
            act_high = a.get("act_high", 1.0)
            action_spaces[agent_id] = gym.spaces.Box(
                low=act_low, high=act_high, shape=tuple(action_shape)
            )

        is_multi_agent = len(agent_ids) > 1
        metadata = data.get("metadata", {})

        if not agent_ids:
            raise ValueError(
                f"No agent ids found in {self.tcp_client.ip}:{self.tcp_client.port} during handshake"
            )

        return EnvSpec(
            env_id=env_id,
            agent_ids=agent_ids,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            is_multi_agent=is_multi_agent,
            metadata=metadata,
        )

    def wait_for_handshake(self) -> EnvSpec:
        print(f"Waiting for handshake from {self.tcp_client.ip}:{self.tcp_client.port}")

        while not self.handshake_received:
            data = self.tcp_client.receive_data()
            if data:
                if data.get("type") == "handshake":
                    self.env_spec = self.generate_environment_config(data)
                    if self.env_spec:
                        self.handshake_received = True
                        print(
                            f"Handshake received from {self.tcp_client.ip}:{self.tcp_client.port}: {self.env_spec}"
                        )
                        return self.env_spec
                    else:
                        raise ValueError(
                            f"Error generating environment config from {self.tcp_client.ip}:{self.tcp_client.port}: {data}"
                        )
                else:
                    raise ValueError(
                        f"Received unknown data type from {self.tcp_client.ip}:{self.tcp_client.port}: {data.get('type')}"
                    )
