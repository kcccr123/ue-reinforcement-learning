# socket_factory.py

import socket

def create_unreal_socket(ip, port):
    """
    Creates and returns a new TCP socket connected to the given IP/port.
    Raises an exception if the connection fails.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    print(f"[SocketFactory] Created new socket to {ip}:{port}")
    return s
