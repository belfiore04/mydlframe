# server.py

import socket


def start_server(host='0.0.0.0', port=65432):
    """
    启动服务器，监听指定的主机和端口。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"服务器已启动，监听 {host}:{port} ...")

        conn, addr = s.accept()
        with conn:
            print(f"已连接来自 {addr} 的客户端。")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print(f"收到来自客户端的消息: {data.decode()}")
                response = "消息已收到"
                conn.sendall(response.encode())


if __name__ == "__main__":
    start_server()
