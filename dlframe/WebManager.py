import queue
import socket
import threading
import asyncio
import json
import websockets
import traceback
import os
import time
from dlframe.CalculationNodeManager import CalculationNodeManager
from dlframe.Logger import Logger


class SendSocket:
    def __init__(self, socket) -> None:
        self.sendBuffer = queue.Queue()
        self.socket = socket
        self.sendThread = threading.Thread(target=self.threadWorker, daemon=True)
        self.sendThread.start()

    def threadWorker(self):
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        event_loop.run_until_complete(self.sendWorker())

    async def sendWorker(self):
        while True:
            content = self.sendBuffer.get()
            await self.socket.send(content)

    def send(self, content: str):
        self.sendBuffer.put(content)

class WebManager(CalculationNodeManager):
    def __init__(self, host='0.0.0.0', port=8765, parallel=False) -> None:
        super().__init__(parallel=parallel)
        self.host = host
        self.port = port
        self.logger = Logger.get_logger('WebManager')
        self.updated_content=""
        self.last_mtime = 0
        self.last_content = []

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise exc_val
        self.start(self.host, self.port)
    async def connect_to_server(self):
        server_host ="192.168.224.46"
        server_port=65432
        server_conn = await self.connect_to_remote(server_host, server_port)
        conf = server_conn.recv(4096).decode()
        data = json.loads(conf)
        return (data,server_conn)
    async def execute_training(self, conf):

        # 这里放置实际的训练代码，可以使用 display.py 中的数据集、模型等
        logger = Logger.get_logger('TrainingWorker')
        logger.print(f"Starting training with config: {conf}")
        # ... 根据 conf 中的信息加载数据集、模型 ...
        # ... 执行训练 ...
        self.execute(conf)
        logger.print("Training completed.")
        return {'status': 200, 'data': 'Training completed successfully'}

    async def connect_to_remote(self, host, port):
        try:
                websocket= socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                websocket.bind((host, port))
                websocket.listen()
                websocket,addr = websocket.accept()
                return websocket
        except ConnectionRefusedError:
            print(f"Failed to connect to worker at {host}:{port}. Connection refused.")
            return None
        except Exception as e:
            print(f"An error occurred while connecting to worker at {host}:{port}: {e}")
            return None

    async def send_train_command(self, worker_conn, command):
        try:
            worker_conn.sendall(json.dumps(command).encode())
            # 存储接收到的所有数据
            all_response = b'' # 使用 bytes 类型存储，避免编码问题
            while True:
                response = worker_conn.recv(8192)  # 直接接收 bytes 数据
                all_response += response
                if '$' in response.decode():
                    break
            return all_response.decode() 
        except Exception as e:
            print(f"Error sending command to worker: {e}")
            return None

    async def handle_train_request(self, params):
        # 假设 params 中包含了要执行的训练配置和目标计算节点的地址
        worker_host = "0.0.0.0"
        worker_port = 65432
        train_config = params

        if not worker_host or not worker_port or not train_config:
            return {'status': 500, 'data': 'Missing worker information or train configuration'}

        worker_conn = await self.connect_to_remote(worker_host, worker_port)
        with worker_conn:
            if worker_conn:
                command = {'type': 'train', 'params': train_config}
                response = await self.send_train_command(worker_conn, command)
                return response
            else:
                return {'status': 500, 'data': f"Failed to connect to worker at {worker_host}:{worker_port}"}
    def detect_file_changes(self,file_path,conn):
        """
        检测文件是否有变化，并返回变化的部分

        参数:
        file_path (str): 要检测的文件路径

        返回:
        str: 文件变化的部分，如果没有变化则返回空字符串
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        while True:
            # 获取当前文件的修改时间
            current_mtime = os.path.getmtime(file_path)
            if current_mtime > self.last_mtime:
                with open(file_path, 'r') as f:
                    current_content = f.readlines()
                    if self.last_content:
                        added_lines = []
                        min_lines = min(len(self.last_content), len(current_content))
                        for i in range(min_lines):
                            if current_content[i]!= self.last_content[i]:
                                added_lines.extend(current_content[i:])
                                break
                        else:
                            added_lines = current_content[min_lines:]
                        self.last_content = current_content
                        self.last_mtime = current_mtime
                        self.updated_content = "".join(added_lines).strip()
                    else:
                        self.last_content = current_content
                        self.last_mtime = current_mtime
                        self.updated_content ="".join(current_content).strip()
                conn.sendall((self.updated_content+"$").encode())
            time.sleep(0.1)

    def start(self, host=None, port=None) -> None:
        if host is None:
            host = self.host
        if port is None:
            port = self.port
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        async def onRecv(socket, path):
            msgIdx = -1
            sendSocket = SendSocket(socket)
            async for message in socket:
                msgIdx += 1
                message = json.loads(message)
                params_local = message['params']
                conf = params_local
                # print(msgIdx, message)
                def trigger(x):
                    data_dict = {
                        'status': 200,
                        'type': x['type'],
                        'data': {
                            'content': '[{}]: '.format(x['name']) + ' '.join(
                                [str(_) for _ in x['args']]) + getattr(x['kwargs'], 'end', '\n') if x[
                                                                                                        'type'] == 'print' \
                                else image2base64(x['args'])}
                    }
                    sendSocket.send(json.dumps(data_dict))
                    if x['type'] == 'print':
                        with(open('log.txt', 'a')) as f:
                            if params_local.get('训练位置') == '远程-计算':
                                f.write(data_dict['data']['content'].strip() + '\n')   
                Logger.global_trigger = trigger
                for logger in Logger.loggers.values():
                    if logger.trigger is None:
                        logger.trigger = Logger.global_trigger
                # key error
                if not all([key in message.keys() for key in ['type', 'params']]):
                    response = json.dumps({
                        'status': 500, 
                        'data': 'no key param'
                    })
                    await socket.send(response)
                
                # key correct
                else:
                    if message['type'] == 'overview':
                        response = json.dumps({
                            'status': 200, 
                            'type': 'overview', 
                            'data': self.inspect()
                        })
                        await socket.send(response)

                    elif message['type'] == 'run':
                        params = message['params']
                        conf = params
                        if params.get('训练位置') == '远程-控制':
                            remote_response = await self.handle_train_request(params)
                            response = json.dumps({
                            'status': 200,
                            'type': 'print',
                            'data': {
                                'content': remote_response
                                    }
                        })
                            await socket.send(response)
                        elif params.get('训练位置') == '远程-计算':
                            try:
                                conf,conn = await self.connect_to_server()
                                conf = conf['params']
                                self.execute(conf)
                                threading.Thread(target=self.detect_file_changes, args=("log.txt",conn)).start()
                                
                                #await socket.send(response)
                            except Exception as e:
                                error_msg = traceback.format_exc()
                                response = json.dumps(
                                    {'status': 500, 'data': f'Training error: {e}', 'traceback': error_msg})
                                await socket.send(response)
                        else:
                            def image2base64(img):
                                import base64
                                from io import BytesIO
                                from PIL import Image

                                # 创建一个示例NumPy数组（图像）
                                image_np = img

                                # 将NumPy数组转换为PIL.Image对象
                                image_pil = Image.fromarray(image_np)

                                # 将PIL.Image对象保存为字节流
                                buffer = BytesIO()
                                image_pil.save(buffer, format='JPEG')
                                buffer.seek(0)

                                # 使用base64库将字节流编码为base64字符串
                                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

                                return image_base64

                            # def trigger(x):
                            #     data_dict = {
                            #         'status': 200,
                            #         'type': x['type'],
                            #         'data': {
                            #             'content': '[{}]: '.format(x['name']) + ' '.join(
                            #                 [str(_) for _ in x['args']]) + getattr(x['kwargs'], 'end', '\n') if x[
                            #                                                                                         'type'] == 'print' \
                            #                 else image2base64(x['args'])}
                            #     }
                            #     sendSocket.send(json.dumps(data_dict))
                            #     if x['type'] == 'print':
                            #         with(open('log.txt', 'a')) as f:
                            #             if conf.get('训练位置') == '远程-计算':
                            #                 f.write(data_dict['data']['content'].strip() + '\n')

                            # Logger.global_trigger = trigger
                            # for logger in Logger.loggers.values():
                            #     if logger.trigger is None:
                            #         logger.trigger = Logger.global_trigger

                            try:
                                self.execute(conf)
                            except Exception as e:
                                response = json.dumps({
                                    'status': 200,
                                    'type': 'print',
                                    'data': {
                                        'content': traceback.format_exc()
                                    }
                                })
                                await socket.send(response)

                    # unknown key
                    else:
                        response = json.dumps({
                            'status': 500, 
                            'data': 'unknown type'
                        })
                        await socket.send(response)



        print('The backend server is running on [{}:{}]...'.format(host, port))
        print('The frontend page is at: https://picpic2013.github.io/dlframe-front/')

        event_loop.run_until_complete(websockets.serve(onRecv, host, port))
        event_loop.run_forever()