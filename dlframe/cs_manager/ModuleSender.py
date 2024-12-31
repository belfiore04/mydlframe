# modulesender.py

from dlframe.cs_manager.Pkt import Pkt

class ModuleSender:
    def __init__(self, cs_manager, fn_addr, on_recv_fn) -> None:
        self.cs_manager = cs_manager
        self.fn_addr = fn_addr
        self.on_recv_fn = on_recv_fn

    def send(self, pkt: Pkt):
        # 直接发送 Pkt 对象
        self.cs_manager.send(pkt)
