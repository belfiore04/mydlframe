# dhcpserver.py

from dlframe.cs_manager.Pkt import Pkt
from dlframe.cs_manager.consts import SERVER_ADDR_SPLITTER, DHCP_DISCOVER, DHCP_OFFER, DHCP_REQUEST, DHCP_ACK, DHCP_NAK
from ipaddress import IPv4Address

class DHCPServer:
    def __init__(self, cs_manager, pool_start='192.168.1.100', pool_end='192.168.1.200'):
        self.cs_manager = cs_manager
        self.pool_start = pool_start
        self.pool_end = pool_end
        self.available_ips = self._generate_ip_pool(pool_start, pool_end)
        self.assigned_ips = {}  # client_id: ip_address

    def _generate_ip_pool(self, start, end):
        start_addr = IPv4Address(start)
        end_addr = IPv4Address(end)
        start_int = int(start_addr)
        end_int = int(end_addr)
        # 生成 IP 地址池
        ip_pool = [str(IPv4Address(ip)) for ip in range(start_int, end_int + 1)]
        return ip_pool

    def handle_dhcp_discover(self, pkt):
        # 假设 from_addr 格式为 "server:client_id"
        if SERVER_ADDR_SPLITTER in pkt.from_addr:
            _, client_id = pkt.from_addr.split(SERVER_ADDR_SPLITTER, 1)
        else:
            client_id = pkt.from_addr

        if client_id in self.assigned_ips:
            offered_ip = self.assigned_ips[client_id]
        elif self.available_ips:
            offered_ip = self.available_ips.pop(0)
            self.assigned_ips[client_id] = offered_ip
        else:
            # 没有可用 IP，发送 DHCP_NAK
            nak_pkt = Pkt(
                from_addr=self.cs_manager.addr,
                to_addr=pkt.from_addr,
                data=DHCP_NAK.encode('utf-8')
            )
            self.cs_manager.send(nak_pkt)
            return

        # 发送 DHCP_OFFER
        offer_data = f"{DHCP_OFFER}:{offered_ip}".encode('utf-8')
        offer_pkt = Pkt(
            from_addr=self.cs_manager.addr,
            to_addr=pkt.from_addr,
            data=offer_data
        )
        self.cs_manager.send(offer_pkt)

    def handle_dhcp_request(self, pkt):
        # 假设 from_addr 格式为 "server:client_id"
        if SERVER_ADDR_SPLITTER in pkt.from_addr:
            _, client_id = pkt.from_addr.split(SERVER_ADDR_SPLITTER, 1)
        else:
            client_id = pkt.from_addr

        try:
            _, requested_ip = pkt.data.decode('utf-8').split(':')
        except ValueError:
            # 格式错误，发送 DHCP_NAK
            nak_pkt = Pkt(
                from_addr=self.cs_manager.addr,
                to_addr=pkt.from_addr,
                data=DHCP_NAK.encode('utf-8')
            )
            self.cs_manager.send(nak_pkt)
            return

        if client_id in self.assigned_ips and self.assigned_ips[client_id] == requested_ip:
            # 发送 DHCP_ACK
            ack_data = f"{DHCP_ACK}:{requested_ip}".encode('utf-8')
            ack_pkt = Pkt(
                from_addr=self.cs_manager.addr,
                to_addr=pkt.from_addr,
                data=ack_data
            )
            self.cs_manager.send(ack_pkt)
        else:
            # 无效请求，发送 DHCP_NAK
            nak_pkt = Pkt(
                from_addr=self.cs_manager.addr,
                to_addr=pkt.from_addr,
                data=DHCP_NAK.encode('utf-8')
            )
            self.cs_manager.send(nak_pkt)

    def on_server_recv(self, websocket, pkt, path, send_queue):
        if pkt.data.startswith(DHCP_DISCOVER.encode('utf-8')):
            self.handle_dhcp_discover(pkt)
        elif pkt.data.startswith(DHCP_REQUEST.encode('utf-8')):
            self.handle_dhcp_request(pkt)

    def start(self):
        self.cs_manager.register_event_callback('on_server_recv', self.on_server_recv)
