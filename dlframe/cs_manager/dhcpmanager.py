import threading

from dlframe.cs_manager.Pkt import Pkt


from dlframe.cs_manager.consts import DHCP_ACK, DHCP_OFFER, DHCP_DISCOVER, DHCP_REQUEST, DHCP_NAK


class DHCPManager:
    def __init__(self, cs_manager, pool_start='192.168.1.100', pool_end='192.168.1.105'):
        self.cs_manager = cs_manager
        self.pool_start = pool_start
        self.pool_end = pool_end
        self.available_ips = self._generate_ip_pool(pool_start, pool_end)
        self.assigned_ips = {}  # client_id: ip_address
        print(f"[DHCPManager] IP Pool generated: {self.available_ips}")
        # 注册接收消息的回调（如果需要）
        # self.cs_manager.register_event_callback('on_server_recv', self.on_server_recv)

    def _generate_ip_pool(self, start, end):
        from ipaddress import IPv4Address
        start_addr = IPv4Address(start)
        end_addr = IPv4Address(end)
        start_int = int(start_addr)
        end_int = int(end_addr)
        ip_pool = [str(IPv4Address(ip)) for ip in range(start_int, end_int + 1)]
        return ip_pool

    def handle_dhcp_discover(self, pkt):
        client_id = pkt.from_addr
        print(f"[DHCPManager] Received DHCP_DISCOVER from {client_id}")
        if client_id in self.assigned_ips:
            offered_ip = self.assigned_ips[client_id]
            print(f"[DHCPManager] {client_id} already has IP {offered_ip}")
        elif self.available_ips:
            offered_ip = self.available_ips.pop(0)
            self.assigned_ips[client_id] = offered_ip
            print(f"[DHCPManager] Assigned IP {offered_ip} to {client_id}")
        else:
            # 没有可用 IP，发送 DHCP_NAK
            print(f"[DHCPManager] No available IPs. Sending DHCP_NAK to {client_id}")
            nak_pkt = Pkt(
                from_addr="server",
                to_addr=client_id,
                data=DHCP_NAK.encode('utf-8')
            )
            self.cs_manager.forward(nak_pkt)  # 使用 forward 发送
            return

        # 发送 DHCP_OFFER
        offer_data = f"{DHCP_OFFER}:{offered_ip}".encode('utf-8')
        offer_pkt = Pkt(
            from_addr="server",
            to_addr=client_id,
            data=offer_data
        )
        print(f"[DHCPManager] Sending DHCP_OFFER to {client_id} with IP {offered_ip}")
        self.cs_manager.forward(offer_pkt)  # 使用 forward 发送

    def handle_dhcp_request(self, pkt):
        client_id = pkt.from_addr
        try:
            _, requested_ip = pkt.data.decode('utf-8').split(':')
            print(f"[DHCPManager] Received DHCP_REQUEST from {client_id} for IP {requested_ip}")
        except ValueError:
            # 格式错误，发送 DHCP_NAK
            print(f"[DHCPManager] Invalid DHCP_REQUEST format from {client_id}. Sending DHCP_NAK")
            nak_pkt = Pkt(
                from_addr="server",
                to_addr=client_id,
                data=DHCP_NAK.encode('utf-8')
            )
            self.cs_manager.forward(nak_pkt)  # 使用 forward 发送
            return

        if client_id in self.assigned_ips and self.assigned_ips[client_id] == requested_ip:
            # 发送 DHCP_ACK
            print(f"[DHCPManager] DHCP_REQUEST valid. Sending DHCP_ACK to {client_id} for IP {requested_ip}")
            ack_data = f"{DHCP_ACK}:{requested_ip}".encode('utf-8')
            ack_pkt = Pkt(
                from_addr="server",
                to_addr=client_id,
                data=ack_data
            )
            self.cs_manager.forward(ack_pkt)  # 使用 forward 发送
        else:
            # 无效请求，发送 DHCP_NAK
            print(f"[DHCPManager] DHCP_REQUEST invalid for {client_id}. Sending DHCP_NAK")
            nak_pkt = Pkt(
                from_addr="server",
                to_addr=client_id,
                data=DHCP_NAK.encode('utf-8')
            )
            self.cs_manager.forward(nak_pkt)  # 使用 forward 发送

    def on_server_recv(self, data, from_addr):
        print(f"[DHCPManager] on_server_recv called with data: {data.decode('utf-8')} from {from_addr}")
        message = data.decode('utf-8')
        if message.startswith(DHCP_DISCOVER):
            pkt = Pkt(from_addr=from_addr, to_addr="server", data=data)
            threading.Thread(target=self.handle_dhcp_discover, args=(pkt,)).start()
        elif message.startswith(DHCP_REQUEST):
            pkt = Pkt(from_addr=from_addr, to_addr="server", data=data)
            threading.Thread(target=self.handle_dhcp_request, args=(pkt,)).start()
        else:
            print(f"[DHCPManager] Received non-DHCP packet from {from_addr}")

