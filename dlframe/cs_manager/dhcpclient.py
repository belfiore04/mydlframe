import threading

from dlframe.cs_manager.Pkt import Pkt

from dlframe.cs_manager.consts import DHCP_OFFER, DHCP_ACK, DHCP_NAK, DHCP_REQUEST, DHCP_DISCOVER


class DHCPClient:
    def __init__(self, cs_manager, client_id, server_addr):
        self.cs_manager = cs_manager
        self.client_id = client_id
        self.server_addr = server_addr
        self.assigned_ip = None
        self.module_sender = self.cs_manager.register_fn(client_id, self.on_recv_fn)
        print(f"[DHCPClient] Initializing client {self.client_id}")
        self.send_dhcp_discover()

    def send_dhcp_discover(self):
        discover_data = DHCP_DISCOVER.encode('utf-8')
        print(f"[DHCPClient {self.client_id}] Sending DHCP_DISCOVER")
        discover_pkt = Pkt(
            from_addr=self.client_id,
            to_addr=self.server_addr,
            data=discover_data
        )
        # 通过独立线程发送，防止阻塞
        threading.Thread(target=self.module_sender.send, args=(discover_pkt,)).start()

    def on_recv_fn(self, data, from_addr):
        print(f"[DHCPClient {self.client_id}] on_recv_fn called with data from {from_addr}")
        try:
            message = data.decode('utf-8')
            if message.startswith(DHCP_OFFER):
                _, offered_ip = message.split(':')
                print(f"[DHCPClient {self.client_id}] Received DHCP_OFFER with IP {offered_ip}")
                self.send_dhcp_request(offered_ip)
            elif message.startswith(DHCP_ACK):
                _, ack_ip = message.split(':')
                self.assigned_ip = ack_ip
                print(f"[DHCPClient {self.client_id}] Received DHCP_ACK. Assigned IP: {self.assigned_ip}")
            elif message.startswith(DHCP_NAK):
                print(f"[DHCPClient {self.client_id}] Received DHCP_NAK. No IP assigned.")
            print(f"[DHCPClient {self.client_id}] on_recv_fn finished")
        except Exception as e:
            print(f"[DHCPClient {self.client_id}] Exception in on_recv_fn: {e}")

    def send_dhcp_request(self, requested_ip):
        request_data = f"{DHCP_REQUEST}:{requested_ip}".encode('utf-8')
        print(f"[DHCPClient {self.client_id}] Sending DHCP_REQUEST for IP {requested_ip}")
        request_pkt = Pkt(
            from_addr=self.client_id,
            to_addr=self.server_addr,
            data=request_data
        )
        # 通过独立线程发送，防止阻塞
        threading.Thread(target=self.module_sender.forward, args=(request_pkt,)).start()
