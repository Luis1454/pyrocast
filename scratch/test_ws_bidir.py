import socket, threading, time, json, hashlib, base64, struct
import http.server, socketserver
import asyncio, websockets

def send_ws_json(sock, obj):
    payload = json.dumps(obj).encode("utf-8")
    length = len(payload)
    if length <= 125:
        header = bytes([0x81, length])
    elif length <= 65535:
        header = bytes([0x81, 126]) + struct.pack(">H", length)
    else:
        header = bytes([0x81, 127]) + struct.pack(">Q", length)
    sock.sendall(header + payload)

def read_ws_msg(sock):
    head = sock.recv(2)
    if not head or len(head) < 2: return None
    fin_opcode, mask_len = head[0], head[1]
    is_masked = bool(mask_len & 0x80)
    length = mask_len & 0x7F
    if length == 126:
        length = struct.unpack(">H", sock.recv(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", sock.recv(8))[0]
    mask = sock.recv(4) if is_masked else None
    data = bytearray()
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk: break
        data.extend(chunk)
    if is_masked and mask:
        for i in range(len(data)):
            data[i] ^= mask[i % 4]
    return data.decode("utf-8", errors="ignore")

class TestWSHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_GET(self):
        if self.path == "/ws/test":
            key = self.headers.get("Sec-WebSocket-Key")
            if not key:
                self.send_error(400)
                return
            magic = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
            accept_val = base64.b64encode(hashlib.sha1(key.encode("utf-8") + magic).digest()).decode("utf-8")
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept_val)
            self.end_headers()

            # Read client start message
            msg = read_ws_msg(self.connection)
            print(f"[*] WS Server received: {msg}")

            # Stream 5 test frames
            for i in range(5):
                time.sleep(0.05)
                send_ws_json(self.connection, {"type": "frame", "frame_idx": i, "time_min": i * 2, "temp": 1500 + i * 20})
            send_ws_json(self.connection, {"type": "complete", "status": "ok"})
            return
        super().do_GET()

def run_server():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 5089), TestWSHandler)
    server.handle_request()

t = threading.Thread(target=run_server, daemon=True)
t.start()
time.sleep(0.2)

async def test_client():
    async with websockets.connect("ws://127.0.0.1:5089/ws/test") as ws:
        await ws.send(json.dumps({"action": "start_simulation"}))
        for _ in range(6):
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"[Client Recv]: {data}")

asyncio.run(test_client())
print("[OK] WebSocket bidirectional communication verified 100%")
