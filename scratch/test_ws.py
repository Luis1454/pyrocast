import socket, threading, time, json, hashlib, base64, struct
import http.server, socketserver

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
            raw = self.connection.recv(1024)
            print(f"[*] WS client connected, raw len: {len(raw)}")

            # Stream 5 test frames
            for i in range(5):
                time.sleep(0.1)
                payload = json.dumps({"frame": i, "time_min": i * 2, "temp": 1500 + i * 20}).encode("utf-8")
                length = len(payload)
                if length <= 125:
                    header = bytes([0x81, length])
                else:
                    header = bytes([0x81, 126]) + struct.pack(">H", length)
                self.connection.sendall(header + payload)
            print("[*] Stream complete")
            return
        super().do_GET()

def run_server():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 5088), TestWSHandler)
    server.handle_request()

t = threading.Thread(target=run_server, daemon=True)
t.start()
time.sleep(0.2)

# Client test with websockets
import asyncio, websockets

async def test_client():
    async with websockets.connect("ws://127.0.0.1:5088/ws/test") as ws:
        await ws.send(json.dumps({"action": "start"}))
        for _ in range(5):
            msg = await ws.recv()
            print(f"[Client Recv]: {msg}")

asyncio.run(test_client())
print("[OK] WebSocket test passed 100%")
