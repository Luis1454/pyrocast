import asyncio
import json
import websockets


async def main():
    payload = {
        "lat": 43.5250,
        "lon": 5.4420,
        "wind_speed_kmh": 45.0,
        "wind_dir": "NW",
        "fmc": 6.0,
        "spotting_enabled": False,
        "firebreak": False,
        "retardant": False,
        "num_ensembles": 2,
        "precipitation_mm_h": 0.0,
    }
    frames = []
    async with websockets.connect("ws://127.0.0.1:5050/ws/simulate_3d", max_size=32 * 1024 * 1024) as ws:
        await ws.send(json.dumps(payload))
        while True:
            message = json.loads(await ws.recv())
            if message["type"] == "started":
                assert message["grid_size"] == 128
                assert message["resolution_m"] == 50.0
                assert message["domain_size_m"] == 6400.0
                assert message["frames_total"] == 30
            elif message["type"] == "frame":
                frames.append(message["frame"])
            elif message["type"] == "complete":
                assert message["frames_total"] == 30
                break
            elif message["type"] == "error":
                raise RuntimeError(message["message"])
    assert len(frames) == 30, len(frames)
    assert [frame["time_min"] for frame in frames[:3]] == [0, 2, 4]
    print(f"[OK] WebSocket simulation: {len(frames)} frames, {len(frames[-1]['prob_map_flat'])} cells")


asyncio.run(main())
