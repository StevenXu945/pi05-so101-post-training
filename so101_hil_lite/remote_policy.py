"""Small openpi WebSocket client; no local model or PyTorch inference."""
from collections import deque
import numpy as np
from controller import vector
import codec


class RemotePolicy:
    def __init__(self, host, port, prompt, chunk_steps=8, timeout=10.0, advantage=None):
        from websockets.sync.client import connect
        self.socket = connect(f"ws://{host}:{port}", compression=None, max_size=None, open_timeout=timeout)
        self.timeout, self.prompt, self.chunk_steps = timeout, prompt, chunk_steps
        self.metadata = codec.unpackb(self.socket.recv(timeout=timeout))
        self.queue = deque()
        self.advantage = advantage

    def reset(self):
        self.queue.clear()

    def select_action(self, observation):
        if not self.queue:
            request = {
                "observation/image": np.ascontiguousarray(observation["front"]),
                "observation/wrist_image": np.ascontiguousarray(observation["wrist"]),
                "observation/state": vector(observation),
                "prompt": self.prompt,
            }
            if self.advantage is not None:
                request["adv_ind"] = self.advantage
            self.socket.send(codec.packb(request))
            response = self.socket.recv(timeout=self.timeout)
            if not isinstance(response, bytes):
                raise RuntimeError(f"Policy server error: {response}")
            actions = np.asarray(codec.unpackb(response)["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 6 or not len(actions) or not np.isfinite(actions).all():
                raise ValueError("Invalid action chunk from server")
            self.queue.extend(actions[:self.chunk_steps].copy())
        return self.queue.popleft()

    def close(self):
        self.socket.close()
