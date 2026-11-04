"""One outstanding websocket inference while the SO-101 executes its current chunk."""

from queue import Empty
from queue import Queue
import threading
import time


class AsyncInferenceRunner:
    def __init__(self, client):
        self._client = client
        self._thread: threading.Thread | None = None
        self._result: Queue[tuple[bool, object]] = Queue(maxsize=1)
        self._started_at: float | None = None

    def busy(self) -> bool:
        return self._thread is not None

    def start(self, request: dict) -> None:
        if self.busy():
            raise RuntimeError("An inference request is already outstanding")
        self._started_at = time.monotonic()

        def infer() -> None:
            try:
                self._result.put((True, self._client.infer(request)))
            except BaseException as exc:
                self._result.put((False, exc))

        self._thread = threading.Thread(target=infer, name="so101-rtc-infer", daemon=True)
        self._thread.start()

    def poll(self, timeout: float) -> dict | None:
        try:
            success, result = self._result.get_nowait()
        except Empty:
            if self._started_at is not None and time.monotonic() - self._started_at >= timeout:
                self._client.close()
                raise TimeoutError(f"SO-101 RTC inference exceeded {timeout:.2f} s") from None
            return None
        self._thread = None
        self._started_at = None
        if not success:
            raise result
        return result

    def close(self) -> None:
        self._client.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
