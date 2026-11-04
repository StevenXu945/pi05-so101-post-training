"""Record cached RGB camera frames without touching the robot control bus."""

from fractions import Fraction
import json
import logging
from pathlib import Path
import threading
import time

import numpy as np

LOG = logging.getLogger(__name__)


class CameraVideoRecorder:
    def __init__(self, cameras: dict, directory: Path, fps: float):
        self.cameras = cameras
        self.directory = Path(directory)
        self.fps = fps
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._error = None
        self.frames = dict.fromkeys(cameras, 0)
        self.elapsed_s = 0.0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Video recording has already started")
        self.directory.mkdir(parents=True, exist_ok=False)
        self._thread = threading.Thread(target=self._record, name="so101-video", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            self._stop.set()
            raise TimeoutError("Video recorder did not initialize within 10 seconds")
        if self._error is not None:
            raise RuntimeError("Could not initialize video recording") from self._error
        LOG.info("Recording front/wrist videos to %s", self.directory)

    def _record(self) -> None:
        outputs = {}
        started = time.monotonic()
        try:
            import av

            for name, camera in self.cameras.items():
                image = camera.read_latest()
                height, width, channels = image.shape
                if channels != 3 or image.dtype != np.uint8 or height % 2 or width % 2:
                    raise ValueError(f"Expected an even-sized uint8 RGB image from {name}")
                container = av.open(str(self.directory / f"{name}.mp4"), mode="w")
                outputs[name] = (container, None)
                stream = container.add_stream("libx264", rate=Fraction(str(self.fps)))
                outputs[name] = (container, stream)
                stream.width, stream.height = width, height
                stream.pix_fmt = "yuv420p"
                stream.time_base = Fraction(1, 1000)
                stream.codec_context.thread_count = 1
                stream.options = {"preset": "ultrafast", "crf": "23", "tune": "zerolatency"}
            started = deadline = time.monotonic()
            self._ready.set()
            previous_pts = -1
            while not self._stop.is_set():
                elapsed = time.monotonic() - started
                pts = max(previous_pts + 1, round(elapsed * 1000))
                for name, camera in self.cameras.items():
                    image = np.array(camera.read_latest(), copy=True)
                    container, stream = outputs[name]
                    if image.shape != (stream.height, stream.width, 3) or image.dtype != np.uint8:
                        raise ValueError(f"Camera {name} changed image shape or dtype")
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    frame.pts, frame.time_base = pts, Fraction(1, 1000)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    self.frames[name] += 1
                previous_pts = pts
                # Skip missed deadlines rather than producing a burst of duplicate frames.
                deadline += 1.0 / self.fps
                now = time.monotonic()
                if deadline < now:
                    deadline = now + 1.0 / self.fps
                self._stop.wait(max(0.0, deadline - now))
        except Exception as exc:
            self._error = exc
            LOG.exception("Camera video recording failed")
        finally:
            self.elapsed_s = time.monotonic() - started
            for container, stream in outputs.values():
                try:
                    if stream is not None:
                        for packet in stream.encode():
                            container.mux(packet)
                except Exception as exc:
                    self._error = self._error or exc
                finally:
                    try:
                        container.close()
                    except Exception as exc:
                        self._error = self._error or exc
            self._ready.set()

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise TimeoutError("Video recorder did not stop within 10 seconds")
        metadata = {
            "requested_fps": self.fps,
            "elapsed_s": self.elapsed_s,
            "frames": self.frames,
            "error": str(self._error) if self._error is not None else None,
        }
        (self.directory / "recording.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if self._error is not None:
            raise RuntimeError(f"Video recording failed; see {self.directory}/recording.json") from self._error
        LOG.info("Saved camera videos: %s", metadata)
