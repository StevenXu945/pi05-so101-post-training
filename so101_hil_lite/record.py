"""Official LeRobot SO-101 recorder with keyboard human intervention."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import select
import sys
import termios
import time
import tty
import numpy as np

from controller import AlignmentError, Controller, action_dict, vector
from recording import Recorder
from remote_policy import RemotePolicy


@contextmanager
def keyboard():
    if not sys.stdin.isatty():
        raise RuntimeError("Run in an interactive terminal for takeover keys")
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        yield lambda: sys.stdin.read(1).lower() if select.select([sys.stdin], [], [], 0)[0] else ""
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def align_arms(follower, leader, control, fps):
    leader.disable_torque()
    input("ENTER enables follower tracking of leader (follower will move). Ctrl+C exits: ")
    print("Drag leader to set the start pose; ENTER switches to policy; q exits. Alignment is not recorded.", flush=True)
    previous = vector(follower.get_observation())
    with keyboard() as read_key:
        while True:
            cycle = time.monotonic()
            key = read_key()
            if key == "q":
                raise KeyboardInterrupt
            observation = follower.get_observation()
            if key in {"\n", "\r"}:
                try:
                    control.start(observation)
                    return
                except AlignmentError as error:
                    print(error, flush=True)
                    print("Continue dragging/wait for follower, then ENTER to retry.", flush=True)
            # Limit both target increments and distance from measured follower pose.
            current = vector(observation)
            desired = vector(leader.get_action())
            lower = np.maximum(previous - control.limits, current - control.limits)
            upper = np.minimum(previous + control.limits, current + control.limits)
            if np.any(lower > upper):
                raise ValueError("Follower position deviated too far during alignment")
            target = np.clip(desired, lower, upper)
            previous = vector(follower.send_action(action_dict(target)))
            time.sleep(max(0, 1 / fps - (time.monotonic() - cycle)))


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("follower-port", "leader-port", "follower-id", "leader-id", "front-camera", "wrist-camera", "policy-id"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="Stack the red cube on top of the blue cube")
    parser.add_argument("--root", type=Path, default=Path("data/stack_v1"))
    parser.add_argument("--repo-id", default="local/so101_stack_v1")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--chunk-steps", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--max-arm-step", type=float, default=6.51)
    parser.add_argument("--max-gripper-step", type=float, default=20)
    parser.add_argument("--advantage", choices=["positive", "negative"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Read observations and infer without sending joint targets")
    result = parser.parse_args()
    if min(result.episodes, result.fps, result.duration, result.chunk_steps, result.timeout,
           result.max_arm_step, result.max_gripper_step) <= 0:
        parser.error("Counts, timing and step limits must be positive")
    if result.front_camera == result.wrist_camera:
        parser.error("Front and wrist cameras must differ")
    return result


def run(cfg):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    def camera(value):
        source = int(value) if value.isdigit() else Path(value)
        return OpenCVCameraConfig(index_or_path=source, fps=cfg.fps, width=640, height=480, fourcc="MJPG")

    if not cfg.dry_run and cfg.root.exists():
        raise FileExistsError(f"Use a new dataset root: {cfg.root}")
    follower = SO101Follower(SO101FollowerConfig(
        port=cfg.follower_port, id=cfg.follower_id, use_degrees=True,
        cameras={"front": camera(cfg.front_camera), "wrist": camera(cfg.wrist_camera)}))
    leader = SO101Leader(SO101LeaderConfig(port=cfg.leader_port, id=cfg.leader_id, use_degrees=True))
    policy = recorder = control = None
    try:
        follower.connect(calibrate=False)
        leader.connect(calibrate=False)
        if not follower.is_calibrated or not leader.is_calibrated:
            raise RuntimeError("Calibrate both arms using official LeRobot first")
        for method in ("send_feedback", "enable_torque", "disable_torque"):
            if not callable(getattr(leader, method, None)):
                raise RuntimeError(f"Pinned official leader API missing {method}")
        policy = RemotePolicy(cfg.host, cfg.port, cfg.prompt, cfg.chunk_steps, cfg.timeout, cfg.advantage)
        observation = follower.get_observation()
        first = policy.select_action(observation)
        print("First policy target:", first, flush=True)
        policy.reset()
        if cfg.dry_run:
            return
        recorder = Recorder(cfg.root, cfg.repo_id, cfg.fps, observation, cfg.policy_id)
        control = Controller(follower, leader, policy, cfg.max_arm_step, cfg.max_gripper_step)
        for _ in range(cfg.episodes):
            align_arms(follower, leader, control, cfg.fps)
            started = time.monotonic()
            paused_seconds = 0.0
            outcome = None
            print("i: pause for takeover; ENTER: release leader after gripping; i in MANUAL: resume policy; s/f: outcome; q: exit", flush=True)
            with keyboard() as read_key:
                while outcome is None:
                    cycle = time.monotonic()
                    key = read_key()
                    if key in {"s", "f", "q"}:
                        outcome = {"s": "success", "f": "failure", "q": "aborted"}[key]
                        break
                    if cycle - started - paused_seconds >= cfg.duration and not control.pending_takeover:
                        break
                    observation = follower.get_observation()
                    if key == "i":
                        try:
                            control.toggle(observation)
                        except AlignmentError as error:
                            print(error, flush=True)
                            print("Still MANUAL; align joints, then press i to retry release.", flush=True)
                        if control.pending_takeover:
                            print("PAUSED: both arms hold. Grip/support leader, then ENTER to release torque.", flush=True)
                        else:
                            print("MANUAL" if control.manual else "POLICY", flush=True)
                    elif key in {"\n", "\r"} and control.pending_takeover:
                        control.confirm_takeover(observation)
                        print("MANUAL: leader torque released; keep supporting it. i resumes policy.", flush=True)
                    paused_cycle = control.pending_takeover
                    selected, accepted, proposed, manual = control.step(observation)
                    recorder.add(observation, selected, accepted, proposed, manual,
                                 cycle - started, cfg.prompt, paused=control.pending_takeover)
                    time.sleep(max(0, 1 / cfg.fps - (time.monotonic() - cycle)))
                    if paused_cycle:
                        paused_seconds += time.monotonic() - cycle
            control.stop()
            if outcome is None:
                while outcome not in {"success", "failure", "aborted"}:
                    outcome = input("Time limit. Enter success / failure / aborted: ").strip().lower()
            recorder.finish(outcome)
            if outcome == "aborted":
                break
    finally:
        # Preserve partial episodes as aborted; always attempt all device cleanup.
        try:
            if control is not None and leader.is_connected:
                control.stop()
        finally:
            try:
                if recorder is not None:
                    try:
                        recorder.finish("aborted")
                    finally:
                        recorder.close()
            finally:
                try:
                    if leader.is_connected:
                        leader.disconnect()
                finally:
                    try:
                        if follower.is_connected:
                            follower.disconnect()
                    finally:
                        if policy is not None:
                            policy.close()


if __name__ == "__main__":
    run(args())
