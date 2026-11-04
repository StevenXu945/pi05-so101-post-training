"""Run the served SO-101 policy with synchronous chunks or JAX guided RTC.

This script deliberately does not clamp targets to calibration or user joint limits.
It checks the whole action chunk for excessive adjacent steps, then sends the
selected actions open-loop, or overlaps inference and execution with --rtc.
Use only after verifying the arm's physical range.
The policy checkpoint configured in this repository has a 16-step horizon.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from openpi_client.rtc_runner import AsyncInferenceRunner

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ACTION_HORIZON = 16
STEP_ROUNDING_TOLERANCE = 1e-4  # Joint targets are float32; subtraction can exceed the limit by a few ulps.
LOG = logging.getLogger("eval_so101_pi0_style")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-arm", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--front-camera", type=int, required=True)
    parser.add_argument("--wrist-camera", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--prompt", default="Grab the toy into the box")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--chunk-exec", type=int, default=8, help="Actions per inference in non-RTC mode")
    parser.add_argument("--rtc", action="store_true", help="Enable JAX guided RTC")
    parser.add_argument("--rtc-inference-delay", type=int, default=6)
    parser.add_argument("--rtc-execution-horizon", type=int, default=8)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--rtc-timeout", type=float, default=2.0)
    parser.add_argument("--rtc-warmup-timeout", type=float, default=120.0)
    parser.add_argument("--max-arm-step", type=float, default=6.51)
    parser.add_argument("--max-gripper-step", type=float, default=20.0)
    parser.add_argument("--hold-on-exit", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true", help="Enable motor commands after first-action review")
    args = parser.parse_args()
    values = (args.duration, args.fps, args.max_arm_step, args.max_gripper_step, args.hold_on_exit)
    if not all(np.isfinite(values)) or any(value <= 0 for value in values[:4]) or args.hold_on_exit < 0:
        parser.error("duration, fps and step limits must be positive and finite; hold-on-exit must be nonnegative")
    if not 1 <= args.chunk_exec <= ACTION_HORIZON:
        parser.error(f"--chunk-exec must be between 1 and {ACTION_HORIZON}")
    if args.front_camera == args.wrist_camera:
        parser.error("front and wrist cameras must use different indices")
    if args.rtc:
        if not 0 <= args.rtc_inference_delay <= args.rtc_execution_horizon < ACTION_HORIZON:
            parser.error("RTC requires 0 <= inference delay <= execution horizon < 16")
        if args.rtc_execution_horizon - args.rtc_inference_delay < 2:
            LOG.warning("RTC guided prefix has fewer than two fade steps; measured delay may reach its unguided edge")
        if args.rtc_inference_delay > ACTION_HORIZON - args.rtc_execution_horizon:
            parser.error("RTC inference delay exceeds the actions remaining when inference starts")
        if not np.isfinite(args.rtc_max_guidance_weight) or args.rtc_max_guidance_weight <= 0:
            parser.error("RTC guidance weight must be positive and finite")
        if not np.isfinite(args.rtc_timeout) or args.rtc_timeout <= 0:
            parser.error("RTC timeout must be positive and finite")
        if not np.isfinite(args.rtc_warmup_timeout) or args.rtc_warmup_timeout <= 0:
            parser.error("RTC warmup timeout must be positive and finite")
    return args


def make_observation(robot, prompt: str) -> tuple[dict, np.ndarray]:
    raw = robot.get_observation()
    state = np.asarray([raw[f"{joint}.pos"] for joint in JOINTS], dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Robot returned non-finite joint positions")
    images = {}
    for camera, key in (("front", "observation/image"), ("wrist", "observation/wrist_image")):
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(raw[camera], 224, 224))
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError(f"Unexpected {camera} image: shape={image.shape}, dtype={image.dtype}")
        images[key] = image
    return {**images, "observation/state": state, "prompt": prompt}, state


def check_chunk(actions: np.ndarray, state: np.ndarray, arm_step: float, gripper_step: float) -> None:
    reference = state
    for step, target in enumerate(actions):
        difference = np.abs(target - reference)
        arm_limit = np.inf if step == 0 else arm_step
        exceeded = np.flatnonzero(difference[:5] > arm_limit + STEP_ROUNDING_TOLERANCE).tolist()
        if difference[5] > gripper_step + STEP_ROUNDING_TOLERANCE:
            exceeded.append(5)
        if exceeded:
            details = ", ".join(f"{JOINTS[joint]}={difference[joint]:.2f}" for joint in exceeded)
            raise ValueError(f"Action chunk step {step} exceeds per-step limit: {details}")
        reference = target


def safe_shutdown(robot, hold_s: float) -> None:
    try:
        if not robot.is_connected:
            return
        raw = robot.get_observation()
        current = {f"{joint}.pos": float(raw[f"{joint}.pos"]) for joint in JOINTS}
        robot.send_action(current)
        if hold_s:
            LOG.info("Holding current pose for %.1f s before disconnect", hold_s)
            time.sleep(hold_s)
    except Exception:
        LOG.exception("Could not hold current pose during shutdown")
    finally:
        if robot.is_connected:
            robot.disconnect()
            LOG.info("Follower disconnected; support the arm if torque was released")


def read_actions(result: dict) -> np.ndarray:
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, len(JOINTS)) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite ({ACTION_HORIZON}, {len(JOINTS)}) actions, got {actions.shape}")
    return actions


def send_target(robot, target: np.ndarray) -> None:
    robot.send_action({f"{joint}.pos": float(value) for joint, value in zip(JOINTS, target, strict=True)})


def check_transition(target: np.ndarray, previous: np.ndarray, arm_step: float, gripper_step: float) -> None:
    difference = np.abs(target - previous)
    exceeded = np.flatnonzero(difference[:5] > arm_step + STEP_ROUNDING_TOLERANCE).tolist()
    if difference[5] > gripper_step + STEP_ROUNDING_TOLERANCE:
        exceeded.append(5)
    if exceeded:
        details = ", ".join(f"{JOINTS[joint]}={difference[joint]:.2f}" for joint in exceeded)
        raise ValueError(f"RTC chunk transition exceeds per-step limit: {details}")


def limit_rtc_chunk(actions: np.ndarray, start: int, previous: np.ndarray,
                    arm_step: float, gripper_step: float) -> tuple[np.ndarray, int]:
    """Rate-limit a guided chunk from the last command actually sent to the arm."""
    limited = np.array(actions, copy=True)
    reference = np.array(previous, copy=True)
    limits = np.array([arm_step] * 5 + [gripper_step], dtype=np.float32)
    changed = 0
    for index in range(start, len(limited)):
        target = limited[index]
        safe_target = reference + np.clip(target - reference, -limits, limits)
        if np.any(safe_target != target):
            changed += 1
        limited[index] = safe_target
        reference = safe_target
    return limited, changed


def rtc_request(observation: dict, leftover: np.ndarray, args: argparse.Namespace) -> dict:
    return {**observation, "_rtc": {
        "enabled": True,
        "prev_chunk_left_over": np.array(leftover, copy=True),
        "inference_delay": args.rtc_inference_delay,
        "execution_horizon": args.rtc_execution_horizon,
        "max_guidance_weight": args.rtc_max_guidance_weight,
    }}


def warmup_rtc(client, observation: dict, actions: np.ndarray, args: argparse.Namespace) -> None:
    """Compile the guided JAX path before enabling motor commands."""
    request = rtc_request(observation, actions[args.rtc_execution_horizon:], args)
    runner = AsyncInferenceRunner(client)
    started = time.monotonic()
    LOG.info("Warming up JAX RTC guidance before motor execution")
    runner.start(request)
    try:
        while (result := runner.poll(args.rtc_warmup_timeout)) is None:
            time.sleep(0.05)
        read_actions(result)
    except BaseException:
        runner.close()
        raise
    LOG.info("RTC guidance warmup finished in %.1f s", time.monotonic() - started)


def run_rtc(args: argparse.Namespace, robot, client) -> None:
    observation, state = make_observation(robot, args.prompt)
    actions = read_actions(client.infer(observation))
    warmup_rtc(client, observation, actions, args)
    LOG.info("RTC first chunk: state=%s first=%s", np.round(state, 2), np.round(actions[0], 2))
    if not args.execute:
        LOG.info("Dry run complete; add --execute to send motor commands")
        return

    print("Current:", np.round(state, 2))
    print("First policy target:", np.round(actions[0], 2))
    print("First target minus current:", np.round(actions[0] - state, 2))
    input("Review the target, then press Enter to arm (Ctrl+C aborts): ")
    # Refresh after the review pause, just like the synchronous path.
    observation, state = make_observation(robot, args.prompt)
    actions = read_actions(client.infer(observation))
    check_chunk(actions, state, args.max_arm_step, args.max_gripper_step)
    check_transition(actions[0], state, args.max_arm_step, args.max_gripper_step)

    runner = AsyncInferenceRunner(client)
    action_idx = 0
    total_steps = 0
    request_start: int | None = None
    request_started_at: float | None = None
    last_target: np.ndarray | None = None
    underrun_logged = False
    started = deadline = time.monotonic()
    try:
        while time.monotonic() - started < args.duration:
            completed = runner.poll(args.rtc_timeout)
            if completed is not None:
                if request_start is None or request_started_at is None:
                    raise RuntimeError("RTC response has no request start step")
                rtt_ms = (time.monotonic() - request_started_at) * 1000
                real_delay = total_steps - request_start
                new_actions = read_actions(completed)
                request_start = None
                request_started_at = None
                if real_delay >= ACTION_HORIZON:
                    raise TimeoutError(f"RTC result expired after {real_delay} control steps")
                if real_delay >= args.rtc_execution_horizon:
                    LOG.warning(
                        "RTC switch index %d is beyond guided prefix s=%d; check measured latency",
                        real_delay, args.rtc_execution_horizon,
                    )
                if last_target is not None:
                    new_actions, limited_steps = limit_rtc_chunk(
                        new_actions, real_delay, last_target, args.max_arm_step, args.max_gripper_step,
                    )
                    if limited_steps:
                        LOG.warning("RTC rate-limited %d targets in new chunk at step=%d", limited_steps, total_steps)
                actions, action_idx = new_actions, real_delay
                underrun_logged = False
                LOG.info("RTC installed chunk: rtt=%.1f ms real_delay=%d next_index=%d", rtt_ms, real_delay, action_idx)

            if action_idx >= ACTION_HORIZON:
                if not runner.busy() or last_target is None:
                    raise TimeoutError("RTC action queue exhausted without a pending request")
                if not underrun_logged:
                    LOG.warning("RTC action queue underrun; holding the last target")
                    underrun_logged = True
                send_target(robot, last_target)
                total_steps += 1
                deadline += 1.0 / args.fps
                if (delay := deadline - time.monotonic()) > 0:
                    time.sleep(delay)
                continue

            if not runner.busy() and request_start is None and action_idx >= args.rtc_execution_horizon:
                observation, _ = make_observation(robot, args.prompt)
                leftover = np.array(actions[action_idx:], copy=True)
                request = rtc_request(observation, leftover, args)
                request_start = total_steps
                request_started_at = time.monotonic()
                runner.start(request)
                LOG.info("RTC requested at step=%d old_index=%d remaining=%d", total_steps, action_idx, len(leftover))

            target = actions[action_idx]
            send_target(robot, target)
            last_target = target
            action_idx += 1
            total_steps += 1
            deadline += 1.0 / args.fps
            if (delay := deadline - time.monotonic()) > 0:
                time.sleep(delay)
    finally:
        runner.close()


def run(args: argparse.Namespace) -> None:
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower import SO101FollowerConfig

    cameras = {
        name: OpenCVCameraConfig(index_or_path=index, fps=int(args.fps), width=640, height=480, fourcc="MJPG")
        for name, index in (("front", args.front_camera), ("wrist", args.wrist_camera))
    }
    robot = SO101Follower(SO101FollowerConfig(
        port=args.port_arm, id=args.robot_id, cameras=cameras, use_degrees=True,
        max_relative_target=None,
    ))
    if not robot.calibration_fpath.is_file():
        raise FileNotFoundError(f"Missing LeRobot calibration: {robot.calibration_fpath}")

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.server_port)
    LOG.info("Connected to policy server; metadata: %s", client.get_server_metadata())
    try:
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            raise RuntimeError("SO-101 calibration does not match the selected robot ID")

        if args.rtc:
            run_rtc(args, robot, client)
            return

        armed = False
        started = time.monotonic()
        cycle = 0
        while not armed or time.monotonic() - started < args.duration:
            observation, state = make_observation(robot, args.prompt)
            query_start = time.monotonic()
            result = client.infer(observation)
            rtt_ms = (time.monotonic() - query_start) * 1000
            actions = read_actions(result)
            selected = actions[:args.chunk_exec]
            LOG.info("cycle=%d rtt=%.1f ms state=%s first=%s", cycle, rtt_ms,
                     np.round(state, 2), np.round(selected[0], 2))

            if not args.execute:
                LOG.info("Dry run complete; add --execute to send motor commands")
                return

            if not armed:
                print("Current:", np.round(state, 2))
                print("First policy target:", np.round(selected[0], 2))
                print("First target minus current:", np.round(selected[0] - state, 2))
                input("Review the target, then press Enter to arm (Ctrl+C aborts): ")
                armed = True
                started = time.monotonic()
                # Refresh the observation after the review pause.
                continue

            check_chunk(selected, state, args.max_arm_step, args.max_gripper_step)
            deadline = time.monotonic()
            executed = 0
            for target in selected:
                if time.monotonic() - started >= args.duration:
                    break
                robot.send_action({f"{joint}.pos": float(value)
                                   for joint, value in zip(JOINTS, target, strict=True)})
                executed += 1
                deadline += 1.0 / args.fps
                if (delay := deadline - time.monotonic()) > 0:
                    time.sleep(delay)
            cycle += 1
            LOG.info("cycle=%d executed=%d elapsed=%.1f s", cycle, executed,
                     time.monotonic() - started)
    finally:
        client.close()
        safe_shutdown(robot, args.hold_on_exit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(parse_args())
    except KeyboardInterrupt:
        LOG.info("Interrupted")
