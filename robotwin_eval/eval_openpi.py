"""Evaluate an openpi WebSocket policy directly in RoboTwin, without XPolicyLab."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import yaml

PROJECT_ROOT = Path(os.environ.get("OPENPI_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
ROOT = Path(os.environ.get("ROBOTWIN_ROOT", PROJECT_ROOT.parent / "RoboTwin")).expanduser().resolve()
CONFIGS_PATH = ROOT / "env_cfg/task_config"

if TYPE_CHECKING:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy


def configure_paths(robotwin_root: Path):
    """Use RoboTwin's asset-relative cwd while keeping results in this project."""
    global ROOT, CONFIGS_PATH
    ROOT = robotwin_root.expanduser().resolve()
    for directory in ("envs", "env_cfg/task_config", "assets", "description/utils"):
        if not (ROOT / directory).is_dir():
            raise FileNotFoundError(f"Invalid RoboTwin root: missing {ROOT / directory}")
    CONFIGS_PATH = ROOT / "env_cfg/task_config"
    for path in (PROJECT_ROOT / "packages/openpi-client/src", ROOT, ROOT / "description/utils"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(ROOT)


def load_config(task_name: str, task_config: str) -> dict:
    config_path = Path(CONFIGS_PATH) / f"{task_config}.yml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    with (Path(CONFIGS_PATH) / "_embodiment_config.yml").open(encoding="utf-8") as stream:
        embodiments = yaml.safe_load(stream)
    with (Path(CONFIGS_PATH) / "_camera_config.yml").open(encoding="utf-8") as stream:
        cameras = yaml.safe_load(stream)

    embodiment = config["embodiment"]
    if len(embodiment) == 1:
        left_name = right_name = embodiment[0]
        config["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        left_name, right_name, config["embodiment_dis"] = embodiment
        config["dual_arm_embodied"] = False
    else:
        raise ValueError(f"Unsupported embodiment: {embodiment}")
    for side, name in (("left", left_name), ("right", right_name)):
        robot_path = ROOT / embodiments[name]["file_path"]
        config[f"{side}_robot_file"] = str(robot_path.resolve())
        with (robot_path / "config.yml").open(encoding="utf-8") as stream:
            config[f"{side}_embodiment_config"] = yaml.safe_load(stream)

    camera = cameras[config["camera"]["head_camera_type"]]
    config["head_camera_h"] = camera["h"]
    config["head_camera_w"] = camera["w"]
    config["task_name"] = task_name
    config["task_config"] = task_config
    config["eval_mode"] = True
    config["collect_data"] = False
    config.setdefault("data_type", {}).update(rgb=True, qpos=True)
    return config


def create_task(task_name: str):
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def make_instruction(task_name: str, episode_info: dict, instruction_type: str, seed: int) -> str:
    from generate_episode_instructions import generate_episode_descriptions

    generated = generate_episode_descriptions(
        task_name, [episode_info.get("info", {})], 100, seed=seed
    )
    candidates = generated[0].get(instruction_type) if generated else None
    return random.Random(f"{seed}:select").choice(candidates) if candidates else task_name.replace("_", " ")


def encode_observation(raw: dict, prompt: str) -> dict:
    cameras = raw["observation"]
    state = np.asarray(raw["joint_action"]["vector"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite 14D qpos state, got {state.shape}")
    result = {"observation/state": state, "prompt": prompt}
    for output_key, camera_key in (
        ("observation/image", "head_camera"),
        ("observation/left_wrist_image", "left_camera"),
        ("observation/right_wrist_image", "right_camera"),
    ):
        image = np.asarray(cameras[camera_key]["rgb"])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image for {camera_key}, got {image.shape}")
        result[output_key] = image.astype(np.uint8, copy=False)
    return result


def run_episode(task_env, client: WebsocketClientPolicy, config: dict, episode_id: int,
                seed: int, instruction_type: str, exec_steps: int, *, expert_check: bool,
                video_dir: Path | None, video_fps: int) -> dict:
    episode_info = {"info": {}}
    if expert_check:
        try:
            task_env.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True,
                                **{**config, "eval_video_save_dir": None})
            episode_info = task_env.play_once() or {"info": {}}
            expert_succeeded = bool(task_env.plan_success and task_env.check_success())
        finally:
            task_env.close_env()
        if not expert_succeeded:
            return {"episode": episode_id, "seed": seed, "skipped": True, "reason": "expert_failed"}

    video_path = video_dir / f"episode_{episode_id:03d}.mp4" if video_dir is not None else None
    ffmpeg = None
    try:
        task_env.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True,
                            **{**config, "eval_video_save_dir": str(video_dir) if video_dir else None})
        if not expert_check:
            episode_info = task_env.info
        instruction = make_instruction(config["task_name"], episode_info, instruction_type, seed)
        task_env.set_instruction(instruction=instruction)
        if video_path is not None:
            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                 "-video_size", f"{config['head_camera_w']}x{config['head_camera_h']}",
                 "-framerate", str(video_fps), "-i", "-", "-pix_fmt", "yuv420p",
                 "-vcodec", "libx264", "-crf", "23", str(video_path)],
                stdin=subprocess.PIPE,
            )
            task_env._set_eval_video_ffmpeg(ffmpeg)  # noqa: SLF001 - RoboTwin's recorder hook
        while not task_env.eval_success and task_env.take_action_cnt < task_env.step_lim:
            raw = task_env.get_obs()
            actions = np.asarray(client.infer(encode_observation(raw, instruction))["actions"], dtype=np.float32)
            if actions.shape != (16, 14) or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite (16, 14) actions, got {actions.shape}")
            for action in actions[:exec_steps]:
                task_env.take_action(action, action_type="qpos")
                if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                    break
        return {
            "episode": episode_id,
            "seed": seed,
            "instruction": instruction,
            "success": bool(task_env.eval_success),
            "steps": int(task_env.take_action_cnt),
            "video": str(video_path) if video_path is not None else None,
        }
    finally:
        try:
            if ffmpeg is not None:
                task_env._del_eval_video_ffmpeg()  # noqa: SLF001 - RoboTwin's recorder hook
                if ffmpeg.returncode != 0:
                    raise RuntimeError(f"ffmpeg failed for {video_path}: exit code {ffmpeg.returncode}")
        finally:
            task_env.close_env()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, default=ROOT, help="RoboTwin checkout containing envs/ and assets/")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-seed-attempts", type=int, default=None)
    parser.add_argument("--exec-steps", type=int, default=8)
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default=None)
    parser.add_argument("--expert-check", action="store_true", help="Replay expert before each evaluation episode")
    parser.add_argument("--no-video", action="store_true", help="Disable MP4 recording")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "eval_result/robotwin_openpi")
    parser.add_argument("--validate-only", action="store_true", help="Validate configuration without starting simulation")
    args = parser.parse_args(argv)
    if args.episodes < 1 or not 1 <= args.exec_steps <= 16:
        parser.error("--episodes must be positive and --exec-steps must be between 1 and 16")
    max_seed_attempts = args.max_seed_attempts or args.episodes * 50
    if max_seed_attempts < args.episodes:
        parser.error("--max-seed-attempts must be at least --episodes")
    if args.video_fps < 1:
        parser.error("--video-fps must be positive")

    # Resolve output before changing cwd for RoboTwin's relative resource paths.
    args.output_dir = args.output_dir.expanduser().resolve()
    configure_paths(args.robotwin_root)
    if not (ROOT / "envs" / f"{args.task_name}.py").is_file():
        raise FileNotFoundError(f"Unknown RoboTwin task: {args.task_name}")
    config = load_config(args.task_name, args.task_config)
    instruction_type = args.instruction_type or config["eval_instruction"]
    if instruction_type not in {"seen", "unseen"}:
        raise ValueError(f"Invalid instruction type: {instruction_type}")
    print(f"task={args.task_name} config={args.task_config} instruction={instruction_type} "
          f"server={args.host}:{args.port} exec_steps={args.exec_steps}")
    if args.validate_only:
        return

    # Simulation imports are deferred so --help / --validate-only need no SAPIEN.
    from envs.utils.create_actor import UnStableError
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    run_dir = args.output_dir / args.task_name / args.task_config / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir.mkdir(parents=True, exist_ok=True)
    task_env = create_task(args.task_name)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    results = []
    try:
        candidate_seed = 100000 * (1 + args.seed)
        attempts = 0
        while len(results) < args.episodes:
            if attempts >= max_seed_attempts:
                raise RuntimeError(f"Only {len(results)} episodes completed after {attempts} seed attempts")
            attempts += 1
            try:
                result = run_episode(task_env, client, config, len(results), candidate_seed,
                                     instruction_type, args.exec_steps, expert_check=args.expert_check,
                                     video_dir=None if args.no_video else run_dir, video_fps=args.video_fps)
            except UnStableError:
                result = {"seed": candidate_seed, "skipped": True, "reason": "unstable"}
            candidate_seed += 1
            if result.get("skipped"):
                print(f"skipped seed={result['seed']}: {result['reason']}")
                continue
            results.append(result)
            successes = sum(item["success"] for item in results)
            print(f"episode={result['episode']} seed={result['seed']} success={result['success']} "
                  f"steps={result['steps']} cumulative={successes}/{len(results)}")
    finally:
        client.close()

    summary = {"task": args.task_name, "task_config": args.task_config,
               "successes": sum(item["success"] for item in results), "episodes": len(results), "results": results}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
