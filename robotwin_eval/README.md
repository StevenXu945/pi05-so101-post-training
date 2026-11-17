# Run RoboTwin Evaluation from This openpi Project

This directory provides entry points adapted from `RoboTwin/scripts/eval_openpi.py`
and `eval_openpi.sh`. It requires the original RoboTwin checkout's `envs`, `assets`,
`env_cfg`, and `description` directories, plus an existing RoboTwin Conda environment.
Keep simulation dependencies such as SAPIEN in that environment. The two original
scripts in the RoboTwin checkout remain unchanged.

## 1. Policy Server

Start a RoboTwin checkpoint server in the openpi environment with a matching
configuration, such as `pi05_robotwin_50ep_lora_finetune`. The server must return
`(16, 14)` actions; SO-101 checkpoints return six-dimensional actions and are
incompatible. An existing compatible server can be reused.

## 2. Validate Configuration

This checks configuration without starting simulation or connecting to the policy server.

```bash
conda activate RoboTwin
cd "/home/huayi/PycharmProjects/openpi (copy)"
bash robotwin_eval/eval_openpi.sh \
  --robotwin-root="/home/huayi/PycharmProjects/RoboTwin" \
  --task-name=place_a2b_left --task-config=demo_clean --validate-only
```

The default RoboTwin location is `../RoboTwin`, alongside this project. Override it
with `--robotwin-root` or the `ROBOTWIN_ROOT` environment variable. `OPENPI_ROOT`
is inferred from this directory's location by default.

## 3. Simulation Evaluation

Start the policy server and select a task matching the trained model, then run:

```bash
bash robotwin_eval/eval_openpi.sh \
  --robotwin-root="/home/huayi/PycharmProjects/RoboTwin" \
  --task-name=place_a2b_left --task-config=demo_clean \
  --host=127.0.0.1 --port=8000 --episodes=5 --exec-steps=8
```

Video recording is enabled by default and requires ffmpeg; use `--no-video` to
disable it. `--expert-check` runs an expert check before each episode. Other options
retain the original script's behavior. Replace `--task-name` with your evaluation task.

Results are saved under this project by default:

```text
eval_result/robotwin_openpi/<task>/<config>/<timestamp>/
  summary.json
  episode_000.mp4
  ...
```

Use `--output-dir` to choose another location. Relative paths resolve against the
working directory at launch. The Python subprocess switches to the RoboTwin root
for relative simulation asset paths; the calling terminal stays in this project.

After activating the RoboTwin environment, you can also run Python directly:

```bash
python robotwin_eval/eval_openpi.py --task-name=place_a2b_left --episodes=5
```

Validation to date covers configuration loading, entry points, and mocked rollouts.
Actual simulation evaluation has not been run as part of that validation.
