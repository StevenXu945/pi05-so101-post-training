# SO-101 lightweight human-in-the-loop collection

This directory provides real-robot collection with policy execution, keyboard-driven
human intervention, dataset validation, and RECAP conversion and merging. It uses
official LeRobot, pinned to
`huggingface/lerobot@145cc7ea53e5864bd5efec4e9bb70ee799e47bd8`, rather than the
Evo-RL fork. This revision requires Python 3.12 and provides the SO leader APIs
`send_feedback()`, `enable_torque()`, and `disable_torque()`.

## Installation

From the repository root:

```bash
cd so101_hil_lite
bash setup_env.sh
source .venv/bin/activate
```

You need `uv`, network access to download Python 3.12 and dependencies, and an
interactive Linux terminal. The setup script installs dependencies into this
directory's `.venv`; it does not modify the root project environment or an existing
`so101_hil` environment. Official LeRobot still depends on PyTorch: “lightweight”
refers to the collection code and runtime workflow, not the absence of ML dependencies.

## Terminal 1: start the policy server

Use the existing openpi environment from the repository root:

```bash
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101_lora_finetune \
  --policy.dir=checkpoints/pi05_so101_lora_finetune/so101_run_1/29999 \
  --port=8000
```

The configuration must match the checkpoint. For this example, the policy must
already have learned the cube-stacking task.

## Terminal 2: verify observations and inference

Activate the collection environment in this directory. Replace the device paths,
camera indices, and robot IDs with those for your calibrated hardware:

```bash
python record.py \
  --follower-port=/dev/ttyACM0 --follower-id=my_follower \
  --leader-port=/dev/ttyACM1 --leader-id=my_leader \
  --front-camera=0 --wrist-camera=2 \
  --policy-id=so101_run_1_29999 \
  --dry-run
```

The dry run connects the devices, reads observations, and requests one inference
without sending joint targets. Calibrate both arms with the pinned official
LeRobot version first. Connections use `calibrate=False` and do not implicitly
recalibrate. If an older calibration format is incompatible, recalibrate without
overwriting calibration files used by the original environment.

The default policy endpoint is `127.0.0.1:8000`; use `--host` and `--port` to change
it. The default `--prompt` is `Stack the red cube on top of the blue cube`.
`--policy-id` records the collector policy version in dataset metadata; it does
not select the checkpoint served by the policy server.

## Collect episodes

Remove `--dry-run` and choose a new dataset directory:

```bash
python record.py \
  --follower-port=/dev/ttyACM0 --follower-id=my_follower \
  --leader-port=/dev/ttyACM1 --leader-id=my_leader \
  --front-camera=0 --wrist-camera=2 \
  --policy-id=so101_run_1_29999 \
  --root=data/stack_v1 --repo-id=local/so101_stack_v1 \
  --episodes=10 --duration=30 --fps=30
```

The first inference warms up the policy and prints its target. Before each episode,
leader torque is disabled. Press **Enter** to enable follower tracking, then move
the leader to set the starting pose and arrange the cubes. Press **Enter** again
to check alignment and start the policy. Alignment frames are not recorded.
During autonomous execution, the leader actively follows the accepted follower
targets. The first five joints use degrees; the gripper uses LeRobot's 0–100 scale.

Enter controls in the terminal running the recorder; no global keyboard listener
is used:

| Key / event | Behavior |
| --- | --- |
| `i` during policy execution | Pause the policy and hold both arms, keeping leader torque enabled. |
| **Enter** while paused | Release leader torque and begin manual control. Grip and support the leader before confirming. |
| `i` during manual control | Check alignment, clear cached policy actions, and return control to the policy. If alignment fails, remain in manual control and retry after aligning. |
| `s` | End the episode as `success`. |
| `f` | End the episode as `failure`. |
| `q` | Save the episode as `aborted` and exit. |
| Time limit reached | Explicitly enter `success`, `failure`, or `aborted`; there is no default outcome. |

At episode end, the leader releases torque and the follower holds its current
pose for manual reset. There is no automatic return to a home pose. Before
returning to policy control, the controller checks alignment and writes a
current-position hold target before enabling leader torque.

Default per-step limits are 6.51 degrees for the first five joints and 20 for the
gripper. Adjust them with `--max-arm-step` and `--max-gripper-step`. Policy and
manual targets are clipped relative to both the previous sent target and the
current measured position. Rate-limit messages show `raw -> command`; the raw
target is saved as `selected_action`, and the accepted target as `action`. If a
tracking gap makes both limits impossible to satisfy, collection stops with an
error.

Inference requests are synchronous. By default, the recorder uses the first eight
actions of each returned chunk (`--chunk-steps=8`), with a request timeout of
10 seconds (`--timeout=10`). No next action is sent while waiting for inference.
Network and inference latency therefore affect the actual collection rate;
`--fps=30` does not guarantee a constant 30 Hz. Keyboard events are handled at
loop boundaries, so takeover can also be delayed during an inference request.

For an advantage-conditioned RECAP policy, `--advantage=positive` or
`--advantage=negative` adds `adv_ind` to inference requests. Omit this option for
the original LoRA workflow.

## Recorded data

LeRobot stores camera images, `observation.state`, and `action`. Here, `action` is
the accepted target returned by `follower.send_action()`, not the measured joint
position at the next timestep. Additional per-frame fields use the
`complementary_info.` prefix:

| Field | Meaning |
| --- | --- |
| `selected_action` | Target selected before rate limiting. |
| `policy_action` | Raw policy target, or a zero placeholder when unavailable. |
| `policy_action_valid` | Whether the frame contains a valid policy target; zero during manual control and takeover pauses. |
| `is_intervention` | Whether manual control is active. |
| `is_paused` | Whether the recorder is holding while waiting for takeover confirmation. |
| `elapsed_s` | Actual elapsed time since the episode started. |

Episode outcomes and collector policy versions are saved in `outcomes.jsonl` at
the dataset root. This is a custom sidecar, not Evo-RL's `episode_success`
metadata; downstream converters must read it. Episodes missing sidecar records
require inspection and must not be assumed to be failures. Exclude aborted
trajectories from training conversion.

Takeover-confirmation pauses are recorded with `is_paused=1`. They do not count
toward `--duration`, but `elapsed_s` retains wall-clock elapsed time. Handle these
hold frames explicitly during training conversion. Existing dataset paths are
rejected to prevent overwrites; resuming collection is not supported.

## Validate and convert to RECAP

Use the same environment as the recorder; no additional LeRobot version is needed:

```bash
python check_dataset.py data/stack_v1 --report=stack_v1_report.json
python convert_to_recap.py data/stack_v1 data/stack_v1_recap_v21 \
  --fps=30 --task-max-steps=1201 --round-id=1 --validation-episodes 1
```

The destination must be new. Conversion leaves the source unchanged and produces
LeRobot v2.1 data readable by the older openpi pipeline: per-episode Parquet files,
re-encoded and aligned videos, and a `meta` directory.

The converter excludes aborted episodes, removes paused frames and pause time,
and resamples images, states, and actions by holding the preceding source frame.
It preserves the terminal frame and stores `source_frame_index` for traceability.
Active sampling gaps longer than 0.5 seconds are rejected by default. After
reviewing the report, you can adjust `--max-gap-s`; this cannot recover missing
observations.

Labels use the `paper-v1` recipe:

- Nonterminal raw reward: `-1`.
- Successful terminal reward: `0`.
- Failed terminal reward: `-C_fail`.
- `reward_label`: reward normalized with a fixed per-task scale fitted on training episodes.
- `value_label`: the exact cumulative return of the normalized rewards.
- Human intervention frames: `positive`; autonomous frames: `none`.

Use `--validation-episodes` to specify source episode IDs to hold out. Scales and
splits are saved in `meta/conversion.json`. Explicitly reserve validation episodes
for a full training run. Reconvert older `remaining-time-v1` datasets before using
them for the current value-training workflow.

The output includes `meta/recap_manifest.json` for merging demonstrations and
rollouts from successive rounds with `merge_recap_datasets.py`. Later rounds must
reuse the first round's task scales through `--task-scales-json`. The current
RECAP configuration reads `adv_ind`; the original LoRA configuration retains its
existing behavior. Do not add a conditional-policy warm-up before value training.

Place the converted dataset under the existing `HF_LEROBOT_HOME/repo_id` location,
or register it through your training workflow. The converter does not change
training configurations or cache locations.

See the [SO-101 RECAP guide](../docs/so101_recap.md) for the cumulative iteration
workflow, v2.1 demonstration import, threshold statistics, and runtime commands.
See also the [value runtime README](../value_runtime/README.md).

## Code layout

| File | Purpose |
| --- | --- |
| `record.py` | Hardware setup, episode loop, alignment, and terminal controls. |
| `controller.py` | Control ownership, takeover transitions, and action limits. |
| `remote_policy.py` / `codec.py` | WebSocket inference client and message serialization. |
| `recording.py` | LeRobot frame writing and outcome sidecar metadata. |
| `check_dataset.py` / `dataset_tools.py` | Dataset validation, resampling helpers, and labels. |
| `convert_to_recap.py` | Conversion to the RECAP v2.1 dataset format. |
| `merge_recap_datasets.py` | Demonstration import and cumulative dataset merging. |

Standalone value-model training and advantage labeling live in
[`../value_runtime/`](../value_runtime/), with `train.py` and `label_advantage.py`
as the main entry points. See the [repository code layout](../docs/code_layout.md)
for the broader structure.
