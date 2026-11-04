# π₀.₅ Post-Training and Deployment: SO-101, RoboTwin 2.0, Guided RTC, and RECAP

Built on [openpi](https://github.com/Physical-Intelligence/openpi), this project provides:

- **π₀.₅ post-training and deployment**: LoRA post-training, policy serving, and inference on SO-101 hardware and in RoboTwin 2.0 simulation.
- **SO-101 Guided RTC**: prefix-guided real-time chunking that overlaps policy inference with action execution on hardware.
- **SO-101 RECAP implementation**: autonomous and human-intervention rollout collection, value-model training, advantage labeling, and iterative advantage-conditioned policy training.

RECAP starts from task-specific SFT and does not reproduce the full official π*₀.₆ pretraining pipeline. Hardware examples use `Grab the toy into the box`; keep task instructions consistent across collection, training, and inference.

## Demo Highlights

**SO-101 Guided RTC**

Coming soon.

<!-- ![SO-101 Guided RTC](media/so101_rtc.gif) -->

**RoboTwin: Adjust Bottle**

Coming soon.

<!-- ![RoboTwin adjust bottle](media/robotwin_adjust_bottle_1.gif) -->

**SO-101 RECAP Autonomous Inference**

Coming soon.

<!-- ![SO-101 RECAP autonomous inference](media/recap_autonomous.gif) -->

## Contents

- [Environments and Base Weights](#environment)
- [SO-101 Calibration, Post-Training, Deployment, and RTC](#so101)
- [RoboTwin 2.0 Post-Training and Simulation](#robotwin)
- [SO-101 RECAP Workflow](#recap)

<a id="environment"></a>
## 1. Environments and Base Weights

Development setup: **Ubuntu 22.04 · RTX 3090 Ti (24 GB VRAM) · 32 GB RAM**. Adjust batch size, training steps, frozen modules, data-loader workers, and validation frequency to suit your hardware.

Use separate environments to avoid dependency conflicts:

| Environment | Python | Purpose |
|---|---|---|
| Root `.venv` | 3.11, using the project lockfile | openpi policy training, normalization statistics, and serving |
| Conda `so101_hil_lite` | 3.12 | SO-101 calibration, collection, hardware clients, and data conversion/merging |
| Conda `openpi-recap-value` | 3.11 | Independent value-model training and advantage labeling |
| Conda `RoboTwin` | 3.10 | RoboTwin simulation and evaluation clients |

Clone this repository together with its submodules:

```bash
git clone --recurse-submodules https://github.com/StevenXu945/pi05-so101-post-training.git
cd pi05-so101-post-training
export PROJECT_ROOT="$PWD"
```

Run subsequent commands from the repository root unless stated otherwise. In each new terminal, enter the repository directory and set `PROJECT_ROOT="$PWD"`.

### 1.1 openpi: Official Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git LFS, then follow the [official openpi installation instructions](https://github.com/Physical-Intelligence/openpi#installation):

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
.venv/bin/python -c 'import jax; print(jax.devices())'
```

Both SO-101 and RoboTwin training configurations load base weights from `./pi05_base/params`. Download them if needed:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil
from openpi.shared import download
source = Path(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base"))
target = Path("pi05_base")
if target.exists():
    raise SystemExit("pi05_base already exists; check its params directory instead of overwriting it")
shutil.copytree(source, target)
PY
```

LoRA checkpoints store only trainable parameters. Keep `pi05_base/params` available for deployment.

### 1.2 SO-101: Separate LeRobot Environment

LeRobot is pinned to `huggingface/lerobot@145cc7ea53e5864bd5efec4e9bb70ee799e47bd8`, which supports leader feedback control.

Install with Conda:

```bash
conda create -n so101_hil_lite python=3.12 -y
conda activate so101_hil_lite
python -m pip install -r so101_hil_lite/requirements.txt
python -m pip check
```

After activating the SO-101 environment, set the client source path:

```bash
export PYTHONPATH="$PROJECT_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
```

### 1.3 RECAP Value-Model Environment

Required only for RECAP. Keep the constraints file when installing to avoid TensorFlow/protobuf dependency conflicts:

```bash
conda create -n openpi-recap-value python=3.11 -y
conda activate openpi-recap-value
python -m pip install -r value_runtime/requirements.txt -c value_runtime/constraints-tested.txt
python -m pip check
python -c 'import jax; print(jax.devices())'
```

Install `hf` using the [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli), then download the value-model initialization weights:

```bash
hf download ybpy/vlm_ckpt \
  --include 'gemma-3-270m/step_00020000/*' --include 'tokenizer.model' \
  --local-dir .openpi_cache/recap_value
```

<a id="so101"></a>
## 2. SO-101 Hardware Post-Training and Deployment

**GIF demos** — add the files under `media/` and replace the placeholders with inline images.

| Data collection: three combined views | Standard inference | RTC inference |
|---|---|---|
| Coming soon | Coming soon | Coming soon |

<!-- Replace the placeholders with:
![Three-view data collection](media/so101_collection.gif)
![Standard inference](media/so101_inference.gif)
![Guided RTC inference](media/so101_rtc.gif)
-->

### 2.1 Assembly, Motor Setup, and Calibration

Follow the [official Seeed Studio SO-ARM100/SO-ARM101 guide](https://wiki.seeedstudio.com/lerobot_so100m_new/) for assembly, motor IDs, wiring, and calibration. Use the LeRobot version specified in Section 1.2.

Calibrate in the SO-101 environment, replacing ports and IDs with your device settings:

```bash
conda activate so101_hil_lite
lerobot-calibrate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm

lerobot-calibrate \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm

lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm
```

Reuse the same calibration IDs for collection and deployment. Stable `/dev/serial/by-id/...` paths can replace numbered ports. Examples use camera `0` for the front view and `2` for the wrist view.

### 2.2 Collect Demonstrations

Run the official `lerobot-record` in the SO-101 environment for human teleoperation; no policy server is required:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}' \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm \
  --dataset.root="$PROJECT_ROOT/raw_data/so101_sft" \
  --dataset.repo_id=local/so101_sft_raw \
  --dataset.single_task="Grab the toy into the box" \
  --dataset.num_episodes=50 --dataset.episode_time_s=40 \
  --dataset.reset_time_s=10 --dataset.push_to_hub=false --display_data=false
```

Adjust the episode count as needed and keep demonstrations consistent with the task instruction.

### 2.3 Convert to LeRobot v2.1

Convert using the openpi environment:

```bash
export HF_LEROBOT_HOME="$PROJECT_ROOT/converted_lerobot"
.venv/bin/python scripts/convert_so101_v3_to_v21.py \
  --source="$PROJECT_ROOT/raw_data/so101_sft" \
  --repo-id=local/so101_train_dataset
```

Use a new output directory. Skip conversion if your v2.1 dataset already matches the configuration.

### 2.4 Normalization Statistics and LoRA Post-Training

```bash
export HF_LEROBOT_HOME="$PROJECT_ROOT/converted_lerobot"
.venv/bin/python scripts/compute_norm_stats.py \
  --config-name=pi05_so101_lora_finetune

.venv/bin/python scripts/train.py pi05_so101_lora_finetune \
  --exp-name=so101_run_1 --num-train-steps=30000 --no-wandb-enabled
```

Configuration: front/wrist images, 6-dimensional state/action, 16-step action chunks, and batch size 2.

The final checkpoint in this example is `29999`. Update deployment paths when changing the experiment name or training steps.

### 2.5 Standard Inference

Terminal 1: start the policy server in the openpi environment:

```bash
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101_lora_finetune \
  --policy.dir=checkpoints/pi05_so101_lora_finetune/so101_run_1/29999 \
  --default-prompt="Grab the toy into the box" --port=8000
```

Terminal 2: activate the SO-101 environment and set `PYTHONPATH` as in Section 1.2. Omit `--execute` to check observations and inference without sending motor commands:

```bash
python scripts/eval_so101.py \
  --port-arm=/dev/ttyACM0 --robot-id=my_awesome_follower_arm \
  --front-camera=0 --wrist-camera=2 --host=127.0.0.1 --server-port=8000 \
  --prompt="Grab the toy into the box" \
  --duration=40 --fps=30 --chunk-exec=8 --execute \
  --video-dir=eval_result/so101_standard_run1
```

### 2.6 RTC Inference

Use the same policy server. The RTC client overlaps inference with execution and uses prefix guidance to connect action chunks.

```bash
python scripts/eval_so101.py \
  --port-arm=/dev/ttyACM0 --robot-id=my_awesome_follower_arm \
  --front-camera=0 --wrist-camera=2 --host=127.0.0.1 --server-port=8000 \
  --prompt="Grab the toy into the box" --duration=40 --fps=30 \
  --rtc --rtc-inference-delay=6 --rtc-execution-horizon=8 \
  --rtc-max-guidance-weight=10 --execute \
  --video-dir=eval_result/so101_rtc_run1
```

Optional `--video-dir` saves full-resolution RGB camera views as `front.mp4` and `wrist.mp4`, with a `recording.json` summary. Recording starts after arming and runs in a background thread, including inference waits. Video timestamps follow elapsed time; the requested FPS is a sampling target, not a guarantee. Use a new output directory for each run. Dry runs do not record.


<a id="robotwin"></a>
## 3. RoboTwin 2.0 Post-Training and Simulation

**GIF demos: adjust bottle**

| Inference: clip 1 | Inference: clip 2 |
|---|---|
| Coming soon | Coming soon |

<!-- Replace the placeholders with:
![Adjust bottle: clip 1](media/robotwin_adjust_bottle_1.gif)
![Adjust bottle: clip 2](media/robotwin_adjust_bottle_2.gif)
-->

### 3.1 Install the Simulation Environment

Follow the [official RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) to prepare Vulkan, GPU drivers, ffmpeg, and simulation assets:

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git "$PROJECT_ROOT/../RoboTwin"
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
cd "$PROJECT_ROOT/../RoboTwin"
bash scripts/_install.sh
bash scripts/_download_assets.sh
cd "$PROJECT_ROOT"
```

Skip installation if the environment is already available. Evaluation still requires the RoboTwin checkout and assets.

### 3.2 Prepare Training Data

Training uses **LeRobot v2.1** data. Choose either of the following methods.

#### Option A: Download the Official v2.1 Dataset

Download the official [LeRobot v2.1 dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/lerobot_dataset) directly, without HDF5 conversion. The archive is approximately 2.40 GB:

```bash
hf download TianxingChen/RoboTwin2.0 \
  --repo-type=dataset \
  --include 'lerobot_dataset/RoboTwin_lerobot_v21.zip' \
  --local-dir "$PROJECT_ROOT/datasets/robotwin_download"

unzip "$PROJECT_ROOT/datasets/robotwin_download/lerobot_dataset/RoboTwin_lerobot_v21.zip" \
  -d "$PROJECT_ROOT/datasets/robotwin"
find "$PROJECT_ROOT/datasets/robotwin" -path '*/meta/info.json'
```

Set `HF_LEROBOT_HOME / repo_id` to the extracted dataset root containing `meta/`, `data/`, and `videos/`. For `datasets/robotwin/RoboTwin_lerobot_v21`, set `HF_LEROBOT_HOME="$PROJECT_ROOT/datasets/robotwin"` and change the training configuration’s `repo_id` to `RoboTwin_lerobot_v21`. Use the same path for statistics and training.

#### Option B: Collect and Convert Using Official Tools

Run the official collection script in the RoboTwin environment. This example uses `adjust_bottle`, `demo_clean`, and GPU 0. The task configuration controls episode count, cameras, and robot settings:

```bash
conda activate RoboTwin
cd "$PROJECT_ROOT/../RoboTwin"
bash collect_data.sh adjust_bottle demo_clean 0
```

HDF5 trajectories are saved under `data/<task_config>/<task_name>/<embodiment>/data/`. Convert the selected task with the official v2.1 converter:

```bash
# Run in a separate environment with the required LeRobot v2.1 conversion dependencies.
cd "$PROJECT_ROOT/../RoboTwin"
export HF_LEROBOT_HOME="$PROJECT_ROOT"
python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "demo_clean.adjust_bottle.aloha_agilex" \
  --repo_id RoboTwin_lerobot_v21_50ep \
  --max_episode 50
cd "$PROJECT_ROOT"
```

Configure conversion dependencies separately according to your RoboTwin version. `--max_episode 50` limits each matching task to at most 50 episodes; it does not create missing episodes.

The following examples use `$PROJECT_ROOT/RoboTwin_lerobot_v21_50ep`. The `50ep` configuration name does not automatically subset data. Update `repo_id` and `HF_LEROBOT_HOME` when using the full official dataset. The policy uses three camera views and 14-dimensional dual-arm state/action.

### 3.3 Normalization Statistics and Training

Run from the project root using the openpi environment. These commands target the 50-episode subset; for the full official dataset, replace `HF_LEROBOT_HOME` with the path configured in Section 3.2:

```bash
export HF_LEROBOT_HOME="$PROJECT_ROOT"
.venv/bin/python scripts/compute_norm_stats.py \
  --config-name=pi05_robotwin_50ep_lora_finetune

.venv/bin/python scripts/train.py pi05_robotwin_50ep_lora_finetune \
  --exp-name=robotwin_50ep_run1 --num-train-steps=30000 --no-wandb-enabled
```

Task instructions are read from the dataset. Defaults are batch size 2 and horizon 16. For a short test, use 1,000 steps and checkpoint `999`.

### 3.4 Policy Serving and Simulation Evaluation

Terminal 1:

```bash
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_robotwin_50ep_lora_finetune \
  --policy.dir=checkpoints/pi05_robotwin_50ep_lora_finetune/robotwin_50ep_run1/29999 \
  --port=8000
```

Terminal 2: activate `RoboTwin` and run evaluation. Add `--validate-only` to check the configuration without starting simulation:

```bash
conda activate RoboTwin
bash robotwin_eval/eval_openpi.sh \
  --robotwin-root="$PROJECT_ROOT/../RoboTwin" \
  --task-name=adjust_bottle --task-config=demo_clean \
  --host=127.0.0.1 --port=8000 --episodes=5 --exec-steps=8
```

Match the task to your training data. Videos are saved by default (ffmpeg required); results are stored under `eval_result/robotwin_openpi/`.

<a id="recap"></a>
## 4. SO-101 RECAP: Collection, Value Training, and Policy Iteration

**GIF demos**

| Autonomous inference: leader and follower | Human intervention and return to policy control |
|---|---|
| Coming soon | Coming soon |

<!-- Replace the placeholders with:
![RECAP autonomous inference](media/recap_autonomous.gif)
![RECAP human intervention](media/recap_intervention.gif)
-->

Workflow: **SFT initialization → rollouts → cumulative data pool → value training → advantage labeling → conditioned policy training → deployment and next-round collection**. The value model runs offline only.

### 4.1 Deploy the Initial Policy and Collect Rollouts

Use the SFT server from Section 2.5 in Terminal 1. In Terminal 2, activate the SO-101 environment:

```bash
conda activate so101_hil_lite
python so101_hil_lite/record.py \
  --follower-port=/dev/ttyACM0 --follower-id=my_awesome_follower_arm \
  --leader-port=/dev/ttyACM1 --leader-id=my_awesome_leader_arm \
  --front-camera=0 --wrist-camera=2 --host=127.0.0.1 --port=8000 \
  --policy-id=so101_run_1_29999 --prompt="Grab the toy into the box" \
  --root=so101_hil_lite/data/toy_round1 --repo-id=local/so101_toy_round1 \
  --episodes=20 --duration=40 --fps=30
```

Add `--dry-run` to check connectivity and inference first. During collection:

1. Press Enter to enable alignment. Move the leader so the follower tracks it to the initial pose.
2. Arrange the objects and press Enter again to start policy execution and recording. The leader follows the policy during autonomous execution.
3. Press `i` to pause; both arms hold position. Support the leader, then press Enter to take manual control.
4. After correcting the motion, press `i` to check alignment, clear the action cache, and return control to the policy.
5. Press `s` for success or `f` for failure. Press `q` to mark the episode as aborted and exit. Select an outcome explicitly after a timeout.

Reset the scene manually between episodes. Per-frame intervention flags are stored in the dataset, and episode outcomes in `outcomes.jsonl`. Conversion excludes paused frames and aborted episodes. Include intervention-free success/failure rollouts for training and threshold estimation. HIL uses synchronous inference, without RTC.

### 4.2 Validate, Set Reward Scales, and Split Data

In the SO-101 environment:

```bash
python so101_hil_lite/check_dataset.py so101_hil_lite/data/toy_round1
python so101_hil_lite/convert_to_recap.py \
  so101_hil_lite/data/toy_round1 datasets/toy/prepared/round1 \
  --fps=30 --task-max-steps=1201 --round-id=1 \
  --validation-episodes 16 17 18 19
```

Adjust validation IDs to your episodes. This example allows 1201 frames: nonterminal reward -1, successful terminal reward 0, failed terminal reward -1201, and normalization scale 2401. Reuse the fixed scales in the first round’s `recap_manifest.json` for later rounds.

### 4.3 Import Demonstrations and Build the Cumulative Pool

If the demonstrations from Section 2 cover the same toy task and are all successful:

```bash
python so101_hil_lite/merge_recap_datasets.py prepare-demonstrations \
  converted_lerobot/local/so101_train_dataset datasets/toy/prepared/demos \
  --task-scales-json=datasets/toy/prepared/round1/meta/recap_manifest.json \
  --policy-id=human_demo --assume-success

python so101_hil_lite/merge_recap_datasets.py merge \
  datasets/toy/prepared/demos datasets/toy/prepared/round1 \
  --output=datasets/toy/pools/round1
```

Use `--assume-success` only for entirely successful demonstrations; otherwise provide `--outcomes-json`. To omit demonstrations, pass only round1 to the merge command. All output directories must be new.

### 4.4 Train the Value Model

Stop the GPU policy server and run in the value-model environment:

```bash
conda activate openpi-recap-value
python value_runtime/train.py \
  --data=datasets/toy/pools/round1 \
  --checkpoint=.openpi_cache/recap_value/gemma-3-270m/step_00020000 \
  --tokenizer=.openpi_cache/recap_value/tokenizer.model \
  --output=checkpoints/recap_value/toy_round1 \
  --freeze-mode=all_backbones --batch-size=1 --steps=5000 \
  --val-interval=1000 --save-interval=1000
```

The SigLIP + Gemma3-270M value model takes images, task instructions, and state as input and predicts a value distribution. `all_backbones` freezes the backbones and trains only fusion layers and the value head.

### 4.5 Compute N-Step Advantages and Labels

```bash
python value_runtime/label_advantage.py \
  --data=datasets/toy/pools/round1 \
  --checkpoint=checkpoints/recap_value/toy_round1/step_00005000 \
  --tokenizer=.openpi_cache/recap_value/tokenizer.model \
  --output=datasets/toy/labeled/round1 \
  --threshold-round=1 --lookahead=50 --positive-ratio=0.4 --batch-size=2
```

At 30 Hz, `N=50` spans approximately 1.67 seconds. Thresholds are estimated using only complete, intervention-free training rollouts from this round. Human frames remain positive, and validation frames remain none. Labels are written to a new dataset copy.

### 4.6 Advantage-Conditioned LoRA Training

Use the openpi environment and register this round’s dataset under a separate data parent directory:

```bash
export HF_LEROBOT_HOME="$PROJECT_ROOT/datasets/toy/policy_round1"
mkdir -p "$HF_LEROBOT_HOME/local"
ln -s "$PROJECT_ROOT/datasets/toy/labeled/round1" \
  "$HF_LEROBOT_HOME/local/so101_stack_recap_labeled_v21"

.venv/bin/python scripts/compute_norm_stats.py \
  --config-name=pi05_so101_recap_lora_relabeled

.venv/bin/python scripts/train.py pi05_so101_recap_lora_relabeled \
  --data.default-prompt="Grab the toy into the box" \
  --exp-name=toy_recap_round1 --num-train-steps=10000 --no-wandb-enabled
```

Keep the configured dataset identifier `so101_stack_recap_labeled_v21` and override the task instruction in the command. Each round starts from the fixed SFT seed `so101_run_1/29999`, not the previous adapter or `--resume`. Retain both the base weights and SFT seed. If changing the seed, update both training and inference configurations.

### 4.7 Deploy the Updated Policy

Terminal 1:

```bash
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101_recap_lora_infer \
  --policy.dir=checkpoints/pi05_so101_recap_lora_relabeled/toy_recap_round1/9999 \
  --default-prompt="Grab the toy into the box" --port=8000
```

Terminal 2: collect the next round in the SO-101 environment with the positive condition:

```bash
conda activate so101_hil_lite
python so101_hil_lite/record.py \
  --follower-port=/dev/ttyACM0 --follower-id=my_awesome_follower_arm \
  --leader-port=/dev/ttyACM1 --leader-id=my_awesome_leader_arm \
  --front-camera=0 --wrist-camera=2 --host=127.0.0.1 --port=8000 \
  --policy-id=toy_recap_round1_9999 --prompt="Grab the toy into the box" \
  --advantage=positive \
  --root=so101_hil_lite/data/toy_round2 --repo-id=local/so101_toy_round2 \
  --episodes=20 --duration=40 --fps=30
```

Do not intervene during autonomous evaluation; corrections are allowed during training collection. Deploy RECAP with `record.py --advantage=positive`; the standard/RTC client does not expose an advantage-condition argument.

### 4.8 Continue to the Next Round

Convert new data in the SO-101 environment, reusing first-round scales and retaining the historical pool:

```bash
python so101_hil_lite/convert_to_recap.py \
  so101_hil_lite/data/toy_round2 datasets/toy/prepared/round2 \
  --task-scales-json=datasets/toy/prepared/round1/meta/recap_manifest.json \
  --round-id=2 --validation-episodes 16 17 18 19

python so101_hil_lite/merge_recap_datasets.py merge \
  datasets/toy/pools/round1 datasets/toy/prepared/round2 \
  --output=datasets/toy/pools/round2
```

Repeat Sections 4.4–4.7 with round2 paths and experiment names, set `--threshold-round=2`, and recompute normalization statistics. Both the value model and policy reuse their respective fixed initializations. Pool1 already contains historical data; do not merge its constituent datasets again.
