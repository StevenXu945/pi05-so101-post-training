# Independent SigLIP + Gemma3-270M value runtime

Source: Destiny000621/RECAP commit a44350b4a99f1d7f6137635ab3e894a5f72f5bda,
Apache-2.0. `vendor/gemma` retains RECAP's compatibility patches on DeepMind
Gemma 3.3.0. Headers and RECAP_LICENSE preserve attribution.
Only value scripts add this vendor path; importing normal openpi does not import Gemma.

## Code layout

This directory contains the complete independent value-model workflow:

```text
value_runtime/
  train.py                 # primary training command
  label_advantage.py       # primary value/advantage/indicator inference command
  recap_value/
    config.py              # model configuration
    model.py               # SigLIP + Gemma and distributional loss
    observation.py         # input observation structure
    data.py                # video/state/task/value target loading
    weights.py             # strict checkpoint load/save
    train.py               # training and compiled validation implementation
    label_advantage.py     # labeling implementation
  vendor/gemma/            # upstream compatibility code
  setup_env.sh
  requirements.txt
  constraints-tested.txt
```

`src/openpi/recap/value_*.py` retains compatibility for old Python imports.
Run value training with `python value_runtime/train.py` and advantage labeling with
`python value_runtime/label_advantage.py`; the former scripts/ entries were removed.
Run both commands from the project root. The runtime configures its source and
vendored Gemma paths automatically; no package reinstall is required.
HIL code, data and environment remain in `so101_hil_lite/`.


Install in a separate environment, never into the root LoRA/RTC environment:

```bash
cd value_runtime
bash setup_env.sh
source .venv/bin/activate
cd ..
```

Conda alternative (run from the project root):

```bash
conda create -n openpi-recap-value python=3.11 -y
conda activate openpi-recap-value
python -m pip install -r value_runtime/requirements.txt -c value_runtime/constraints-tested.txt
python -m pip check
```

If the environment already exists, skip `conda create`. Do not use `--no-deps`.
The vendored gm/data package is included; if rebuilding the vendor tree, use the copy
block from setup_env.sh. All commands below run from the project root in this environment.

Download the public value-model bundle (a trained full VLM snapshot, not policy weights):

```bash
hf download ybpy/vlm_ckpt \
  --include 'gemma-3-270m/step_00020000/*' --include 'tokenizer.model' \
  --local-dir .openpi_cache/recap_value
```

Training requires the new meta/recap_manifest.json, fixed task scales and an episode-level validation split.
Use the cumulative pool described in ../docs/so101_recap.md; no policy warm-up stage is required.
Every round starts from the same initial value checkpoint below, not the last round output:

```bash
python value_runtime/train.py \
  --data=datasets/pools/round1 \
  --checkpoint=.openpi_cache/recap_value/gemma-3-270m/step_00020000 \
  --tokenizer=.openpi_cache/recap_value/tokenizer.model \
  --output=checkpoints/recap_value/stack_v1 \
  --batch-size=1 --steps=5000 --freeze-mode=all_backbones
```

This loads weights for further training; it does not resume the original optimizer.
The output stores trainable optimizer state for inspection, but optimizer-resume CLI
is not implemented. Default validation visits all held-out frames. For a short smoke
only, use `--steps=2 --val-interval=2 --save-interval=2 --validation-max-batches=1`.
Freezing all backbones still trains projection, cross-attention, norms and value head.
The model uses front/wrist images plus task and six-joint position text, unlike the
reference tokenizer which ignores state. It does not reproduce paper web-data co-training.

Relabel a new dataset copy:

```bash
python value_runtime/label_advantage.py \
  --data=datasets/pools/round1 \
  --checkpoint=checkpoints/recap_value/stack_v1/step_00005000 \
  --tokenizer=.openpi_cache/recap_value/tokenizer.model \
  --output=datasets/labeled/round1 \
  --threshold-round=1 --batch-size=2 --lookahead=50 --positive-ratio=0.4
```

`meta/advantage_predictions.parquet` stores V, advantage and indicators for auditing.
Task scales and dataset_version must match the value checkpoint. Terminal bootstrap is zero; training human frames
remain positive; held-out indicators remain none. Thresholds are fitted separately per task using only
whole intervention-free training rollouts from --threshold-round, and applied to cumulative training data.
Missing calibration data fails explicitly; validation predictions never set thresholds.
N is in output frames: 50 frames at 30 Hz is 1.67 s, not the paper's 50 Hz / 1 s.
No robot or existing checkpoint is modified by value scripts.

Validation uses one JIT-compiled forward pass for both cross-entropy and expected
value. The first batch for a new shape includes compilation. Validation progress
shows per-batch read and infer times; these include video decoding and synchronized
model execution respectively. Running processes keep the code they already loaded.
