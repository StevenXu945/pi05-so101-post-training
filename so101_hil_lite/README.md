# SO-101 lightweight human-in-loop collection

本目录使用官方 LeRobot，而非 Evo-RL 分支。固定版本：
`huggingface/lerobot@145cc7ea53e5864bd5efec4e9bb70ee799e47bd8`。
该版本要求 Python 3.12，并实现 SO leader 的 `send_feedback()`、
`enable_torque()`、`disable_torque()`。官方依赖仅安装到本目录 `.venv`。

## 安装

```bash
cd "/home/huayi/PycharmProjects/openpi (copy)/so101_hil_lite"
bash setup_env.sh
source .venv/bin/activate
```

需要 uv、可下载 Python 3.12 和依赖的网络，以及 Linux 终端。
安装脚本不修改项目根目录环境，也不修改已有 `so101_hil` 环境。
官方 LeRobot 仍依赖 PyTorch；轻量指采集代码和运行逻辑的范围，非零 ML 依赖。

## 终端 1：启动模型

在项目根目录使用原 openpi 环境：

```bash
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101_lora_finetune \
  --policy.dir=checkpoints/pi05_so101_lora_finetune/so101_run_1/29999 \
  --port=8000
```

配置名与 checkpoint 必须匹配；任务模型需要已经学过方块堆叠。

## 终端 2：读取观测与推理验证

在本目录激活新环境，替换设备路径、相机索引和已校准的 ID：

```bash
python record.py \
  --follower-port=/dev/ttyACM0 --follower-id=my_follower \
  --leader-port=/dev/ttyACM1 --leader-id=my_leader \
  --front-camera=0 --wrist-camera=2 \
  --policy-id=so101_run_1_29999 \
  --dry-run
```

dry-run 连接设备、读取观测和请求一次推理，但不发送关节目标。
两臂先用固定版本官方 LeRobot 校准；启动使用 `calibrate=False`，不隐式重标定。
旧版本校准格式不兼容时应重新校准，不要覆盖原环境校准文件。

## 采集

移除 `--dry-run`，指定全新数据目录：

```bash
python record.py \
  --follower-port=/dev/ttyACM0 --follower-id=my_follower \
  --leader-port=/dev/ttyACM1 --leader-id=my_leader \
  --front-camera=0 --wrist-camera=2 \
  --policy-id=so101_run_1_29999 \
  --root=data/stack_v1 --repo-id=local/so101_stack_v1 \
  --episodes=10 --duration=30 --fps=30
```

首次推理用于热身并打印目标。每轮先按 Enter 开启遥操作对齐：leader 释放扭矩，
follower 按步幅限制跟随 leader。拖动 leader 摆好初始姿态、摆放方块后，
再次按 Enter 检查对齐并切入策略。对齐阶段不写入数据。
自主时 leader 会主动跟随。前五关节使用度数，夹爪保持官方 0–100 标度。
操作键需在运行程序的终端输入，不依赖全局键盘监听：

* 自主时按 `i`：暂停策略，两臂保持位置，leader 仍有扭矩支撑。
* 握稳并托住 leader 后按 Enter：释放 leader 扭矩，开始人工控制。
* 人工控制时按 `i`：检查对齐、清缓存并交回策略。
* `s`：成功结束；`f`：失败结束。
* `q`：记录为 aborted 并退出。
* 超时后必须输入 success、failure 或 aborted，不默认判失败。

结束后 leader 释放扭矩，follower 保持当前姿态等待人工复位；没有自动回零。
交回策略时先写当前保持目标，再开启 leader 扭矩；位置差超限则终止该轮。
动作步幅默认前五关节 6.51、夹爪 20，可通过命令行调整。
策略或人工目标超出单步范围时，按上一发送目标及当前测量位置限幅，并打印
raw -> command 提示；原始目标保存在 selected_action，接受目标保存在 action。
若测量位置和上次发送目标偏离过大、无法同时满足限制，则停止并报 tracking gap。

推理采用同步请求，默认每个动作块使用前 8 步。请求期间不会发出下一步动作，
实际采集速率会受网络和推理延迟影响，不能保证恒定 30 Hz；请求超时默认 10 秒。
键盘事件在循环边界处理，因此推理等待期间的接管响应也会延迟。

## 数据与后续处理

LeRobot 保存图像、state 和 action；`action` 使用 follower.send_action 返回的
实际接受目标，不是实际测量到的下一时刻关节位置。
额外逐帧字段为 selected_action、policy_action、policy_action_valid、
is_intervention 和 elapsed_s。人工接管时 policy_action_valid=0，策略动作占位为零。
episode 结果和采集策略版本存放在数据集根目录 `outcomes.jsonl`。
这是自有 sidecar，不是 Evo-RL 的 episode_success 元数据；后续转换器必须读取它。
缺少 sidecar 记录的 episode 不能默认当作失败，应视作待检查数据。
aborted 轨迹应在训练转换时排除。暂停确认阶段以 is_paused=1 单独标记，
不计入 duration，但保留实际 elapsed_s；这些保持动作帧应在训练转换时检查处理。
数据路径已存在时拒绝覆盖，本版不支持续录。

## 阶段 3：校验与转换

工具与采集程序共用本目录环境，无需安装另一个 LeRobot 版本：

```bash
python check_dataset.py data/stack_v1 --report=stack_v1_report.json
python convert_to_recap.py data/stack_v1 data/stack_v1_recap_v21 \
  --fps=30 --task-max-steps=1201 --round-id=1 --validation-episodes 1
```

输出目录必须全新，原始数据不修改。转换器输出旧版 openpi 可读的 LeRobot
v2.1：每个 episode 的 parquet、重新编码并对齐的视频，以及 meta 目录。
它排除 aborted，去除暂停帧和暂停时长，以前一源帧保持的方式重采样图像、
状态和动作；终帧保留，source_frame_index 提供溯源。超过 0.5 秒的活动采样
空档默认拒绝；检查报告后可用 --max-gap-s 显式调整，不能恢复缺失的真实观测。

标签采用 paper-v1：非终帧 raw reward=-1，成功终帧=0，失败终帧=-C_fail。
每任务在训练 episode 上拟合固定归一化尺度，reward_label 为归一化奖励，
value_label 为其精确累计回报。人工帧 positive、自主帧 none。
用 --validation-episodes 指定源 episode 的验证集 ID；尺度和划分存入
meta/conversion.json。完整训练需显式留出验证 episode。
旧 remaining-time-v1 数据应重新转换，不直接用于新的价值训练。
详见 ../docs/so101_recap.md 和 ../value_runtime/README.md。

输出带 meta/recap_manifest.json，可由 merge_recap_datasets.py 合并演示和历轮 rollout。
当前 RECAP 配置读取 adv_ind；原 LoRA 配置保持原行为。后续轮次必须通过
--task-scales-json 复用首轮任务尺度。不要在价值训练前额外做条件策略热身。
将目标路径设到项目现有 HF_LEROBOT_HOME/repo_id 下，或按既有训练流程注册
该本地数据集；转换器本身不改训练配置和缓存位置。

完整累计迭代、v2.1 演示导入、阈值统计和运行命令见 ../docs/so101_recap.md。

## 代码分组

本目录集中真机采集、人工接管、数据检查、转换和合并。独立价值模型训练与优势
标注现统一在 ../value_runtime/，主入口为 train.py 和 label_advantage.py。
目录总览见 ../docs/code_layout.md。已有环境、校准、数据路径和录制命令不变。
