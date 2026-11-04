# 从当前 openpi 项目启动 RoboTwin 评测

本目录移植 RoboTwin/scripts/eval_openpi.py 与 eval_openpi.sh 的运行入口。
仍依赖原 RoboTwin checkout 的 envs、assets、env_cfg 和 description，以及现有
RoboTwin Conda 环境；不把 SAPIEN 等依赖安装进 openpi 或价值模型环境。
原 RoboTwin 下两个文件保留，未修改。

## 1. 策略服务

在 openpi 原环境启动现有 RoboTwin checkpoint 服务，使用与 checkpoint 匹配的
pi05_robotwin_50ep_lora_finetune 等配置。服务端必须返回 (16, 14) 动作，不能使用
返回六维动作的 SO-101 checkpoint。已有服务可以继续使用。

## 2. 校验配置（不启动仿真、不连接策略服务）

```bash
conda activate RoboTwin
cd "/home/huayi/PycharmProjects/openpi (copy)"
bash robotwin_eval/eval_openpi.sh \
  --robotwin-root="/home/huayi/PycharmProjects/RoboTwin" \
  --task-name=place_a2b_left --task-config=demo_clean --validate-only
```

默认 RoboTwin 位置为当前项目的同级 ../RoboTwin，可用 --robotwin-root 或
ROBOTWIN_ROOT 环境变量指定。OPENPI_ROOT 默认由本目录位置自动确定。

## 3. 仿真评测

确认策略服务已启动、任务与训练模型匹配后：

```bash
bash robotwin_eval/eval_openpi.sh \
  --robotwin-root="/home/huayi/PycharmProjects/RoboTwin" \
  --task-name=place_a2b_left --task-config=demo_clean \
  --host=127.0.0.1 --port=8000 --episodes=5 --exec-steps=8
```

视频默认开启，需要 ffmpeg；--no-video 可关闭。--expert-check 会在每轮先运行
专家检查。其余参数保持原脚本行为。task-name 请替换为实际评测任务。

结果默认保存到本项目：

```text
eval_result/robotwin_openpi/<task>/<config>/<timestamp>/
  summary.json
  episode_000.mp4
  ...
```

--output-dir 可指定其他路径；相对路径按启动时的工作目录解析。Python 子进程
会切到 RoboTwin 根目录以兼容仿真内部的相对资源路径，调用终端仍在当前项目。

也可以在已激活 RoboTwin 环境后直接运行：

```bash
python robotwin_eval/eval_openpi.py --task-name=place_a2b_left --episodes=5
```

本次只验证了真实配置读取、入口和模拟 rollout，未启动实际仿真评测。
