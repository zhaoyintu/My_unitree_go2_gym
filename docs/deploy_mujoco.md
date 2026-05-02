# MuJoCo 部署 / Play 指南

## Lite3 handstand 快速测试

`scripts/deploy_lite3_handstand.py` 会默认查找最新训练日志：

```text
logs/rsl_rl/lite3_handstand/<latest_run>/model_*.pt
```

然后调用 `deploy_mujoco_viewer/deploy_mjlab_lite3_handstand.py` 在纯 MuJoCo
里运行策略，不需要 `mjlab`。

### 自动加载最新 checkpoint

```bash
cd /home/yiz/workspace/src/locomotion/My_unitree_go2_gym
python scripts/deploy_lite3_handstand.py
```

### 指定速度命令

```bash
python scripts/deploy_lite3_handstand.py --cmd 0.2 0.0 0.0
```

### Headless smoke test

```bash
python scripts/deploy_lite3_handstand.py --no-viewer --duration 20
```

### 录制视频

```bash
python scripts/deploy_lite3_handstand.py \
  --video outputs/lite3_handstand.mp4 \
  --duration 30
```

### 指定 checkpoint

```bash
python scripts/deploy_lite3_handstand.py \
  --policy logs/rsl_rl/lite3_handstand/<run>/model_14999.pt
```

也可以指定 run 和 checkpoint 文件名：

```bash
python scripts/deploy_lite3_handstand.py \
  --run 2026-05-02_12-30-00 \
  --ckpt model_5000.pt
```

先确认自动选择的命令但不启动仿真：

```bash
python scripts/deploy_lite3_handstand.py --dry-run
```

### Robust 对照任务

原始任务保持不变：

```bash
python scripts/train_go2.py Mjlab-Lite3-Handstand
```

新增 robust 对照任务会写到独立 experiment：

```bash
python scripts/train_go2.py Mjlab-Lite3-Handstand-Robust
```

对照 deploy 时指定对应日志根目录：

```bash
python scripts/deploy_lite3_handstand.py \
  --log-root logs/rsl_rl/lite3_handstand_robust \
  --no-viewer \
  --duration 20
```

---

## 方式一：mjlab play（使用 mjlab 环境回放）

mjlab 自带 play 脚本，通过导入 `go2_mjlab` 注册任务后即可使用。

### 1. Zero agent（测试环境）

```bash
cd /home/yiz/workspace/src/locomotion/My_unitree_go2_gym
python -c "import go2_mjlab; from mjlab.scripts.play import main; main()" \
  Mjlab-Lite3-Handstand --agent zero --num-envs 1 --no-terminations
```

### 2. 加载本地 checkpoint

```bash
python -c "import go2_mjlab; from mjlab.scripts.play import main; main()" \
  Mjlab-Lite3-Handstand \
  --checkpoint-file logs/rsl_rl/lite3_handstand/<run>/model_14999.pt \
  --num-envs 1
```

### 3. 从 wandb 加载

```bash
python -c "import go2_mjlab; from mjlab.scripts.play import main; main()" \
  Mjlab-Lite3-Handstand \
  --wandb-run-path <entity>/<project>/<run_id> \
  --num-envs 1
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `--agent {zero,random,trained}` | zero=无动作, random=随机, trained=加载模型 |
| `--num-envs N` | 并行环境数，默认全部；1 为单环境 |
| `--viewer {native,viser,auto}` | native=MuJoCo原生, viser=Web, auto=自动 |
| `--no-terminations` | 禁用终止条件，方便观察 |
| `--video` | 录制视频 |
| `--video-length N` | 视频帧数，默认 200 |
| `--wandb-checkpoint-name model_4000.pt` | 加载指定 checkpoint |

---

## 方式二：IsaacGym sim2sim viewer（使用旧 IsaacGym 模型）

位于 `deploy_mujoco_viewer/sim2sim_handstand_viewer.py`，加载 IsaacGym 训练的 torchscript 模型。

```bash
cd /home/lionrock/workspace/locomotion/My_unitree_go2_gym
python deploy_mujoco_viewer/sim2sim_handstand_viewer.py \
  --load_model logs/go2_handstand/exported/policies/policy_1.pt
```

默认通过键盘控制速度指令：
- `6/7` — x 方向前进/后退
- `8/9` — y 方向左/右
- `- =` — yaw 旋转
- `1` — 复位零指令

---

## 可用的 Go2 任务

| 任务名 | 说明 |
|--------|------|
| `Mjlab-Go2-Handstand` | 手倒立（前腿支撑，身体倒立） |
| `Mjlab-Go2-Trot-Flat` | 平地 trot 步态 |
| `Mjlab-Go2-Stairs` | 楼梯地形 |
| `Mjlab-Go2-Jump` | 跳跃步态 |
| `Mjlab-Go2-Leggedstand` | 后腿站立 |
| `Mjlab-Go2-Spring-Jump` | 弹簧跳跃 |
| `Mjlab-Go2-Backflip` | 后空翻 |

替换命令中的任务名即可部署不同任务。
