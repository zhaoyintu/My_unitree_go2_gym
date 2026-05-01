# MuJoCo 部署 / Play 指南

## 方式一：mjlab play（推荐，使用 mjlab 训练模型）

mjlab 自带 play 脚本，通过导入 `go2_mjlab` 注册任务后即可使用。

### 1. Zero agent（测试环境）

```bash
cd /home/lionrock/workspace/locomotion/My_unitree_go2_gym
python -c "import go2_mjlab; from mjlab.scripts.play import main; main()" \
  Mjlab-Go2-Handstand --agent zero --num-envs 1 --no-terminations
```

### 2. 加载本地 checkpoint

```bash
python -c "import go2_mjlab; from mjlab.scripts.play import main; main()" \
  Mjlab-Go2-Handstand \
  --checkpoint-file wandb/run-20260501_065244-u766girr/files/model_14999.pt \
  --num-envs 1
```

### 3. 从 wandb 加载

```bash
python -c "import go2_mjlab; from mjlab.scripts.play import main; main()" \
  Mjlab-Go2-Handstand \
  --wandb-run-path joystudy3/mjlab/u766girr \
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
