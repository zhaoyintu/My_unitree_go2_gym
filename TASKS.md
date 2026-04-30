# Go2 任务说明:输入输出 / 奖励 / Sim-to-Real

本文档对 `legged_gym/envs/__init__.py` 中注册的 7 个 Unitree Go2 强化学习任务做系统说明,涵盖:

- 共同基础(基类层面的观测、动作、奖励)
- 每个任务的差异(obs/action/reward/课程)
- Sim-to-Real Gap 的处理手段

> 引用格式 `path:line` 指向源码,便于跳转。

---

## 一、共同基础(基类)

### 1.1 观测(非对称 PPO)

来源:`legged_gym/envs/base/legged_robot.py:232-290`

- **Actor obs(机载可获取)**:命令(sin/cos 步态相位 + 速度命令) + IMU(角速度 + 欧拉角) + 12 关节位置/速度 + 上一动作 ≈ **47 维/帧**,堆叠 **10 帧** → 470 维
- **Critic obs(特权信息)**:在 actor obs 基础上额外提供基座线速度、绝对位置、接触状态、PD 参数等 ≈ **70 维**
- **观测噪声**:加性高斯噪声(scale 由 `noise_scales` 控制)
  - `dof_pos: 0.01`、`dof_vel: 1.5`、`ang_vel: 0.2`、`quat: 0.1`
  - `noise_level` 全局缩放(默认 1.0)

非对称 PPO:Actor 仅看可在实机获取的量,Critic 看完整状态。这迫使策略学到对噪声鲁棒的表征,同时保留 Critic 的优势估计精度。

### 1.2 动作

来源:`legged_gym/envs/base/legged_robot_config.py:90-98`

- **12 维关节 PD 目标**:`q_target = action * action_scale + q_default`
- **基类默认**:`action_scale=0.5, P=10, D=1`(各任务覆盖,见下表)
- 力矩通过 PD 计算后由 URDF 力矩限制裁剪

### 1.3 通用奖励项

| 类型 | 项 | 含义 |
|---|---|---|
| 正向 | `tracking_lin_vel`、`tracking_ang_vel` | 速度命令跟踪(指数衰减) |
| 惩罚 | `lin_vel_z`、`ang_vel_xy` | 不期望的竖直/翻滚速度 |
| 惩罚 | `orientation` | 基座倾斜 |
| 惩罚 | `base_height` | 偏离目标基座高度 |
| 惩罚 | `torques`、`dof_acc`、`action_rate` | 控制平滑性 |
| 惩罚 | `collision` | 非脚部碰撞 |

具体奖励权重各任务自行覆盖。

---

## 二、各任务差异

| Task | obs 堆叠 | action_scale / PD | 关键特异奖励 | episode | 地形 |
|---|---|---|---|---|---|
| **go2_trot** | 10 帧 actor + 3 帧 critic | 0.25 / P=20, D=0.5 | `trot=0.8`(对角同步)、`feet_clearance=0.1` 目标 0.06m、`tracking_lin_vel=2.0`(仅 trot>0.7 时)、`base_height=-5.0` @0.29m | 24s | 平地,命令 curriculum |
| **go2_stairs** | 单帧(无堆叠) | 同 trot,critic 含 **17×11=187 维高度图** | trot 奖励 + 高度图驱动落足 | 20s | trimesh 阶梯 0.7 占比,terrain curriculum |
| **go2_jump** | 10 帧 | 0.25 / P=20, D=0.5 | `jump=2.0`(四足同时离地)、`feet_air_time=1.0`(>0.5s 起算)、`orientation=0.6`、`base_height=1.0` @0.3m | — | 平地,无 curriculum;摩擦窄到 [0.4, 0.8] |
| **go2_handstand** | 单帧,critic 89 维 | 0.2 / P=40, D=0.7 | 后腿目标关节角 `thigh=2.25, calf=-1.75`(倒立姿态);命令窗口 [-0.4, 0.4](近静止) | 20s | 平地 |
| **go2_leggedstand** | 单帧,critic 89 维 | 同 handstand | 目标关节角为正常站姿 `thigh=0.8, calf=-1.5`,侧重 `base_height` / `orientation` 稳定 | 20s | 平地 |
| **go2_spring_jump** | 10 帧 + 3 帧 critic | 同 trot | 蓄力—释放相位匹配 | 短 episode | 平地 |
| **go2_backflip** | 10 帧 + 3 帧 critic | 同 trot,`reset_height=0.1m` | 旋转对称(基座欧拉角变化率)、自扶正(着地恢复) | **4s** | 平地 |

### 文件位置

- `go2_trot`     → `legged_gym/envs/Go2_MoB/GO2_Trot/GO2_Trot.py` + `GO2_Trot_config.py`
- `go2_stairs`   → `legged_gym/envs/Go2_MoB/GO2_Trot/GO2_Stairs.py` + `GO2_Stairs_config.py`
- `go2_jump`     → `legged_gym/envs/Go2_MoB/GO2_JUMP/go2_jump_env.py` + `GO2_JUMP_config.py`
- `go2_handstand`→ `legged_gym/envs/GO2_Stand/GO2_Handstand/Go2_handstand.py` + `Go2_handstand_Config.py`
- `go2_leggedstand` → `legged_gym/envs/GO2_Stand/GO2_Leggedstand/Go2_legstand.py` + `Go2_legstand_Config.py`
- `go2_spring_jump` → `legged_gym/envs/GO2_Flip/GO2_Spring_Jump/GO2_Spring_Jump_env.py` + `GO2_Spring_Jump_Config.py`
- `go2_backflip` → `legged_gym/envs/GO2_Flip/GO2_BackFlip/GO2_BackFlip_env.py` + `GO2_BackFlip_Config.py`

---

## 三、Sim-to-Real Gap 处理

参考模板:`legged_gym/envs/Go2_MoB/GO2_Trot/GO2_Trot_config.py:105-141`,其它任务在此基础上调参。

### A. 域随机化(Domain Randomization)

| 类别 | 参数范围 | 目的 |
|---|---|---|
| 基座质量 | ±1 kg | 载重/电池差异 |
| 连杆质量 | 0.9–1.1× | 制造公差 |
| 质心位置 | ±0.03 m | 装配偏差 |
| 摩擦系数 | 0.2–1.2(jump 收紧到 0.4–0.8) | 地面材质变化 |
| PD 增益 | ±10% | 电机个体差异、老化 |
| 电机零点偏置 | ±0.035 rad | 关节零点漂移 |
| 计算力矩乘数 | 随机缩放 | 电机非线性 |
| 推力扰动 | 每 4s 一次,xy 0.4 m/s,yaw 0.6 rad/s | 抗碰撞/抗滑动 |

### B. 时延建模

`legged_gym/envs/base/legged_robot.py:122-145`

- **Action 延迟**:1–3 仿真步(@1 kHz ≈ 1–3 ms),每个 episode 重新采样
- **Observation 延迟**:**分通道独立采样**
  - Motor 通道(关节位置/速度):1–3 步
  - IMU 通道(角速度/姿态):1–3 步

这里把"传感器并不同步"也建模进去了,比单一全局延迟更真实。

### C. 观测噪声 + 非对称 PPO

- 机载可得量(IMU、关节位置/速度)加高斯噪声
- 特权量(基座线速度、接触状态、PD 参数)只暴露给 Critic
- 强迫 Actor 在仅有噪声机载观测的条件下输出动作,提升 zero-shot 鲁棒性

### D. 课程学习

`legged_gym/envs/base/legged_robot.py:525-545`

- **地形 curriculum(stairs)**:平地 → 斜坡 → 阶梯,根据成功率推进 `terrain_level`
- **命令 curriculum(trot / jump)**:速度上限 `max_curriculum` 渐增
- **Episode 长度差异**:翻转类任务 4 s,行走类 24 s,匹配各自动作时间尺度

### E. 部署侧验证

目录:`deploy_mujoco_viewer/`

- **Sim2Sim**:在 IsaacGym 训练完成后,用 MuJoCo 重放同一策略
- 状态提取(关节、IMU、足端接触)与训练时观测严格对齐
- 用作部署到实机前的最后一道隔离层,排除"策略只在某个仿真器有效"的失效模式

---

## 四、综合策略

```
[训练]                          [部署]
IsaacGym + DR + 噪声 + 延迟  ──► MuJoCo 验证 ──► 实机
  │                               │
  └─ 非对称 PPO(privileged critic)
  └─ 课程学习(地形 / 命令)
  └─ 任务级 reward shaping
```

核心打法:**非对称 PPO + 多维域随机化(质量 / 摩擦 / PD / 延迟 / 扰动) + MuJoCo sim2sim 验证**。各任务通过裁剪命令窗口、调整 `action_scale` 与 PD 增益、改 reward shaping,从行走 → 站立 → 翻腾覆盖了不同动力学需求。
