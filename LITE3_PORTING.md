# Lite3 移植到本仓库 — 参数对比与实施报告

> 目标:把仓库 `go2_rl_gym-recovery` 中已经成熟的 Lite3 模型(参考任务 `lite3_moe_cts_vel_est`)迁移到当前仓库 `My_unitree_go2_gym`,并仿照已有的 `go2_handstand` / `go2_leggedstand` 完成 `lite3_handstand` / `lite3_leggedstand` 两个任务。

- 仓库 A(目标):`/home/yiz/workspace/src/locomotion/My_unitree_go2_gym`
- 仓库 B(参考):`/home/yiz/workspace/src/locomotion/go2_rl_gym-recovery`

---

## 一、Lite3 与 Go2 本体差异

| 参数 | Lite3 | Go2 | 影响 |
|---|---|---|---|
| 总质量 | ~10.4 kg | ~13.6 kg | Lite3 轻 ~25%,惯性小、动作更快 |
| 躯干质量 | 5.61 kg | 6.92 kg | — |
| 单腿质量 | ~1.55 kg | ~2.03 kg | — |
| 关节数 / DoF | 12 (3×4) | 12 (3×4) | 相同 |
| 标称站立高度 | 0.35 m | 0.29 m | Lite3 重心更高,倒立更敏感 |

**资源路径**:
- Lite3 URDF:`go2_rl_gym-recovery/resources/robots/lite3/urdf/Lite3.urdf`
- Lite3 MJCF:`go2_rl_gym-recovery/resources/robots/lite3/lite3.xml`
- Lite3 mesh:`go2_rl_gym-recovery/resources/robots/lite3/meshes/`

**关节限位**(Lite3.urdf):

| 关节 | pos_min | pos_max | τ_max | ω_max |
|---|---|---|---|---|
| HipX | -0.523 | 0.523 | 24 N·m | 26.2 rad/s |
| HipY | -2.67 | 0.314 | 24 N·m | 26.2 rad/s |
| Knee | 0.524 | 2.792 | 36 N·m | 17.3 rad/s |

**默认关节角差异**:
- Lite3:`HipY=-0.8, Knee=1.6`(`lite3_config.py:7-24`)
- Go2:`thigh=0.8, calf=-1.5`(`GO2_Trot_config.py:58-73`)
- 注意 sign 约定不同(URDF 朝向),不能盲目复用 Go2 的 `desire_joint_angles`。

---

## 二、关键参数对比表

### 2.1 控制 / 网络

| 项 | Lite3 (`lite3_moe_cts_vel_est`) | Go2 Trot | Go2 Handstand | Go2 Leggedstand |
|---|---|---|---|---|
| Kp(全关节) | **30** | 20 | **40** | **40** |
| Kd | **1.0** | 0.5 | 1.0 | 1.0 |
| action_scale | 0.25 | 0.25 | **0.20** | 0.20 |
| **hip_action_scale** | **0.125** | — | — | — |
| dt / decimation | 0.004 / 4 (250 Hz) | 0.004 / 4 | 0.004 / 4 | 0.004 / 4 |
| 单帧 obs 维度 | 45 | 47 | 48 | 48 |
| frame_stack(actor) | **1** | 10 | 1 | 1 |
| frame_stack(critic) | 1 | 3 | 1 | 1 |
| 特权 obs 维度 | **263**(含 187 维高度图) | 70(×3) | 89 | 89 |

### 2.2 域随机化

| 项 | Lite3 | Go2 Trot | Go2 Handstand |
|---|---|---|---|
| 摩擦范围 | **[0.0, 2.0]**(更激进) | [0.2, 1.2] | [0.2, 0.8] |
| 基座质量 | ±0.5 kg | ±1 kg | ±1 kg |
| 连杆质量倍数 | ×[0.9, 1.1] | ×[0.9, 1.1] | ×[0.9, 1.1] |
| COM 偏移 | ±0.03 m | ±0.03 m | ±0.03 m |
| PD 增益 | ×[0.9, 1.1] | ×[0.9, 1.1] | ×[0.9, 1.1] |
| 电机零点 | ±0.035 rad | ±0.035 rad | ±0.035 rad |
| **电机强度倍数** | ×[0.8, 1.2] | — | — |
| Push 间隔 / 强度 | 4 s / 0.4 m/s, 0.6 rad/s | 4 s / 0.4, 0.6 | 8 s / 1.0, 1.0 |
| Action 延迟 | 1–3 步 | 1–3 步 | (基本不加) |
| Obs 延迟(motor / IMU) | 1–3 步 / 1–3 步 | 1–3 / 1–3 | (基本不加) |

观测噪声 `noise_scales` 两个仓库**几乎一致**(`dof_pos:0.01, dof_vel:1.5, ang_vel:0.2, quat:0.1, gravity:0.05, height:0.1`)。

### 2.3 地形 / 课程 / Episode

| 项 | Lite3 | Go2 Trot | Go2 Handstand |
|---|---|---|---|
| 地形 | 9 类(wave/slope/stairs/obstacles/…) | plane | plane |
| terrain curriculum | False | True | False |
| episode 长度 | 25 s | 24 s | 20 s |

### 2.4 奖励函数侧重

Lite3(`lite3_config.py:156-174`)主要项:
`tracking_lin_vel=1.0, tracking_ang_vel=0.5, lin_vel_z=-2.0, ang_vel_xy=-0.05, dof_acc=-2.5e-7, dof_power=-2e-5, torques=-1e-4, correct_base_height=-1.0, action_rate=-0.01, action_smoothness=-0.01, collision=-1.0, dof_pos_limits=-2.0, feet_air_time=1.0, feet_gait=0.2, feet_air_time_variance=-2.0, hip_to_default=-0.05`。

Go2 倒立(`Go2_handstand_Config.py`)关键差异:速度跟踪权重置零、强化 `base_height` 与目标关节角(`desire_joint_angles`)。

---

## 三、`lite3_moe_cts_vel_est` 模型机制

> 仓库 B 的这个任务用的是 `Lite3Robot + Lite3Cfg + Lite3CfgMoECTSVelEst`,网络是 `ActorCriticMoECTSVelEst`(`go2_rl_gym-recovery/rsl_rl/rsl_rl/modules/actor_critic_moe_cts_vel_est.py`)。

### 3.1 三个组件

- **MoE(Mixture of Experts)**:8 个并行 expert(`expert_num=8`),通过学习的 gating 把历史观测路由到不同专家,输出 32 维 latent。多模态行为(走 / 跌倒恢复)自然分化到不同专家。
- **CTS(Collaborative Teacher–Student)**:Teacher 用完整特权观测(基座线速度、接触、力矩、高度图…)直接生成 32 维 latent;Student(MoE)只用历史原始观测推断 latent,通过 MSE 与 Teacher 对齐。Algorithm 类是 `CTS`,policy 是 `ActorCriticCTS`。
- **VelEst(velocity estimator head)**:一个 `MLP[64,64]→3` 的小网络,从 latent 估计基座线速度 `v̂`。`v̂` 拼到 actor 输入(`[latent ⊕ obs ⊕ v̂]`),但**梯度不传回 PPO**(`.detach()`),仅由独立 MSE aux loss 监督。

### 3.2 对 sim-to-real 的意义

- MoE 让单策略覆盖更多模态,减少任务切换网络开销
- CTS 把"特权信息只在训练用,部署只用机载观测"做到端到端,比简单 asymmetric PPO 更彻底
- VelEst 给 actor 显式速度信号,部署时可用机载里程计 / 滤波器替换

### 3.3 当前仓库 A 的状况

仓库 A 只支持基础 PPO(`OnPolicyRunner` + `ActorCritic`),**没有** MoE / CTS / VelEst。移植倒立任务有两条路:

- **路线 1(推荐先做)**:仅迁移 Lite3 模型 + 基础 PPO,先把 `lite3_handstand/legstand` 做出来。`lite3_moe_cts_vel_est` 的网络结构**先不要带过来**,因为它依赖仓库 B 的整套 rsl_rl 升级。
- **路线 2(后续)**:如果倒立调不出,再把 rsl_rl 的 `actor_critic_moe_cts_vel_est.py` + `cts.py` 算法迁过来,同时改 `task_registry` 的 runner 选择逻辑。

---

## 四、移植工作清单

### 4.1 资源迁移

```bash
cp -r go2_rl_gym-recovery/resources/robots/lite3 \
      My_unitree_go2_gym/resources/robots/lite3
```

检查 `lite3.xml` / `Lite3.urdf` 内 `meshes/` 引用是相对路径(已是 `../meshes/`)。`setup.py` 的 package_data 若没有 `resources/*`,要确认 import 时能找到。

### 4.2 代码迁移

复制以下到仓库 A:

| 源 | 目标 |
|---|---|
| `go2_rl_gym-recovery/legged_gym/envs/lite3/lite3_env.py` | `My_unitree_go2_gym/legged_gym/envs/lite3/lite3_env.py` |
| `go2_rl_gym-recovery/legged_gym/envs/lite3/lite3_config.py` | `My_unitree_go2_gym/legged_gym/envs/lite3/lite3_config.py` |
| `go2_rl_gym-recovery/legged_gym/envs/lite3/__init__.py` | `My_unitree_go2_gym/legged_gym/envs/lite3/__init__.py` |

修改点:

1. `lite3_config.py:131` 的 URDF 路径改成本仓库的 `LEGGED_GYM_ROOT_DIR`
2. 移除/裁掉本仓库**没有的依赖**:
   - 各种 recovery 相关配置类(只保留 `Lite3Cfg` + `Lite3CfgPPO`,删 `Lite3CfgMoECTS / VelEst` 及其 import)
   - `lite3_env.py` 中如果引用了 base 层在仓库 A 不存在的 hook(见下),需要打补丁

### 4.3 base 层兼容性

仓库 B 的 `legged_robot.py` 比仓库 A **新增**:

- 跌倒初始化 `turn_over`
- 动态命令重采样 `dynamic_resample_commands`
- 命令 / 地形 curriculum 改进(`zero_command_curriculum`、`move_down_by_accumulated_xy_command`)
- 域随机化新增 `randomize_motor_strength`、`randomize_action_delay`(仓库 A 已有 motor_zero_offset / calculated_torque,但字段名可能不同)

**建议**:**不升级 base**,在 `lite3_env.py` / `lite3_config.py` 中:
- 把 Lite3 用不到的字段直接删掉
- 把 Lite3 用到但仓库 A base 没有的 hook,在 `Lite3Robot` 里 override 实现(避免改 base 影响现有 7 个 go2 任务)

升级 base 的代价是要重跑 7 个 go2 任务的回归验证,不划算。

### 4.4 实现 `lite3_handstand` / `lite3_leggedstand`

目录结构(对齐 `GO2_Stand/`):

```
legged_gym/envs/Lite3_Stand/
├── Lite3_Handstand/
│   ├── Lite3_handstand.py            # class Lite3_stand(Lite3Robot)
│   └── Lite3_handstand_Config.py     # Lite3Cfg_Handstand + Lite3CfgPPO_Handstand
└── Lite3_Leggedstand/
    ├── Lite3_legstand.py
    └── Lite3_legstand_Config.py
```

#### `Lite3Cfg_Handstand` 关键修改点

```python
class Lite3Cfg_Handstand(Lite3Cfg):
    class init_state(Lite3Cfg.init_state):
        pos = [0.0, 0.0, 0.35]
        # default 维持站立姿态
        # 新增 desire_joint_angles 作为倒立目标(参考 Go2 思路)
        desire_joint_angles = {
            # 后腿伸直支撑 — 需要对 Lite3 标定,初值参考下
            'HL_HipY_joint':  2.0,  'HL_Knee_joint':  2.5,
            'HR_HipY_joint':  2.0,  'HR_Knee_joint':  2.5,
            'FL_HipY_joint': -0.6,  'FL_Knee_joint':  1.4,
            'FR_HipY_joint': -0.6,  'FR_Knee_joint':  1.4,
        }

    class control(Lite3Cfg.control):
        stiffness = {'joint': 40.0}   # 参考 Go2 倒立
        damping   = {'joint': 1.0}
        action_scale = 0.20
        hip_action_scale = 0.10        # HipX 缩窄,倒立平衡用

    class terrain:
        mesh_type = 'plane'
        curriculum = False

    class rewards(Lite3Cfg.rewards):
        base_height_target = 0.50      # 倒立时目标高度 ≈ 腿长
        class scales:
            tracking_lin_vel    = 0.0
            tracking_ang_vel    = 0.0
            correct_base_height = 5.0  # 倒立姿态权重升高
            orientation         = 1.0
            dof_pos_limits      = -2.0
            torques             = -1e-4
            action_rate         = -0.01
            action_smoothness   = -0.01
            hip_to_default      = -0.1
```

#### 倒立目标姿态怎么标

Go2 的倒立姿态是:`thigh=2.25, calf=-1.75`(后腿伸直支撑)。Lite3 角度约定不同,**不能直接套**。建议:

1. 把 Lite3 的 URDF 在 MuJoCo viewer 里用 `desire_joint_angles` 试不同值,目视判断后腿是否伸直、前腿离地、基座竖起
2. 检查支撑多边形(后两脚连线长度)和质心投影
3. 写一个 `tools/lite3_pose_check.py`:加载 URDF,给定关节角,打印质心高度 / 倒立角度 / 关节是否超限

#### PD 调优经验

| 现象 | 调整 |
|---|---|
| 倒立缓慢偏移 | Kp 30→40,Kd 不变 |
| 倒立震荡 | Kp 不变,Kd 1.0→1.2 |
| 控制过激 | action_scale 0.25→0.20,或 hip_action_scale 0.125→0.10 |
| 关节过扭 | 增加 `torques` / `dof_pos_limits` 惩罚 |

### 4.5 注册 + VSCode

`legged_gym/envs/__init__.py` 末尾新增:

```python
from legged_gym.envs.lite3.lite3_env import Lite3Robot
from legged_gym.envs.lite3.lite3_config import Lite3Cfg, Lite3CfgPPO
from legged_gym.envs.Lite3_Stand.Lite3_Handstand.Lite3_handstand import Lite3_stand
from legged_gym.envs.Lite3_Stand.Lite3_Handstand.Lite3_handstand_Config import Lite3Cfg_Handstand, Lite3CfgPPO_Handstand
from legged_gym.envs.Lite3_Stand.Lite3_Leggedstand.Lite3_legstand import Lite3_legstand
from legged_gym.envs.Lite3_Stand.Lite3_Leggedstand.Lite3_legstand_Config import Lite3Cfg_Leggedstand, Lite3CfgPPO_Leggedstand

task_registry.register("lite3",            Lite3Robot,    Lite3Cfg(),            Lite3CfgPPO())
task_registry.register("lite3_handstand",  Lite3_stand,   Lite3Cfg_Handstand(),  Lite3CfgPPO_Handstand())
task_registry.register("lite3_leggedstand",Lite3_legstand,Lite3Cfg_Leggedstand(),Lite3CfgPPO_Leggedstand())
```

`.vscode/tasks.json` 的 `pickString.options` 加上 `lite3, lite3_handstand, lite3_leggedstand`,Train / Play 任务自动复用。

### 4.6 验证步骤

1. **基础站立**:`python legged_gym/scripts/train.py --task=lite3 --headless`,跑几千 iter,Play 看零命令下能否站稳
2. **倒立**:`python legged_gym/scripts/train.py --task=lite3_handstand --headless --num_envs=2048 --max_iterations=5000`,小规模看 reward 趋势
3. **扩规模**:4096 envs,50k iter
4. **MuJoCo sim2sim**:用 `deploy_mujoco_viewer/` 的脚本切到 Lite3 MJCF,跑 actor.pt 验证

成功标准:零命令下能稳定倒立 ≥ 20 s;轻推扰动可恢复。

---

## 五、风险与建议

### 5.1 物理可行性
Lite3 质量轻 + 站立高(0.35 m)→ 倒立时质心更高、支撑面积小、惯性小,**比 Go2 倒立更难,不是更易**。建议:
- 不追求完全直腿倒立,膝盖留些弯曲(Knee ≈ 2.4 rad)以降低重心
- 如果纯倒立(`handstand`)调不出,先做 `leggedstand`(后腿站起、前腿悬空),物理上更容易

### 5.2 Base 层迁移红线
不要为了多带几个 Lite3 的小特性就换掉仓库 A 的 base — 7 个 go2 任务的 reward 与 reset 逻辑都依赖现有 base 的细节,换了要全部回归。**先在 Lite3 子类里 override**。

### 5.3 是否要带 MoE / CTS / VelEst
**先不要**。这些机制在仓库 B 是为了 fall recovery 多模态行为设计的,倒立 / 后腿站是单一模态任务,基础 PPO 足够。等基础任务跑通后再决定是否引入。

### 5.4 常见踩坑

| 坑 | 表现 | 处理 |
|---|---|---|
| URDF 角度约定 | 看似对的目标姿态导致后腿插地 | URDF viewer 验证 + 标定 |
| PD 过软 | 倒立慢慢倒下 | Kp 30→40 |
| PD 过硬 | 控制爆抖、电流大 | Kd 提到 1.2,或缩 action_scale |
| reward 不平衡 | 学到只压低基座或只稳关节 | 把 scale 都缩到 ~1.0 量级再调 |
| Domain rand 太猛 | 早期完全学不动 | 先关 push、缩摩擦范围,跑通后再开 |

---

## 六、推荐阶段计划

| 阶段 | 内容 | 预计 |
|---|---|---|
| Phase 1 | 资源 + lite3 base 任务跑通 | 1–2 天 |
| Phase 2 | 倒立姿态标定 + 小规模训练 | 3–5 天 |
| Phase 3 | 扩规模训练 + 调参 | 5–10 天 |
| Phase 4(可选) | 实现 `lite3_leggedstand` | 2–3 天 |
| Phase 5(可选) | MuJoCo sim2sim 与导出 | 2–3 天 |

总计 **10–15 人·天**,大头在倒立姿态标定 + reward / PD 调参的迭代。
