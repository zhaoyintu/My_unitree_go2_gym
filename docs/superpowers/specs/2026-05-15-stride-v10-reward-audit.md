# Lite3 Handstand Stride-V10 Reward Audit

Date: 2026-05-15

This is the final reward set inherited by
`Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V8`
(v10 in chat numbering). Each row is judged against the task goal:

> Front-paw handstand with a clean, no-hop, **alternating** stepping gait
> in response to velocity commands; quiet double-stance under zero command.

Legend: ✅ serves goal • ⚠️ subtle conflict / redundancy • ❌ counterproductive

## Pose maintenance

| # | term | weight | role | verdict |
|---|---|---|---|---|
| 1 | `alive` | +1.0 | Reward survival each frame. | ✅ Keep |
| 2 | `handstand_orientation` | -1.0 | Penalty: `(projected_gravity - [1,0,0])²` — drives body pitch +π/2 (head-down inverted). | ✅ Critical |
| 3 | `handstand_feet_on_air` | +5.0 | Reward rear feet (HL/HR) off ground. | ✅ Critical for handstand |
| 4 | `handstand_feet_height_exp` | +10.0 | L2-exp reward rear feet at world-z 0.56 m. | ✅ Critical |
| 5 | `base_height` | +0.8 | Soft reward trunk world-z = 0.39 m. | ✅ Keep |
| 6 | `base_contact` | -4.0 | Penalty trunk-ground collision. | ✅ Critical |
| 7 | `thigh_collision` | -2.0 | Penalty thigh-ground collision. | ✅ Keep |
| 8 | `calf_collision` | -2.0 | Penalty calf-ground collision. | ✅ Keep |

## Command tracking

| # | term | weight | role | verdict |
|---|---|---|---|---|
| 9 | `tracking_lin_vel` | +3.0 | Soft-gated by handstand quality; reward `exp(-vel_error²)`. | ✅ Primary motion driver |
| 10 | `tracking_ang_vel` | +1.5 | Same for yaw. | ✅ |
| 11 | `tracking_lin_vel_zero` | -0.4 | Penalty for moving when zero-command. | ✅ Keep |
| 12 | `tracking_ang_vel_zero` | -0.4 | Same for yaw. | ✅ Keep |

## Zero-command stance

| # | term | weight | role | verdict |
|---|---|---|---|---|
| 13 | `zero_stance_contact` | +3.0 | Reward both front feet planted when cmd=0. | ✅ Anti-hop while standing |
| 14 | `zero_joint_vel` | -0.08 | Penalty joint motion when cmd=0. | ✅ Keep |

## Gait shaping (NEW vs baseline)

| # | term | weight | role | verdict |
|---|---|---|---|---|
| 15 | `contact` | 0.0 | **Disabled**. Was `handstand_stance_contact_mean` (+2.0) which rewarded keeping BOTH feet down — directly opposed stepping. | ✅ Correctly disabled |
| 16 | `single_stance_contact` | +0.1 | Small reward `n_contact==1` when moving. Mild stepping incentive. | ✅ Keep |
| 17 | `feet_air_time` | +3.0 | Reward `(air_time - 0.4) * landing` per foot when moving. >0.4 s swings rewarded, shorter penalized. | ✅ Main stride driver |
| 18 | `feet_clearance` | +1.5 | Sinusoidal reward: each foot tracks `\|sin(2π·t/1.5)\|·0.15 m`, sharpness 20. | ✅ Enforces rhythm + amplitude |
| 19 | `stance_air_penalty` | -5.0 | Penalty both front feet airborne. Anti-hop, no command gate. | ✅ Critical |

## Smoothness / costs

| # | term | weight | role | verdict |
|---|---|---|---|---|
| 20 | `action_rate_l2` | -0.10 | L2 of action time-derivative. | ✅ Keep |
| 21 | `dof_acc` | -0.001 | L2 of joint acceleration. | ✅ Keep |
| 22 | `ang_vel_xy` | +0.2 | Reward low body-y/z angular velocity (non-yaw). | ✅ Keep |

## Suspicious — opposing stepping

| # | term | weight | role | verdict |
|---|---|---|---|---|
| 23 | **`symmetric_joints`** | **-0.1** | Penalty on left-right joint mismatch: `Σ\|FL_i - FR_i\|`. A true alternating gait is **inherently asymmetric** (one foot up while the other is down). This term mildly punishes exactly the behavior we want. | ❌ Should disable / flip sign during moving |
| 24 | **`lin_vel_z`** | **+0.6** | Reward `exp(-\|body_x_vel\|·10)`. In handstand the body's x-axis points down, so this rewards **low vertical motion**. A natural stride has some COM bob at swing/stance transitions; this term penalizes that. | ⚠️ Conservative; might be capping stride amplitude |
| 25 | **`default_pos`** | **-0.15** | L1 penalty on deviation from desired joint angles `(0, 0.1, 2.0)` for front / `(0, -0.8, 1.6)` for rear. During a real swing the front knee retracts and HipY rotates ±0.2 rad — every swing pays this penalty. | ⚠️ Low weight but persistent during every stride |
| 26 | `default_pos_reward` | +0.4 | Exp-reward `exp(-Σ\|joint - target\|)`. Symmetric counterpart of #25. Same concern. | ⚠️ Same |
| 27 | `default_hip_pos` | -0.1 | Penalty on HipX (abduction) deviation. Mostly orthogonal to stride direction. | ✅ Keep |

## Summary

**Likely blockers for visibly bigger strides:**

1. **`symmetric_joints` -0.1** — the most direct opposition. Alternating
   gait creates per-step asymmetry of ~0.3-0.5 rad in the swinging leg vs
   the planted leg; this term continuously punishes that.

2. **`default_pos` -0.15 / `default_pos_reward` +0.4** — together push the
   policy back to the static handstand pose. Combined effect during a
   swing peak (HipY deviates 0.2 rad, Knee deviates 0.3 rad): ~-0.07
   penalty + missing ~+0.2 reward per step — bigger than the
   feet_clearance gain at low tracking quality.

3. **`lin_vel_z` +0.6** — A real "迈腿" gait has small COM bob; suppressing
   it caps the body dynamics that make strides look deliberate.

**Recommended v11 changes:**

- Disable `symmetric_joints` (weight 0.0) — alternating gait should not
  be penalized.
- Drop `default_pos` to -0.05 and `default_pos_reward` to 0.15 (further
  reduce the static-pose pull while moving).
- Drop `lin_vel_z` from 0.6 to 0.2 (still discourage hopping flight but
  allow stride bob).

These changes target the rewards that *fight against* the stepping
behaviour, rather than adding yet more rewards that try to *force* it.
