# Lite3 Handstand No-Hop Reward Variant

## Goal

Eliminate the hopping gait that emerges from the `QuietZeroStance` baseline:

- The trained policy currently makes small two-foot hops while keeping the
  command at zero, instead of staying planted.
- When given a forward/backward command it hops along, again leaving both
  front (stance) feet airborne at the same time, rather than stepping with
  one foot at a time.

Target behavior:

- **Zero command**: keep at least one front foot in ground contact at all
  times. Small back-and-forth shuffling steps are acceptable; ballistic
  hops are not.
- **Non-zero command**: locomote by alternating steps. Same hard constraint:
  the two front (stance) feet must never be airborne simultaneously.

Out of scope: forcing strict single-stance alternation, swing-arc shaping,
or air-time targets. The prior `QuietStep` variant tried that and failed
to train. This spec keeps the door open to a stable double-stance "shuffle"
gait as long as the body does not leave the ground.

## Baseline

The baseline is `Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-QuietZeroStance`
(env_cfg factory
`unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_env_cfg`,
runner factory
`unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_ppo_runner_cfg`).

This task is known to train successfully and is what produces the hopping
gait we want to fix.

Key reward weights inherited from QZS (front feet are stance indices `(0, 1)`):

- `contact` (`handstand_stance_contact_mean`) = **+2.0** — mean of stance feet planted.
  Pays equally for one foot down or both feet down. **No anti-hop signal.**
- `tracking_lin_vel` (soft-gate) = **+3.0** — primary driver of forward motion.
- `handstand_feet_on_air` (rear) = **+5.0** — rear feet should be airborne.
- `handstand_feet_height_exp` (rear) = **+10.0** — rear-foot world-z target.
- `zero_stance_contact` = **+3.0** — both stance feet planted under zero command.
- `lin_vel_z` (body-x reward, exp) = **+0.6** — low vertical body motion.
- `action_rate_l2` = **-0.12**, `dof_acc` = **-1.0e-3**.

Why the baseline hops: nothing in the moving phase punishes the
"both front feet airborne" frame. The `contact` mean-reward is content
with one foot or two feet down. Velocity tracking is happiest with
ballistic forward motion if it can keep the body roughly oriented and
the rear feet up — and hopping does exactly that. Under zero command the
+3.0 `zero_stance_contact` does discourage hopping, but only via missed
reward: the policy can still trade a small slice of that for a brief
hop if it helps balance.

## Approach

Add **one** new reward term, applied to both moving and zero-command
phases, that pays a hard negative whenever the body is in handstand pose
and **both** front (stance) feet are simultaneously off the ground.

This is the minimum-surface change to QZS:

- Inherit the QZS config wholesale.
- Disable `feet_air_time` (which incentivizes long-swing strides) by
  setting its weight to `0.0`. QZS already has it muted indirectly via
  the absence of command/threshold params, but be explicit.
- Add new reward `stance_air_penalty` using a new helper function
  `handstand_stance_air_penalty` exported from `go2_mjlab.mdp.rewards`.
  Weight = **-2.0**.

Gating: the new reward only fires while the body is in handstand pose
(`_handstand_quality(env, target_height=0.08) > 0.70`). No command-based
gating, so it covers both moving and standing.

We deliberately do **not** add any single-stance bonus or double-stance
penalty. The QuietStep variant tried `+2.5` single-stance and a `-0.5`
double-stance penalty and failed to train; the policy could not exit the
all-feet-down basin into reliable alternation. By leaving double-stance
unpenalized, the new task lets the policy choose between (a) a stable
double-stance shuffle when standing and (b) a natural alternating gait
when moving — both meet the user's constraint.

## Code Changes

### 1. New reward function in `go2_mjlab/mdp/rewards.py`

Insert near the existing `handstand_moving_no_stance_contact` definition
(around line 1136). Mirrors the existing helper but drops the moving
mask so the penalty also applies to zero-command standing frames.

```python
def handstand_stance_air_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_indices: tuple[int, ...],
    target_height: float = 0.08,
) -> torch.Tensor:
    """Anti-hop: penalize handstand frames with all stance feet airborne.

    Unlike ``handstand_moving_no_stance_contact`` this term has no
    command-based gate and fires whether the policy is commanded to move
    or stand. The orientation gate keeps the term silent while the body
    is collapsing or not yet in handstand pose.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    selected = contact[:, list(foot_indices)]
    n_contact = torch.sum(selected, dim=1)
    quality = _handstand_quality(env, target_height)
    return (n_contact == 0).float() * (quality > 0.70).float()
```

### 2. New env_cfg in `go2_mjlab/config/lite3_handstand_rldeploy/env_cfgs.py`

```python
def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """QuietZeroStance variant with an unconditional anti-hop penalty."""
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["feet_air_time"].weight = 0.0
    rewards["stance_air_penalty"] = RewardTermCfg(
        func=go2_mdp.handstand_stance_air_penalty,
        weight=-2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "foot_indices": (0, 1),
            "target_height": 0.08,
        },
    )

    return cfg
```

### 3. New rl_cfg in `go2_mjlab/config/lite3_handstand_rldeploy/rl_cfg.py`

```python
def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop"
    return cfg
```

### 4. Register task in `go2_mjlab/__init__.py`

```python
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_ppo_runner_cfg(),
)
```

Add corresponding imports in `__init__.py`.

## Training

Train on lionrock (`192.168.5.142`, user `lionrock`) via `claude-ssh`:

- Set `WANDB_API_KEY=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`
  in the tmux session before launching.
- Run `python scripts/train_go2.py Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop`
  from the project root on lionrock.
- mjlab defaults to wandb logging (`project="mjlab"`); run name is set to
  the experiment name `lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop`.
- Max iterations inherits from the base lite3 handstand runner = **15_000**.

## Validation

- Spec changes pass `python -c "import go2_mjlab"` (catches typos and
  missing reward params during task registration).
- After training, replay the policy locally with
  `python scripts/train_go2.py Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop --play`
  and visually confirm:
    1. Zero command for ≥10 s: at least one front foot stays planted.
    2. Forward command at 0.4 m/s for ≥10 s: stepping/shuffle gait,
       no airborne frames.
    3. Backward command at -0.4 m/s for ≥10 s: same.
- wandb metric to watch: `Reward/stance_air_penalty` should converge
  toward zero. If it stays markedly negative the penalty is being paid
  often — that signals the policy still hops and the weight needs to be
  more negative.
