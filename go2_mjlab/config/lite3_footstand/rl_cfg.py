"""RL configuration for DeepRobotics Lite3 footstand task.

PPO hyperparameters match the handstand_gym `LITECfgPPO_Handstand`
source exactly (they are the standard legged_gym defaults): MLP
(512, 256, 128) ELU, lr 1e-3 adaptive, gamma 0.99, lam 0.95,
24 steps/env.  Iteration budget extended past the source's 5000 to
the repo's usual 15000 (checkpoints every 100 — stop early if done).
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_lite3_footstand_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="lite3_footstand",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=15_000,
    )
