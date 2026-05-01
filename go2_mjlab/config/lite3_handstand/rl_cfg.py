"""RL configuration for DeepRobotics Lite3 handstand task."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_lite3_handstand_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                # init_std=1.0 caused action_std to drift up to 2.5+ on
                # Lite3 (policy explores more when no reward gradient
                # arrives), which combined with action_scale 0.125-0.25
                # produced ~0.6 rad random joint commands per step —
                # enough to flip the dog before a single second of
                # episode time.  0.3 keeps early action magnitude small
                # so policy can find a foothold.
                "init_std": 0.3,
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
        experiment_name="lite3_handstand",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=15_000,
    )
