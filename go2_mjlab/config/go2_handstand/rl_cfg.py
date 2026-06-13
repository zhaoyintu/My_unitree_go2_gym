"""RL configuration for Unitree Go2 handstand task."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_go2_handstand_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
        experiment_name="go2_handstand",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=15_000,
    )


def unitree_go2_handstand_finetune_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Stage-2 finetune runner: resume the stage-1 checkpoint, add energy cost.

    ``experiment_name`` stays "go2_handstand" — mjlab resolves
    ``load_run``/``load_checkpoint`` inside the experiment's log root, so
    the finetune run can restore the stage-1 weights and its own run dir
    lands alongside (tagged via ``run_name``).  rsl_rl restores the
    iteration counter from the checkpoint and trains ``max_iterations``
    ADDITIONAL iterations (14999 -> 17999).

    Override the source checkpoint from the CLI when needed:
      --agent.load_run <run_dir_name> --agent.load_checkpoint model_XXXX.pt
    """
    cfg = unitree_go2_handstand_ppo_runner_cfg()
    cfg.run_name = "finetune"
    cfg.resume = True
    cfg.load_run = "2026-05-01_18-22-08"      # verified-good stage-1 run
    cfg.load_checkpoint = "model_14999.pt"
    cfg.max_iterations = 3_000
    return cfg
