"""Custom AMP on-policy runner for the Lite3 handstand finetune.

Subclasses mjlab's MjlabOnPolicyRunner (so checkpoint/onnx/curriculum behavior
is inherited) and adds Adversarial Motion Priors on top of the STOCK rsl_rl
5.2.0 PPO, with NO change to mjlab or rsl_rl:

  * builds an AMPDiscriminator + AMPLoaderNPZ(expert) + Normalizer + policy
    AMPReplayBuffer in __init__;
  * overrides learn() — a faithful copy of rsl_rl 5.2.0 OnPolicyRunner.learn()
    — to, each rollout step: read obs["amp"] (the env's 43-dim AMP-obs),
    compute the discriminator style reward, BLEND it into the env reward
    (lerp toward the task reward) BEFORE alg.process_env_step, and stash the
    (s_t, s_{t+1}) transition into the replay buffer;
  * after alg.update() (which trains actor/critic on the blended reward),
    runs _update_amp(): an LSGAN discriminator update (expert vs policy) with a
    gradient penalty, on a SEPARATE Adam optimizer.

Registration passes runner_cls=Lite3AmpOnPolicyRunner; mjlab's load_runner_cls
seam wires it in. The actor/critic/PPO stay the 450-dim Stride-V15 contract
(obs_groups routes only "actor"/"critic"; the "amp" group is ignored by the
networks), so this warm-starts from the V15 checkpoint.

Inference-tensor safety: the rsl_rl rollout runs under torch.inference_mode(),
so obs["amp"] tensors are *inference tensors* and cannot enter autograd later.
We therefore (a) only COPY their values into the pre-allocated (normal) replay
buffer during the rollout, and (b) run the discriminator's forward/backward in
_update_amp exclusively on samples drawn from the replay buffer and the expert
loader (both normal tensors) — never on rollout tensors directly.
"""

from __future__ import annotations

import os
import statistics
import time

import torch
import torch.nn as nn
from rsl_rl.utils import check_nan

from mjlab.rl.runner import MjlabOnPolicyRunner

from go2_mjlab.amp.discriminator import AMPDiscriminator
from go2_mjlab.amp.motion_loader import AMPLoaderNPZ
from go2_mjlab.amp.normalizer import Normalizer
from go2_mjlab.amp.replay_buffer import AMPReplayBuffer


class Lite3AmpOnPolicyRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        super().__init__(env, train_cfg, log_dir, device)

        amp_obs_dim = int(env.get_observations()["amp"].shape[-1])  # 43
        step_dt = float(env.unwrapped.step_dt)

        self.amp_loader = AMPLoaderNPZ(
            device=device,
            motion_files=list(train_cfg["amp_motion_files"]),
            time_between_frames=step_dt,
            num_preload_transitions=int(train_cfg["amp_num_preload_transitions"]),
        )
        assert self.amp_loader.observation_dim == amp_obs_dim, (
            f"env amp-obs dim {amp_obs_dim} != dataset {self.amp_loader.observation_dim}"
        )
        self.amp_normalizer = Normalizer(amp_obs_dim)
        self.discriminator = AMPDiscriminator(
            input_dim=2 * amp_obs_dim,
            amp_reward_coef=float(train_cfg["amp_reward_coef"]),
            hidden_layer_sizes=list(train_cfg["amp_discr_hidden_dims"]),
            device=device,
            task_reward_lerp=float(train_cfg["amp_task_reward_lerp"]),
        ).to(device)
        self.amp_storage = AMPReplayBuffer(
            amp_obs_dim, int(train_cfg["amp_replay_buffer_size"]), device
        )
        self.disc_optimizer = torch.optim.Adam(
            [
                {"params": self.discriminator.trunk.parameters(),
                 "weight_decay": float(train_cfg["amp_disc_weight_decay_trunk"])},
                {"params": self.discriminator.amp_linear.parameters(),
                 "weight_decay": float(train_cfg["amp_disc_weight_decay_head"])},
            ],
            lr=float(train_cfg["amp_disc_learning_rate"]),
        )
        self._amp_grad_pen_lambda = float(train_cfg["amp_grad_pen_lambda"])
        self._amp_obs_dim = amp_obs_dim
        # Disc minibatch matches the PPO minibatch size.
        n_steps = int(train_cfg["num_steps_per_env"])
        self._amp_num_mb = int(train_cfg["algorithm"]["num_mini_batches"])
        self._amp_num_epochs = int(train_cfg["algorithm"]["num_learning_epochs"])
        self._amp_mb_size = max(1, (env.num_envs * n_steps) // self._amp_num_mb)

    # ------------------------------------------------------------------ learn
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            style_rew_acc: list[float] = []
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    prev_amp = obs["amp"]                       # (N, 43) inference tensor
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    )
                    next_amp = obs["amp"]
                    # Style reward (already lerped toward the task reward).
                    style_reward, _ = self.discriminator.predict_amp_reward(
                        prev_amp, next_amp, rewards, self.amp_normalizer
                    )
                    style_rew_acc.append(float(style_reward.mean()))
                    rewards = style_reward
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    )
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                    # Copy AMP transition into the (normal) replay buffer.
                    self.amp_storage.insert(prev_amp, next_amp)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            amp_stats = self._update_amp()
            loss_dict.update(amp_stats)
            loss_dict["amp/style_reward"] = (
                statistics.mean(style_rew_acc) if style_rew_acc else 0.0
            )

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    # --------------------------------------------------------------- amp update
    def _update_amp(self) -> dict:
        if self.amp_storage.num_samples < self._amp_mb_size:
            return {"amp/disc_loss": 0.0, "amp/grad_pen": 0.0,
                    "amp/policy_pred": 0.0, "amp/expert_pred": 0.0}
        n_batches = self._amp_num_epochs * self._amp_num_mb
        policy_gen = self.amp_storage.feed_forward_generator(n_batches, self._amp_mb_size)
        expert_gen = self.amp_loader.feed_forward_generator(n_batches, self._amp_mb_size)
        mse = nn.MSELoss()
        d_loss = gp_loss = pol_pred = exp_pred = 0.0
        n = 0
        for (policy_s, policy_s_next), (expert_s, expert_s_next) in zip(policy_gen, expert_gen):
            policy_s_un, expert_s_un = policy_s.clone(), expert_s.clone()
            policy_s = self.amp_normalizer.normalize_torch(policy_s, self.device)
            policy_s_next = self.amp_normalizer.normalize_torch(policy_s_next, self.device)
            expert_s = self.amp_normalizer.normalize_torch(expert_s, self.device)
            expert_s_next = self.amp_normalizer.normalize_torch(expert_s_next, self.device)

            policy_d = self.discriminator(torch.cat([policy_s, policy_s_next], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_s, expert_s_next], dim=-1))
            expert_loss = mse(expert_d, torch.ones_like(expert_d))
            policy_loss = mse(policy_d, -torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen = self.discriminator.compute_grad_pen(
                expert_s, expert_s_next, lambda_=self._amp_grad_pen_lambda
            )

            self.disc_optimizer.zero_grad()
            (amp_loss + grad_pen).backward()
            self.disc_optimizer.step()

            # Update the running normalizer from UN-normalized states.
            self.amp_normalizer.update(policy_s_un.cpu().numpy())
            self.amp_normalizer.update(expert_s_un.cpu().numpy())

            d_loss += amp_loss.detach().item()
            gp_loss += grad_pen.detach().item()
            pol_pred += policy_d.detach().mean().item()
            exp_pred += expert_d.detach().mean().item()
            n += 1
        n = max(1, n)
        return {
            "amp/disc_loss": d_loss / n,
            "amp/grad_pen": gp_loss / n,
            "amp/policy_pred": pol_pred / n,
            "amp/expert_pred": exp_pred / n,
        }

    # ------------------------------------------------------- checkpoint I/O
    def save(self, path: str, infos=None) -> None:
        infos = {
            **(infos or {}),
            "amp_state": {
                "discriminator": self.discriminator.state_dict(),
                "disc_optimizer": self.disc_optimizer.state_dict(),
                "normalizer": self.amp_normalizer.state_dict(),
            },
        }
        super().save(path, infos)

    def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        if infos and "amp_state" in infos:
            amp = infos["amp_state"]
            self.discriminator.load_state_dict(amp["discriminator"])
            try:
                self.disc_optimizer.load_state_dict(amp["disc_optimizer"])
            except (ValueError, KeyError):
                pass
            self.amp_normalizer.load_state_dict(amp["normalizer"])
            print("[AMP] restored discriminator + normalizer from checkpoint.")
        else:
            print("[AMP] no amp_state in checkpoint (warm-start) — discriminator starts fresh.")
        return infos
