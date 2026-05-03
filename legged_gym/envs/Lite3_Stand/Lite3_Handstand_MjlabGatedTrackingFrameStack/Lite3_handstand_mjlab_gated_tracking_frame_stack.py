"""Lite3 soft-gated handstand task with 10-frame actor/critic observations."""

from collections import deque

import torch

from legged_gym.envs.Lite3_Stand.Lite3_Handstand_MjlabGatedTrackingReward.Lite3_handstand_mjlab_gated_tracking_reward import (
    Lite3_legstand_mjlab_gated_tracking_reward,
)
from legged_gym.envs.Lite3_Stand.Lite3_Handstand_MjlabGatedTrackingFrameStack.Lite3_handstand_mjlab_gated_tracking_frame_stack_Config import (
    Lite3Cfg_MjlabGatedTrackingFrameStack,
    Lite3CfgPPO_MjlabGatedTrackingFrameStack,
)


class Lite3_legstand_mjlab_gated_tracking_frame_stack(
    Lite3_legstand_mjlab_gated_tracking_reward
):
    """Reward-identical gated-tracking task with stacked observations."""

    def __init__(self, cfg: Lite3Cfg_MjlabGatedTrackingFrameStack, sim_params,
                 physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        self.single_noise_scale_vec = self.noise_scale_vec[
            :self.cfg.env.num_single_obs
        ].clone()
        self.noise_scale_vec = self.single_noise_scale_vec
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs,
                self.cfg.env.num_single_obs,
                dtype=torch.float,
                device=self.device,
            ))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs,
                self.cfg.env.single_num_privileged_obs,
                dtype=torch.float,
                device=self.device,
            ))

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def compute_observations(self):
        super().compute_observations()
        single_obs = self.obs_buf.clone()
        single_privileged_obs = self.privileged_obs_buf.clone()

        self.obs_history.append(single_obs)
        self.critic_history.append(single_privileged_obs)

        obs_buf_all = torch.stack(
            [self.obs_history[i] for i in range(self.obs_history.maxlen)],
            dim=1,
        )
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.critic_history.maxlen)],
            dim=1,
        )
