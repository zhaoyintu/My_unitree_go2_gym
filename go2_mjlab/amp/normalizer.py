"""Running mean/std normalizer for AMP observations — ported from rl_amp.

(rl_amp/rsl_rl/rsl_rl/utils/utils.py RunningMeanStd + Normalizer.)  numpy-based
running statistics updated from un-normalized policy + expert AMP-obs; provides
a torch normalize for the discriminator forward pass.  Persisted in the
checkpoint via its (mean, var, count) state, not as torch buffers.
"""

from __future__ import annotations

import numpy as np
import torch


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m_2 / tot_count
        self.count = batch_count + self.count


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon: float = 1e-4, clip_obs: float = 10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize_torch(self, x: torch.Tensor, device) -> torch.Tensor:
        mean_t = torch.tensor(self.mean, device=device, dtype=torch.float32)
        std_t = torch.sqrt(torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp((x - mean_t) / std_t, -self.clip_obs, self.clip_obs)

    # --- checkpoint persistence ---
    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd: dict) -> None:
        self.mean = sd["mean"]
        self.var = sd["var"]
        self.count = sd["count"]
