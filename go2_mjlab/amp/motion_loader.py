"""AMP expert-motion loader for the ahmp .npz handstand data.

Reads the pre-assembled ``amp_frames (N, 61)`` array and slices the SAME
43-dim discriminator AMP-obs the env produces:

    amp_obs = [ amp_frames[:, 7:49] , amp_frames[:, 2:3] ]
            = [ joint_pos(12), foot_pos_base(12), base_lin_vel(3),
                base_ang_vel(3), joint_vel(12) ]  ++  root_z(1)

This mirrors rl_amp's AMPLoader.feed_forward_generator
(motion_loader.py:317-330: s = frame[JOINT_POSE:JOINT_VEL_END] ++ root[z]).
Joint order in the npz (FL,FR,HL,HR × HipX,HipY,Knee) and foot order
(FL,FR,HL,HR) already match the Lite3 MJCF, so NO pybullet→isaac reorder.

amp_frames column layout (verified against the npz named arrays):
  [0:3] root_pos  [3:7] root_quat_xyzw  [7:19] joint_pos  [19:31] foot_pos_base
  [31:34] base_lin_vel  [34:37] base_ang_vel  [37:49] joint_vel  [49:61] foot_vel_base
"""

from __future__ import annotations

import numpy as np
import torch

# Slices into the 61-dim amp_frames row.
_JP_FOOT_LIN_ANG_JV = slice(7, 49)   # joint_pos+foot_pos_base+lin_vel+ang_vel+joint_vel (42)
_ROOT_Z = slice(2, 3)                # world root height (1)
SINGLE_AMP_OBS_DIM = 43

EXPECTED_JOINT_NAMES = (
    "FL_HipX", "FL_HipY", "FL_Knee", "FR_HipX", "FR_HipY", "FR_Knee",
    "HL_HipX", "HL_HipY", "HL_Knee", "HR_HipX", "HR_HipY", "HR_Knee",
)
EXPECTED_FOOT_ORDER = ("FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT")


def _frame_to_amp_obs(amp_frames: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [amp_frames[:, _JP_FOOT_LIN_ANG_JV], amp_frames[:, _ROOT_Z]], axis=-1
    ).astype(np.float32)


class AMPLoaderNPZ:
    """Loads ahmp .npz motions and serves expert AMP transitions (s, s')."""

    def __init__(self, device, motion_files, time_between_frames: float,
                 num_preload_transitions: int = 2_000_000):
        self.device = device
        self.time_between_frames = float(time_between_frames)

        all_s, all_s_next = [], []
        for f in motion_files:
            data = np.load(f)
            # Sanity: joint/foot ordering must match the env's amp_observations.
            jn = tuple(str(x) for x in data["joint_names"])
            fo = tuple(str(x) for x in data["foot_order"])
            assert jn == EXPECTED_JOINT_NAMES, f"{f}: joint_names {jn} != {EXPECTED_JOINT_NAMES}"
            assert fo == EXPECTED_FOOT_ORDER, f"{f}: foot_order {fo} != {EXPECTED_FOOT_ORDER}"

            frame_duration = float(data["frame_duration"])
            step = max(1, int(round(self.time_between_frames / frame_duration)))
            obs = _frame_to_amp_obs(data["amp_frames"])  # (N, 43)
            n = obs.shape[0]
            # Clamp loop mode: next frame is t+step, clamped at the last frame.
            nxt_idx = np.minimum(np.arange(n) + step, n - 1)
            all_s.append(obs)
            all_s_next.append(obs[nxt_idx])

        s = np.concatenate(all_s, axis=0)
        s_next = np.concatenate(all_s_next, axis=0)

        # Preload a large bank of transitions (sampled with replacement) so the
        # discriminator's expert minibatches are i.i.d. draws over all motions.
        n_total = s.shape[0]
        sel = np.random.choice(n_total, size=num_preload_transitions)
        self.preloaded_s = torch.tensor(s[sel], dtype=torch.float32, device=device)
        self.preloaded_s_next = torch.tensor(s_next[sel], dtype=torch.float32, device=device)
        self._n = self.preloaded_s.shape[0]

    @property
    def observation_dim(self) -> int:
        return SINGLE_AMP_OBS_DIM

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        for _ in range(num_mini_batch):
            idxs = np.random.choice(self._n, size=mini_batch_size)
            yield self.preloaded_s[idxs], self.preloaded_s_next[idxs]
