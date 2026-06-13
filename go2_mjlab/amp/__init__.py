"""Self-contained AMP (Adversarial Motion Priors) module for mjlab.

Vendors the rl_amp AMP algorithm (discriminator, replay buffer, expert motion
loader, running normalizer) and a custom OnPolicyRunner that bolts AMP onto the
stock rsl_rl PPO, plus the env-side AMP observation function.
"""

from go2_mjlab.amp.cfg import RslRlAmpOnPolicyRunnerCfg
from go2_mjlab.amp.discriminator import AMPDiscriminator
from go2_mjlab.amp.motion_loader import AMPLoaderNPZ
from go2_mjlab.amp.normalizer import Normalizer
from go2_mjlab.amp.observations import amp_observations
from go2_mjlab.amp.replay_buffer import AMPReplayBuffer
from go2_mjlab.amp.runner import Lite3AmpOnPolicyRunner

__all__ = [
    "RslRlAmpOnPolicyRunnerCfg",
    "AMPDiscriminator",
    "AMPLoaderNPZ",
    "Normalizer",
    "amp_observations",
    "AMPReplayBuffer",
    "Lite3AmpOnPolicyRunner",
]
