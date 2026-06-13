#!/usr/bin/env python3
"""Train wrapper: warm-start actor weights, freeze actor for 1000 iters.

Loads ONLY the actor (and obs normalizer) from the v14 checkpoint. For the
first ``FREEZE_ITERS`` PPO iterations, zeros out actor parameter gradients
before ``optimizer.step()`` — only the critic learns. Once the critic is
roughly aligned, the actor unfreezes and joint PPO resumes.

Also caps the loaded ``distribution.std_param`` to keep exploration noise
under control (v14's learned std=0.9 on front HipX would destabilize the
first rollout).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import go2_mjlab  # noqa: E402, F401 — registers Go2 tasks

STD_CAP = 0.4
FREEZE_ITERS = 1000


def _patch_runner_load() -> None:
    import torch
    from mjlab.rl import runner as _runner_mod

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        d = original_torch_load(*args, **kwargs)
        if isinstance(d, dict) and "actor_state_dict" in d:
            asd = d["actor_state_dict"]
            if "distribution.std_param" in asd:
                std = asd["distribution.std_param"]
                if std.dim() > 0:
                    asd["distribution.std_param"] = std.clamp(max=STD_CAP)
                    print(
                        f"[INFO]: Capped distribution.std_param at {STD_CAP}; "
                        f"pre-cap max={std.max().item():.3f}."
                    )
        return d

    torch.load = patched_torch_load

    original_load = _runner_mod.MjlabOnPolicyRunner.load

    def actor_only_load(self, path, load_cfg=None, strict=True, map_location=None):
        if load_cfg is None:
            load_cfg = {"actor": True}
            print("[INFO]: Warm-start mode — loading actor only (skip critic/optimizer).")
        return original_load(self, path, load_cfg=load_cfg, strict=strict, map_location=map_location)

    _runner_mod.MjlabOnPolicyRunner.load = actor_only_load


def _patch_freeze_actor() -> None:
    """Wrap PPO's optimizer.step so actor grads are zero for FREEZE_ITERS iterations."""
    import torch
    from mjlab.rl import runner as _runner_mod

    original_init = _runner_mod.MjlabOnPolicyRunner.__init__

    def init_with_freeze(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        if FREEZE_ITERS <= 0:
            return

        alg = self.alg
        original_step = alg.optimizer.step
        actor_params = list(alg.actor.parameters())
        start_iter = self.current_learning_iteration

        def freezing_step(*step_args, **step_kwargs):
            iter_offset = self.current_learning_iteration - start_iter
            if iter_offset < FREEZE_ITERS:
                for p in actor_params:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
            return original_step(*step_args, **step_kwargs)

        alg.optimizer.step = freezing_step
        print(
            f"[INFO]: Actor frozen for the first {FREEZE_ITERS} learning iterations "
            f"(start_iter={start_iter}). Critic learns alone during this window."
        )

    _runner_mod.MjlabOnPolicyRunner.__init__ = init_with_freeze


def _apply_task_alias(argv):
    aliases = {"Mjlab-G1-": "Mjlab-Lite3-"}
    for i, arg in enumerate(argv[1:], 1):
        for src, dst in aliases.items():
            if arg.startswith(src):
                argv[i] = dst + arg[len(src):]
                return


if __name__ == "__main__":
    _patch_runner_load()
    _patch_freeze_actor()
    _apply_task_alias(sys.argv)
    from mjlab.scripts.train import main as mjlab_train_main
    mjlab_train_main()
