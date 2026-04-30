from legged_gym.envs.GO2_Stand.GO2_Leggedstand.Go2_legstand import Go2_legstand


class Lite3_legstand(Go2_legstand):
    """Lite3 leggedstand task — reuses Go2_legstand logic, differentiated by config.

    All Lite3-specific behavior (URDF, joint names, foot names, target_gravity,
    PD gains) is provided by Lite3Cfg_Leggedstand.
    """
    pass
