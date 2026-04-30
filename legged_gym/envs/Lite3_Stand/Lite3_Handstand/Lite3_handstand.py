from legged_gym.envs.GO2_Stand.GO2_Handstand.Go2_handstand import Go2_stand


class Lite3_stand(Go2_stand):
    """Lite3 handstand task — reuses Go2_stand logic, differentiated by config.

    All Lite3-specific behavior (URDF, joint names, foot names, target_gravity,
    PD gains) is provided by Lite3Cfg_Handstand.
    """
    pass
