from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Lite3Cfg_Handstand(LeggedRobotCfg):
    """Lite3 handstand: stands on rear legs (HL, HR), front legs (FL, FR) airborne.

    Body x-axis points up in world frame at handstand pose
    (target_gravity = [-1, 0, 0] in body frame).
    """

    class env:
        frame_stack = 1
        c_frame_stack = 1
        num_single_obs = 48
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 89
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20
        num_envs = 4096
        test = False

    class safety:
        pos_limit = 0.9
        vel_limit = 1.0
        torque_limit = 0.9

    class terrain:
        mesh_type = 'plane'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = None
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0., 0., 1.0, 0.0, 0.0]
        slope_treshold = 0.75

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 5.
        heading_command = False

        class ranges:
            lin_vel_x = [-0.4, 0.4]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.4, 0.4]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        # Standing pose (default Lite3 stance)
        default_joint_angles = {
            'FL_HipX_joint': 0.0,  'FL_HipY_joint': -0.8, 'FL_Knee_joint': 1.6,
            'FR_HipX_joint': 0.0,  'FR_HipY_joint': -0.8, 'FR_Knee_joint': 1.6,
            'HL_HipX_joint': 0.0,  'HL_HipY_joint': -0.8, 'HL_Knee_joint': 1.6,
            'HR_HipX_joint': 0.0,  'HR_HipY_joint': -0.8, 'HR_Knee_joint': 1.6,
        }
        # Handstand target pose: rear legs (HL, HR) support, front legs (FL, FR) tucked up.
        # NOTE: initial guess — needs visual calibration in MuJoCo viewer.
        # HipY range [-2.67, 0.314]; Knee range [0.524, 2.792].
        descire_joint_angles = {
            # Front legs tucked toward chest (airborne when body inverted backward)
            'FL_HipX_joint': 0.0,  'FL_HipY_joint': -2.4, 'FL_Knee_joint': 2.5,
            'FR_HipX_joint': 0.0,  'FR_HipY_joint': -2.4, 'FR_Knee_joint': 2.5,
            # Rear legs support — pushed to extend body upward
            'HL_HipX_joint': 0.0,  'HL_HipY_joint': -0.8, 'HL_Knee_joint': 1.6,
            'HR_HipX_joint': 0.0,  'HR_HipY_joint': -0.8, 'HR_Knee_joint': 1.6,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 40.}
        damping = {'joint': 1.}
        action_scale = 0.25
        decimation = 4

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/lite3/urdf/Lite3.urdf'
        name = "lite3"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "SHANK", "HIP"]
        terminate_after_contacts_on = ["TORSO"]
        # Handstand: rear legs (HL, HR) on ground, front legs (FL, FR) in air.
        feet_name_reward = ['FL_FOOT', 'FR_FOOT']
        contact_foot = ['HL_FOOT', 'HR_FOOT']
        target_gravity = [-1.0, 0.0, 0.0]
        threshold = 5.0
        self_collisions = 1
        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        default_dof_drive_mode = 3
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False  # Lite3 STL meshes already z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        push_towards_goal = True
        randomize_friction = True
        friction_range = [0.2, 0.8]

        randomize_restitution = True
        restitution_range = [0.0, 0.3]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 1.0
        max_push_ang_vel = 1.0

        randomize_base_mass = True
        added_base_mass_range = [-0.5, 1.0]  # Lite3 ~10.4 kg, smaller perturbation than Go2

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.05, 0.05]

        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]

        randomize_joint_friction = True
        joint_friction_range = [0.01, 0.2]

        randomize_joint_damping = True
        joint_damping_range = [0.0, 0.2]

        randomize_joint_armature = True
        joint_armature_range = [0.005, 0.015]

        add_obs_latency = True
        randomize_obs_motor_latency = True
        randomize_obs_imu_latency = True
        range_obs_motor_latency = [1, 3]
        range_obs_imu_latency = [1, 3]

        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [1, 3]

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 2.5
            tracking_ang_vel = 2.5
            tracking_lin_vel_zero = -0.2
            tracking_ang_vel_zero = -0.2
            lin_vel_z = 0.2
            ang_vel_xy = 0.2
            handstand_orientation = -1.0
            torques = -0.0002
            dof_vel = -0.
            dof_acc = -2.5e-4
            base_height = 1.5
            handstand_feet_on_air = 0.4
            collision = -2.
            feet_stumble = -0.0
            action_rate = -0.05
            default_pos = -0.1
            default_hip_pos = -0.1
            feet_clearance = 0.4
            ang_xz = -0.5
            contact = 0.3
            feet_air_time = 2.0
            symmetric_joints = -0.1
            handstand_feet_height_exp = 5.0
            default_pos_reward = 0.5
            dof_pos_limits = -2.0
            alive = 1.0

        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        # Lite3 stand height ~0.35 m + extended legs in handstand → estimate ~0.55 m.
        # Calibrate after first runs.
        base_height_target = 0.55
        target_foot_height = 0.06
        max_contact_force = 200.
        cycle_time = 1.6

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class Lite3CfgPPO_Handstand(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        sym_loss = False
        # Permutations follow Go2 handstand (left-right symmetry across 12 joints).
        # Lite3 joint order in URDF (FL HipX,HipY,Knee; FR ...; HL ...; HR ...) matches Go2's
        # FL/FR/RL/RR convention modulo naming, so the same permutation applies.
        obs_permutation = [-0.0001, -1, 2, -3, 4, -5,
                           -11, -12, 13, 14, 15, -16, -5, -6, 7, 8, 9, -10,
                           -23, -24, 25, 26, 27, -28, -17, -18, 19, 20, 21, -22,
                           -35, -36, 37, 38, 39, -40, -29, -30, 31, 32, 33, -34,
                           -41, 42, -43, -44, 45, -46]
        act_permutation = [-3, 4, 5, -0.0001, 1, 2, -9, 10, 11, -6, 7, 8]
        frame_stack = 10
        sym_coef = 1.0

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 15000

        save_interval = 100
        experiment_name = 'lite3_handstand'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
