import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import mdp
from .assets import G1_CFG
from .terrains import (
    CORRIDOR_TERRAIN_NAMES,
    CORRIDOR_TERRAINS_CFG,
    LOCOMOTION_VALIDATION_TERRAINS_CFG,
    TRAINING_TERRAINS_CFG,
)

FEET = SceneEntityCfg("robot", body_names=".*ankle_roll.*")
FEET_CONTACT = SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")
LEG_JOINTS = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]
SCAN_PARAMS = {
    "sensor_cfg": SceneEntityCfg("height_scanner"),
    "scan_resolution": 0.05,
    "scan_size": (1.6, 1.0),
    "patch_size": 7,
}


@configclass
class CommandsCfg:
    base_velocity = mdp.TurnInPlaceVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_turning_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.TurnInPlaceVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=mdp.UniformSpikeNoiseCfg(n_min=-0.15, n_max=0.15, spike_probability=1e-4, spike_amplitude=5.0),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=mdp.UniformSpikeNoiseCfg(n_min=-0.2, n_max=0.2, spike_probability=1e-4, spike_amplitude=10.0),
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))
        foothold_scan = ObsTerm(
            func=mdp.foothold_scan,
            params={**SCAN_PARAMS, "xy_drift_range": (-0.05, 0.05), "z_drift_range": (-0.05, 0.05)},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))
        height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0)
        )
        foothold_scan = ObsTerm(func=mdp.foothold_scan, params=dict(SCAN_PARAMS), clip=(-1.0, 1.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemStateCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class SystemContactCfg(ObsGroup):
        feet_contact = ObsTerm(func=mdp.body_contact, params={"sensor_cfg": FEET_CONTACT, "threshold": 1.0})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemPoseCfg(ObsGroup):
        base_pose_w = ObsTerm(func=mdp.base_pose_w)
        feet_pose_w = ObsTerm(func=mdp.body_pose_w, params={"asset_cfg": FEET})
        feet_pos_b = ObsTerm(func=mdp.body_pos_b, params={"asset_cfg": FEET})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    system_state: SystemStateCfg = SystemStateCfg()
    system_contact: SystemContactCfg = SystemContactCfg()
    system_pose: SystemPoseCfg = SystemPoseCfg()


@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    stand_still = RewTerm(
        func=mdp.stand_still_nominal_pose,
        weight=2.5,
        params={"command_name": "base_velocity", "threshold": 0.1, "std": 2.0},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-8)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_bounded, weight=-0.01, params={"max_delta": 5.0})
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])}
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"]
            )
        },
    )
    upright_posture = RewTerm(
        func=mdp.upright_posture, weight=0.25, params={"asset_cfg": FEET, "target_height": 0.72, "std": 0.06}
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})
    nan_state = DoneTerm(func=mdp.nan_physics_state)


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    default_joint_pos_offset = EventTerm(
        func=mdp.randomize_joint_default_pos, mode="startup", params={"position_range": (-0.01, 0.01)}
    )
    actuator_pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "friction_distribution_params": (-0.05, 0.05),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    gravity_assist = EventTerm(
        func=mdp.apply_gravity_compensation,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "init_offset": 0.5, "cutoff_level": 4.0},
    )
    constant_disturbance = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "force_range": (-7.0, 7.0),
            "torque_range": (0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.3, 0.3),
                "z": (-0.2, 0.2),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.2, 0.2),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            }
        },
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    gravity_offset_level = CurrTerm(func=mdp.gravity_offset_level, params={"init_offset": 0.5, "cutoff_level": 4.0})


@configclass
class FpmEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    load_run_dir: str | None = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8192
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0])
        self.scene.terrain.terrain_generator = TRAINING_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 2
        self.actions.joint_pos.scale = 0.25
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        self.sim.physx.gpu_collision_stack_size = 2**28


@configclass
class FpmEnvCfg_PLAY(FpmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None

        self.observations.policy.enable_corruption = False
        for group in (self.observations.policy, self.observations.critic):
            group.foothold_scan.params.update(
                xy_drift_range=(0.0, 0.0), z_drift_range=(0.0, 0.0), train_fpm=False, force_model=True
            )

        self.events.gravity_assist = None
        self.events.push_robot = None
        self.events.actuator_pd_gains = None
        self.events.joint_friction = None
        self.events.constant_disturbance = None
        self.events.physics_material.params["static_friction_range"] = (1.0, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.0)

        self.commands.base_velocity.resampling_time_range = (1e9, 1e9)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_control_stiffness = 0.0


@configclass
class FpmLocomotionValidationEnvCfg(FpmEnvCfg_PLAY):
    episode_length: float = 10.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 13 * 800
        self.scene.env_spacing = 0.0
        self.scene.terrain.terrain_generator = LOCOMOTION_VALIDATION_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        self.episode_length_s = self.episode_length
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (-1.0, 1.0)
        cmd.ranges.lin_vel_y = (-0.5, 0.5)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.resampling_time_range = (self.episode_length, self.episode_length)

        self.events.reset_base.params["pose_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "yaw": (-math.pi, math.pi)}
        self.events.reset_base.params["velocity_range"] = {
            k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
        }


@configclass
class FpmFootPredValidationEnvCfg(FpmEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 13 * 400
        self.scene.env_spacing = 0.0
        self.scene.terrain.terrain_generator = LOCOMOTION_VALIDATION_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None


@configclass
class FpmCorridorValidationEnvCfg(FpmEnvCfg_PLAY):
    robots_per_variant: int = 48
    goal_distance: float = 4.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = CORRIDOR_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.scene.num_envs = len(CORRIDOR_TERRAIN_NAMES) * self.robots_per_variant
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.episode_length_s = 12.0

        cmd = self.commands.base_velocity
        cmd.resampling_time_range = (1.0e9, 1.0e9)
        cmd.rel_standing_envs = 0.0
        cmd.rel_turning_envs = 0.0
        cmd.ranges.lin_vel_x = (0.5, 1.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)

        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "yaw": (math.pi / 2, math.pi / 2),
        }
        self.events.reset_base.params["velocity_range"] = {
            k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (1.0, 1.0)
        self.events.add_base_mass.params["mass_distribution_params"] = (1.0, 1.0)
        self.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        for group in (self.observations.policy, self.observations.critic):
            group.enable_corruption = False
            for term in group.__dict__.values():
                if isinstance(term, ObsTerm):
                    term.noise = None

        self.terminations.reached_goal = DoneTerm(
            func=mdp.displacement_from_origin, params={"distance": self.goal_distance}
        )
