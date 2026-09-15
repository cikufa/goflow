"""Native connector scene derived from Gears; geometry diagnostic, not a trained skill.

Import after AppLauncher. Original Gears globals and registered task are unchanged.
"""
from pathlib import Path
import hashlib

import torch
import gymnasium as gym
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.utils.math as math_utils

from goflow.environments.med_gear.direct_panda_position import (
    MyPandaEnv, MyPandaEnvCfg, INITIAL_CFG, IPose,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_FRONT_RADIUS = .120
FIXTURE_SIDE_RADIUS = .050
GRASP_OFFSET = .012
INSERTION_TIP_EXTENSION = .024


def _box(stage, path, center, size, color):
    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.)
    box.AddTranslateOp().Set(Gf.Vec3d(*center))
    box.AddScaleOp().Set(Gf.Vec3d(*size))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    collision = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
    collision.CreateContactOffsetAttr(.0005)
    collision.CreateRestOffsetAttr(0.)


def geometry_assets():
    """Build small reproducible USD assets locally, with explicit compound colliders."""
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    folder = ROOT / '.cache/connector_assets' / version
    folder.mkdir(parents=True, exist_ok=True)
    connector_path, socket_path = folder / 'connector.usda', folder / 'socket.usda'
    if connector_path.exists() and socket_path.exists():
        return connector_path, socket_path
    for path, dynamic in ((connector_path, True), (socket_path, False)):
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, '/Asset').GetPrim()
        stage.SetDefaultPrim(root)
        if dynamic:
            UsdPhysics.RigidBodyAPI.Apply(root)
            UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.05)
            PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateDisableGravityAttr(True)
            # Local +Z points down: leave an exposed insertion tip below fingers.
            _box(stage, '/Asset/body', (0, 0, INSERTION_TIP_EXTENSION/2),
                 (.070, .024, .024+INSERTION_TIP_EXTENSION), (.12, .14, .16))
            _box(stage, '/Asset/key', (.015, .014, INSERTION_TIP_EXTENSION/2),
                 (.010, .004, .018+INSERTION_TIP_EXTENSION), (.12, .14, .16))
        else:
            # Opening: +/-35.6 mm in local x, +/-12.6 mm in local y, with key notch.
            gray = (.45, .47, .5)
            _box(stage, '/Asset/end_left', (-.043, 0, 0), (.0148, .052, .020), gray)
            _box(stage, '/Asset/end_right', (.043, 0, 0), (.0148, .052, .020), gray)
            _box(stage, '/Asset/side_lower', (0, -.0193, 0), (.1008, .0134, .020), gray)
            _box(stage, '/Asset/side_upper_left', (-.02045, .0193, 0), (.0599, .0134, .020), gray)
            _box(stage, '/Asset/side_upper_right', (.03545, .0193, 0), (.0299, .0134, .020), gray)
            _box(stage, '/Asset/key_back', (.015, .0213, 0), (.011, .0094, .020), gray)
        stage.GetRootLayer().Save()
    return connector_path, socket_path


@configclass
class ConnectorEnvCfg(MyPandaEnvCfg):
    # Retain two seconds and three translational residual actions for this probe.
    dr_ranges = {'yaw_offset': (-.12, .12), 'x_offset': (-.040, .040),
                 'y_offset': (-.004, .004), 'fixture_angle': (-torch.pi, torch.pi)}
    fixture_distance_scale = 1.0  # explicit removal control in the geometry probe

    def __post_init__(self):
        super().__post_init__()
        self.num_states = self.num_observations + len(self.dr_ranges)
        self.state_space = self.num_states
        connector, socket = geometry_assets()
        goal = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).multiply(
            IPose.from_pose(INITIAL_CFG.HOLE_T_PEG_GOAL))
        peg = next(part for part in self.parts if part.prim_path.endswith('/peg'))
        peg.spawn.usd_path = str(connector)
        self.parts = [peg, RigidObjectCfg(
            prim_path='/World/envs/env_.*/fixture',
            spawn=sim_utils.CuboidCfg(size=(.012, .160, .150),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.25, .28, .32))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(2., 0., .1)))]
        self.peripherals = [p for p in self.peripherals if p.prim_path.endswith('/table')]
        socket_pos = goal.pos[0].clone()
        socket_pos[2] -= INSERTION_TIP_EXTENSION
        self.peripherals.append(AssetBaseCfg(
            prim_path='/World/envs/env_.*/socket', spawn=sim_utils.UsdFileCfg(usd_path=str(socket)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(socket_pos.tolist()),
                                                   rot=tuple(goal.quat[0].tolist()))))
        self.viewer.lookat = tuple(goal.pos[0].tolist())
        self.viewer.eye = tuple((goal.pos[0] + torch.tensor([.4, .4, .35], device=goal.pos.device)).tolist())


class ConnectorEnv(MyPandaEnv):
    def create_rigid_attachments(self, attachment_path, robot_path, extra_attachments=()):
        return super().create_rigid_attachments('peg', robot_path, list(extra_attachments))

    def update_rigid_attachments(self, robot_name, fixed_joints, obj_T_flange, rigid_object, env_ids):
        result = super().update_rigid_attachments(robot_name, fixed_joints, obj_T_flange, rigid_object, env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        yaw = self.context[ids, 0]
        zeros = torch.zeros_like(yaw)
        rotation = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        for i, env_id in enumerate(ids.tolist()):
            fixed_joints[env_id].GetLocalRot0Attr().Set(Gf.Quatf(*rotation[i].tolist()))
        # Initialize the object consistently with the commanded rigid grasp.
        tool = IPose.from_pose(INITIAL_CFG.WORLD_T_PEG_START).multiply(obj_T_flange).repeat(len(ids))
        offset = obj_T_flange.invert().repeat(len(ids))
        offset.pos[:, 0] += self.context[ids, 1]
        offset.pos[:, 1] += self.context[ids, 2]
        offset.quat = rotation
        peg = tool.multiply(offset)
        rigid_object.write_root_pose_to_sim(torch.cat((peg.pos + self.scene.env_origins[ids], peg.quat), 1), env_ids=ids)
        rigid_object.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)

        goal = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).multiply(IPose.from_pose(INITIAL_CFG.HOLE_T_PEG_GOAL))
        angle = self.context[ids, 3]
        position = goal.pos.expand(len(ids), -1).clone() + self.scene.env_origins[ids]
        position[:, 0] += self.cfg.fixture_distance_scale * FIXTURE_FRONT_RADIUS * torch.cos(angle)
        position[:, 1] += self.cfg.fixture_distance_scale * FIXTURE_SIDE_RADIUS * torch.sin(angle)
        position[:, 2] += .070
        quat = math_utils.quat_from_euler_xyz(zeros, zeros, angle)
        self.scene['fixture'].write_root_pose_to_sim(torch.cat((position, quat), 1), env_ids=ids)
        return result

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # The diagnostic resets all environments synchronously, including batches <10.
        if len(env_ids) == self.num_envs:
            self.pose_history.clear()

    def _get_rewards(self):
        reward = super()._get_rewards()
        goal = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).multiply(IPose.from_pose(INITIAL_CFG.HOLE_T_PEG_GOAL))
        self.last_peg_pose = self.scene['peg'].data.root_state_w[:, :7].clone()
        self.last_goal_distance = torch.linalg.vector_norm(
            self.last_peg_pose[:, :3] - self.scene.env_origins - goal.pos, dim=1)
        return reward


@configclass
class ConnectorInsertEnvCfg(ConnectorEnvCfg):
    """INSERT starts with the closed fingers produced by the grasp macro."""
    def __post_init__(self):
        super().__post_init__()
        for name in ('panda_finger_joint1', 'panda_finger_joint2'):
            self.robot1.init_state.joint_pos[name] = .012


gym.register(
    id='Connector-GOFLOW-v0',
    entry_point=ConnectorEnv,
    disable_env_checker=True,
    kwargs={'env_cfg_entry_point': ConnectorInsertEnvCfg,
            'rl_games_cfg_entry_point': str(ROOT/'experiments/original_gears/privileged_goflow.yaml')},
)
