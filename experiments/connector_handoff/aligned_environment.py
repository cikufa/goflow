"""INSERT resets from measured physical grasp/stage executions.

The stored states initialize simulation only. PPO rollouts and GoFlow updates
remain online. No demonstrations enter a learning loss or the actor observation.
"""
import hashlib
import json

import gymnasium as gym
import numpy as np
import torch
from pxr import Gf
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from .environment import ROOT, ConnectorEnv, ConnectorInsertEnvCfg, IPose

SPEC_PATH = ROOT / 'experiments/connector_handoff/aligned_initialization.json'
SPEC = json.loads(SPEC_PATH.read_text())


@configclass
class AlignedConnectorEnvCfg(ConnectorInsertEnvCfg):
    dr_ranges = {key: tuple(value) for key, value in SPEC['dr_ranges'].items()}


class AlignedConnectorEnv(ConnectorEnv):
    def update_rigid_attachments(self, robot_name, fixed_joints, obj_T_flange, rigid_object, env_ids):
        # Retain the released context draw and the current physical fixture map.
        super().update_rigid_attachments(robot_name, fixed_joints, obj_T_flange, rigid_object, env_ids)
        if not hasattr(self, '_handoff_bank'):
            path = ROOT / SPEC['bank']
            if hashlib.sha256(path.read_bytes()).hexdigest() != SPEC['bank_sha256']:
                raise ValueError('Handoff bank hash differs from the recorded calibration')
            with np.load(path) as data:
                self._handoff_bank = {k: torch.as_tensor(data[k], device=self.device) for k in data.files}
        bank = self._handoff_bank
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        context = self.context[ids]
        bounds = torch.tensor(list(self.cfg.dr_ranges.values()), device=self.device)
        scale = bounds[:, 1] - bounds[:, 0]
        distance = ((context[:, None] - bank['context'][None]) / scale).square().sum(-1)
        neighbors = distance.topk(min(16, distance.shape[1]), largest=False).indices
        selected = neighbors[torch.arange(len(ids), device=self.device),
                             torch.randint(neighbors.shape[1], (len(ids),), device=self.device)]
        self.last_handoff_bank_indices = selected.clone()
        robot = self.robots[robot_name]
        robot.write_joint_state_to_sim(bank['robot_joint_pos'][selected], bank['robot_joint_vel'][selected], env_ids=ids)
        robot.set_joint_position_target(bank['robot_joint_targets'][selected], env_ids=ids)
        relative = bank['relative_pose'][selected].clone()
        roll, pitch, _ = euler_xyz_from_quat(relative[:, 3:7])
        relative[:, :2] = context[:, 1:3]
        relative[:, 3:7] = quat_from_euler_xyz(roll, pitch, context[:, 0])
        measured = bank['relative_pose'][selected]
        authored = bank['constraint_pose'][selected]
        # Contact-loaded actual pose is not the joint's rest transform. Reusing
        # the deflected actual pose as its rest pose would apply deflection twice.
        constraint = IPose(relative[:, :3], relative[:, 3:7]).multiply(
            IPose(measured[:, :3], measured[:, 3:7]).invert()).multiply(
            IPose(authored[:, :3], authored[:, 3:7]))
        hand = bank['hand_pose'][selected]
        peg = IPose(hand[:, :3], hand[:, 3:7]).multiply(IPose(relative[:, :3], relative[:, 3:7]))
        rigid_object.write_root_pose_to_sim(torch.cat((peg.pos + self.scene.env_origins[ids], peg.quat), 1), env_ids=ids)
        rigid_object.write_root_velocity_to_sim(bank['peg_root_state'][selected, 7:13], env_ids=ids)
        for row, env_id in enumerate(ids.tolist()):
            joint = fixed_joints[env_id]
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*constraint.pos[row].tolist()))
            joint.GetLocalRot0Attr().Set(Gf.Quatf(*constraint.quat[row].tolist()))
            joint.GetJointEnabledAttr().Set(True)

    def step_sim(self):
        # The released reset overwrites motor targets with current joint angles.
        # Preserve the empirical targets supporting the loaded grasp instead.
        if not hasattr(self, '_handoff_bank'):
            return super().step_sim()
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(dt=self.physics_dt)


gym.register(
    id='ConnectorAligned-GOFLOW-v0', entry_point=AlignedConnectorEnv,
    disable_env_checker=True,
    kwargs={'env_cfg_entry_point': AlignedConnectorEnvCfg,
            'rl_games_cfg_entry_point': str(ROOT/'experiments/original_gears/privileged_goflow.yaml')},
)
