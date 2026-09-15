"""Execute visible-pose IK approach/close/attach/lift macros in native physics.

Uses the released gravity-disabled object and fixed-grasp approximation. This
validates macro execution and constrained hold, not frictional pickup quality.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common.runtime import kit_arguments

parser = argparse.ArgumentParser()
parser.add_argument('--trials-per-grasp', type=int, default=50)
parser.add_argument('--seed', type=int, default=41000)
parser.add_argument('--output', type=Path, default=ROOT/'results/custom_connector/grasp_validation')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
kit = kit_arguments()
from omni.isaac.lab.app import AppLauncher
app = AppLauncher(headless=True, kit_args=kit).app
try:
    import numpy as np
    import torch
    from pxr import Gf
    from experiments.connector_handoff.environment import ConnectorEnv, ConnectorEnvCfg, GRASP_OFFSET
    from goflow.environments.med_gear.direct_panda_position import IPose, INITIAL_CFG

    cfg = ConnectorEnvCfg()
    cfg.scene.num_envs = 2 * args.trials_per_grasp
    cfg.seed = args.seed
    env = ConnectorEnv(cfg)
    n = env.num_envs
    contexts = torch.zeros(n, 4, device=env.device)
    contexts[:, 1] = torch.tensor([-GRASP_OFFSET, GRASP_OFFSET], device=env.device).repeat_interleave(args.trials_per_grasp)
    class Cases:
        def rsample(self, shape):
            assert shape[0] == n
            return contexts.clone()
    env.set_sampling_dist(Cases())
    env.reset()
    robot = env.robots['robot1']
    fingers, _ = robot.find_joints('panda_finger_joint.*')
    for joint in env.robot1_fixed_joints:
        joint.GetJointEnabledAttr().Set(False)
    # Visible placement jitter, shared domain for the two local grasp actions.
    jitter = torch.zeros(n, 3, device=env.device).uniform_(-.003, .003)
    jitter[:, 2] = 0
    state = env.scene['peg'].data.root_state_w[:, :7].clone()
    state[:, :3] += jitter
    env.scene['peg'].write_root_pose_to_sim(state)
    env.scene['peg'].write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device))
    offset = IPose.from_pose(INITIAL_CFG.PEG_T_TOOL).invert().repeat(n)
    offset.pos[:, 0] += contexts[:, 1]
    visible_object = IPose(state[:, :3] - env.scene.env_origins, state[:, 3:7])
    target = visible_object.multiply(offset.invert())
    rows = []
    attached = torch.zeros(n, dtype=torch.bool, device=env.device)
    initial_object_z = state[:, 2].clone()

    def advance(phase, target_pos, finger_width, steps):
        for _ in range(steps):
            pos, quat = env._compute_frame_pose('robot1', robot)
            command = torch.cat((target_pos, target.quat), 1)
            env._ik_controllers['robot1'].set_command(command, pos, quat)
            for _ in range(cfg.decimation):
                env._apply_action()
                robot.set_joint_position_target(finger_width, joint_ids=fingers)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
            pos, quat = env._compute_frame_pose('robot1', robot)
            peg = env.scene['peg'].data.root_state_w[:, :7].clone()
            relative = IPose(pos, quat).invert().multiply(IPose(peg[:, :3]-env.scene.env_origins, peg[:, 3:7]))
            rows.append(dict(phase=np.full(n, phase), hand_pose=torch.cat((pos, quat), 1).cpu().numpy(),
                             object_pose=peg.cpu().numpy(), command=command.cpu().numpy(),
                             fingers=robot.data.joint_pos[:, fingers].cpu().numpy(),
                             relative_pose=relative.to_vec().cpu().numpy(), attached=attached.cpu().numpy().copy()))

    above = target.pos.clone()
    above[:, 2] += .04
    advance(0, above, .04, 120)
    advance(1, target.pos, .04, 144)
    advance(2, target.pos, .012, 48)
    pos, quat = env._compute_frame_pose('robot1', robot)
    peg = env.scene['peg'].data.root_state_w[:, :7]
    actual = IPose(pos, quat).invert().multiply(IPose(peg[:, :3]-env.scene.env_origins, peg[:, 3:7]))
    grasp_error = torch.linalg.vector_norm(actual.pos-offset.pos, dim=1)
    attached = grasp_error < .003
    for i, joint in enumerate(env.robot1_fixed_joints):
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*actual.pos[i].tolist()))
        joint.GetLocalRot0Attr().Set(Gf.Quatf(*actual.quat[i].tolist()))
        joint.GetJointEnabledAttr().Set(bool(attached[i]))
    effect = actual.to_vec().cpu().numpy().copy()
    advance(3, above, .012, 144)
    advance(4, above, .012, 48)
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    drift = np.linalg.norm(arrays['relative_pose'][-48:, :, :3]-effect[None, :, :3], axis=2).max(0)
    lift = arrays['object_pose'][-1, :, 2] - initial_object_z.cpu().numpy()
    success = attached.cpu().numpy() & (drift < .002) & (lift > .03)
    np.savez_compressed(args.output/'trajectories.npz', **arrays, contexts=contexts.cpu().numpy(),
                        effect=effect, success=success, grasp_error=grasp_error.cpu().numpy())
    summary = dict(seed=args.seed, scope=__doc__, trials_per_grasp=args.trials_per_grasp,
                   success_rates={name: float(success[i*args.trials_per_grasp:(i+1)*args.trials_per_grasp].mean())
                                  for i, name in enumerate(('GraspLeft', 'GraspRight'))},
                   grasp_error_m=grasp_error.tolist(), lift_m=lift.tolist(), hold_drift_m=drift.tolist(),
                   trajectory=str(args.output/'trajectories.npz'))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    env.close()
except Exception:
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
app.close()
