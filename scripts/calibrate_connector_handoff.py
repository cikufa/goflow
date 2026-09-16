"""Calibrate actual grasp and staging states, then optionally execute an INSERT probe.

Uses the released gravity-disabled object and fixed-grasp approximation. This
validates macro execution and constrained hold, not frictional pickup quality.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common.runtime import kit_arguments

parser = argparse.ArgumentParser()
parser.add_argument('--trials-per-grasp', type=int, default=100)
parser.add_argument('--seed', type=int, default=61000)
parser.add_argument('--output', type=Path, default=ROOT/'results/custom_connector/final_handoff_experiment/calibration')
parser.add_argument('--video', action='store_true', help='Record native env_0 calibration video')
parser.add_argument('--fixture-distance-scale', type=float, default=1.)
parser.add_argument('--attach-before-close', action='store_true', help='Explicit fixed-grasp timing diagnostic: constrain measured approach pose before closure')
parser.add_argument('--stage-frame', choices=('hand','connector'), default='hand')
parser.add_argument('--stage-height', type=float, default=0., help='Common hand waypoint height above old INSERT reset, metres')
parser.add_argument('--stage-hold-steps', type=int, default=48)
parser.add_argument('--insert-controller', choices=('oracle','policy'))
parser.add_argument('--checkpoint', type=Path, default=ROOT/'results/custom_connector/insert_seed0_to10m/checkpoints/final.pth')
args = parser.parse_args()
if args.trials_per_grasp < 1:
    parser.error('--trials-per-grasp must be positive')
args.output.mkdir(parents=True, exist_ok=True)
if (args.output/'summary.json').exists():
    raise FileExistsError('Refusing to overwrite completed calibration')
kit = kit_arguments()
from omni.isaac.lab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=args.video, kit_args=kit).app
try:
    import numpy as np
    import torch
    from pxr import Gf
    from scipy.spatial.transform import Rotation
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.connector_handoff.environment import ConnectorEnv, ConnectorEnvCfg, GRASP_OFFSET
    from goflow.environments.med_gear.direct_panda_position import IPose, INITIAL_CFG

    cfg = ConnectorEnvCfg()
    cfg.scene.num_envs = 2 * args.trials_per_grasp
    cfg.seed = args.seed
    cfg.fixture_distance_scale = args.fixture_distance_scale
    env = ConnectorEnv(cfg, render_mode="rgb_array" if args.video else None)
    n = env.num_envs
    contexts = torch.zeros(n, 4, device=env.device)
    contexts[:, 3] = torch.tensor([-torch.pi/2, torch.pi/2], device=env.device).repeat((n+1)//2)[:n]
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
    # Both actions must start from the same visible-object placement domain.
    # INSERT's reset otherwise shifts the object with its sampled grasp offset.
    common_object = IPose.from_pose(INITIAL_CFG.WORLD_T_PEG_START).repeat(n)
    state = torch.cat((common_object.pos + env.scene.env_origins, common_object.quat), 1)
    state[:, :3] += jitter
    env.scene['peg'].write_root_pose_to_sim(state)
    env.scene['peg'].write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device))
    offset = IPose.from_pose(INITIAL_CFG.PEG_T_TOOL).invert().repeat(n)
    offset.pos[:, 0] += contexts[:, 1]
    visible_object = IPose(state[:, :3] - env.scene.env_origins, state[:, 3:7])
    target = visible_object.multiply(offset.invert())
    nominal_hand = IPose.from_pose(INITIAL_CFG.WORLD_T_PEG_START).multiply(
        IPose.from_pose(INITIAL_CFG.PEG_T_TOOL)).repeat(n)
    writer = None
    if args.video:
        import imageio.v2 as imageio
        env.render()
        for _ in range(8): env.sim.render()
        writer = imageio.get_writer(str(args.output/'handoff.mp4'),fps=24)
    rows = []
    attached = torch.zeros(n, dtype=torch.bool, device=env.device)
    initial_object_z = state[:, 2].clone()

    def advance(phase, target_pos, finger_width, steps):
        for _ in range(steps):
            if writer: writer.append_data(env.render())
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
                             relative_pose=relative.to_vec().cpu().numpy(), attached=attached.cpu().numpy().copy(),
                             joint_positions=robot.data.joint_pos.cpu().numpy().copy(),
                             joint_targets=robot.data.joint_pos_target.cpu().numpy().copy(),
                             joint_velocities=robot.data.joint_vel.cpu().numpy().copy(),
                             object_velocity=env.scene['peg'].data.root_state_w[:, 7:13].cpu().numpy().copy()))

    above = target.pos.clone()
    above[:, 2] += .04
    advance(0, above, .04, 120)
    advance(1, target.pos, .04, 144)
    if not args.attach_before_close:
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
    if args.attach_before_close:
        advance(2, target.pos, .012, 48)
    advance(3, above, .012, 144)
    advance(4, above, .012, 48)
    grasp_terminal = {key: value.copy() for key, value in rows[-1].items()}
    # A common hand waypoint, independent of grasp side and fixture state.
    # Keep measured hand orientation fixed: transport never corrects object yaw.
    target.quat = torch.tensor(grasp_terminal['hand_pose'][:, 3:7], device=env.device)
    if args.stage_frame == 'connector':
        from omni.isaac.lab.utils.math import quat_apply
        desired_connector = IPose.from_pose(INITIAL_CFG.WORLD_T_PEG_START).repeat(n).pos
        nominal_hand.pos = desired_connector - quat_apply(target.quat, torch.tensor(grasp_terminal['relative_pose'][:, :3],device=env.device))
    nominal_hand.pos[:, 2] += args.stage_height
    stage_above = nominal_hand.pos.clone()
    stage_above[:, 2] += .04
    advance(5, stage_above, .012, 144)
    transport_terminal = {key: value.copy() for key, value in rows[-1].items()}
    advance(6, nominal_hand.pos, .012, 144)
    advance(7, nominal_hand.pos, .012, args.stage_hold_steps)
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    assert all(np.isfinite(value).all() for value in arrays.values())
    hold = arrays['relative_pose'][arrays['phase'][:, 0] == 4]
    drift = np.linalg.norm(hold[:, :, :3]-effect[None, :, :3], axis=2).max(0)
    lift = grasp_terminal['object_pose'][:, 2] - initial_object_z.cpu().numpy()
    success = attached.cpu().numpy() & (drift < .002) & (lift > .03)
    stage_error = np.linalg.norm(arrays['hand_pose'][-1, :, :3]-nominal_hand.pos.cpu().numpy(), axis=1)
    transform_drift = np.linalg.norm(arrays['relative_pose'][-1, :, :3]-grasp_terminal['relative_pose'][:, :3], axis=1)
    def angles(pose):
        return Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_euler('xyz')
    def angular_distance(a, b):
        return (Rotation.from_quat(a[:, [4,5,6,3]]) * Rotation.from_quat(b[:, [4,5,6,3]]).inv()).magnitude()
    terminal_angles = angles(grasp_terminal['relative_pose'])
    angular_drift = angular_distance(arrays['relative_pose'][-1], grasp_terminal['relative_pose'])
    stage_success = success & (stage_error < .003) & (transform_drift < .002) & (angular_drift < .02)
    records, consistency = [], []
    for i in range(n):
        side = 'GraspLeft' if i < args.trials_per_grasp else 'GraspRight'
        r = dict(trial=i, seed=args.seed, grasp=side, fixture_angle=float(contexts[i,3]),
                 success=int(success[i]), lift_m=float(lift[i]), hold_drift_m=float(drift[i]),
                 grasp_error_m=float(grasp_error[i]))
        for prefix, vector, labels in (
            ('connector_world', grasp_terminal['object_pose'][i], ('x','y','z','qw','qx','qy','qz')),
            ('hand_local', grasp_terminal['hand_pose'][i], ('x','y','z','qw','qx','qy','qz')),
            ('hand_T_connector', grasp_terminal['relative_pose'][i], ('x','y','z','qw','qx','qy','qz')),
            ('relative', terminal_angles[i], ('roll','pitch','yaw')),
            ('joint', grasp_terminal['joint_positions'][i], range(9)),
            ('connector_velocity', grasp_terminal['object_velocity'][i], ('vx','vy','vz','wx','wy','wz')),
            ('initial_jitter', jitter[i].cpu().numpy(), ('x','y','z'))):
            r.update({f'{prefix}_{label}': float(value) for label,value in zip(labels,vector)})
        r.update({f'hand_world_{axis}':float(grasp_terminal['hand_pose'][i,j]+env.scene.env_origins[i,j]) for j,axis in enumerate(('x','y','z'))})
        records.append(r)
        c = dict(trial=i, grasp=side, grasp_success=int(success[i]), stage_success=int(stage_success[i]),
                 stage_hand_error_m=float(stage_error[i]), translation_drift_m=float(transform_drift[i]),
                 rotation_drift_rad=float(angular_drift[i]))
        for phase, data in [('after_grasp',grasp_terminal), ('after_transport',transport_terminal), ('insert_initial',rows[-1])]:
            c.update({f'{phase}_{label}':float(v) for label,v in zip(('x','y','z','qw','qx','qy','qz'),data['relative_pose'][i])})
        consistency.append(c)
    for name, data in [('grasp_terminal_states', records), ('handoff_transform_consistency', consistency)]:
        with (args.output/f'{name}.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    metrics = np.column_stack((grasp_terminal['relative_pose'][:, :3], terminal_angles))
    statistics = {}
    for side, sl in [('GraspLeft',slice(0,args.trials_per_grasp)), ('GraspRight',slice(args.trials_per_grasp,n))]:
        values = metrics[sl][success[sl]]
        statistics[side] = {name: dict(mean=float(v.mean()), std=float(v.std()),
                                     percentiles_5_50_95=np.percentile(v,[5,50,95]).tolist())
                            for name,v in zip(('x','y','z','roll','pitch','yaw'),values.T)} if len(values) else {}
        statistics[side]['correlation_xyz_rpy'] = np.corrcoef(values.T).tolist() if len(values)>1 else None
    fig, axes = plt.subplots(1,3,figsize=(12,3.5))
    for side, sl in [('GraspLeft',slice(0,args.trials_per_grasp)), ('GraspRight',slice(args.trials_per_grasp,n))]:
        for ax, idx, name in zip(axes,(0,1,5),('x (m)','y (m)','yaw (rad)')):
            ax.hist(metrics[sl,idx],bins=15,alpha=.6,label=side); ax.set_xlabel(name)
    axes[0].legend(); fig.tight_layout(); fig.savefig(args.output/'grasp_distributions.png'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(9,4))
    for ax, side in zip(axes,statistics):
        ax.imshow(statistics[side]['correlation_xyz_rpy'],vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(6),('x','y','z','roll','pitch','yaw'),rotation=45)
        ax.set_yticks(range(6),('x','y','z','roll','pitch','yaw')); ax.set_title(side)
    fig.tight_layout(); fig.savefig(args.output/'grasp_correlations.png'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(9,3.5))
    axes[0].plot(transform_drift*1000,'.'); axes[0].set_ylabel('Grasp to stage translation drift (mm)')
    axes[1].plot(angular_drift*180/np.pi,'.'); axes[1].set_ylabel('Rotation drift (degrees)')
    for ax in axes: ax.set_xlabel('Trial')
    fig.tight_layout(); fig.savefig(args.output/'handoff_consistency.png'); plt.close(fig)
    np.savez_compressed(args.output/'trajectories.npz', **arrays, contexts=contexts.cpu().numpy(),
                        effect=effect, success=success, stage_success=stage_success,
                        origins=env.scene.env_origins.cpu().numpy(), grasp_error=grasp_error.cpu().numpy())
    summary = dict(seed=args.seed, attach_before_close=args.attach_before_close, fixture_distance_scale=args.fixture_distance_scale, stage_hold_steps=args.stage_hold_steps, stage_height=args.stage_height, stage_frame=args.stage_frame, scope=__doc__, trials_per_grasp=args.trials_per_grasp,
                   perturbation='Independent uniform +/-3 mm planar visible placement; nominal yaw, same existing macro domain',
                   success_rates={name: float(success[i*args.trials_per_grasp:(i+1)*args.trials_per_grasp].mean())
                                  for i, name in enumerate(('GraspLeft', 'GraspRight'))},
                   stage_success_rates={name: float(stage_success[i*args.trials_per_grasp:(i+1)*args.trials_per_grasp].mean())
                                  for i, name in enumerate(('GraspLeft', 'GraspRight'))},
                   statistics=statistics, max_stage_error_m=float(stage_error.max()),
                   max_transform_translation_drift_m=float(transform_drift.max()),
                   max_transform_rotation_drift_rad=float(angular_drift.max()),
                   trajectory=str(args.output/'trajectories.npz'))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    if args.insert_controller:
        # Start a fresh local control horizon without resetting any physical state.
        env.episode_length_buf.zero_()
        env.pose_history.clear()
        hand_pos, hand_quat = env._compute_frame_pose('robot1', robot)
        env.fixed_quat['robot1'] = hand_quat.clone()
        env.prev_pose['robot1'] = (hand_pos.clone(),hand_quat.clone())
        relative = rows[-1]['relative_pose']
        env.context[:,0] = torch.tensor(angles(relative)[:,2],device=env.device)
        env.context[:,1:3] = torch.tensor(relative[:,:2],device=env.device)
        obs = env._get_observations()
        initial_context = env.context.clone()
        np.savez_compressed(args.output/'insert_initial_states.npz',
            robot_joint_pos=robot.data.joint_pos.cpu().numpy(),
            robot_joint_vel=robot.data.joint_vel.cpu().numpy(),
            robot_joint_targets=robot.data.joint_pos_target.cpu().numpy(), constraint_pose=effect,
            peg_root_state=env.scene['peg'].data.root_state_w.cpu().numpy(),
            hand_pose=torch.cat((hand_pos,hand_quat),1).cpu().numpy(),
            relative_pose=relative, context=initial_context.cpu().numpy(),
            origins=env.scene.env_origins.cpu().numpy(), observation=obs['policy'].cpu().numpy())
        if args.insert_controller == 'policy':
            import yaml
            from goflow.rl_components.my_models import ModelA2CContinuousLogStd
            from goflow.rl_components.my_network_builder import A2CBuilder
            from experiments.common.checkpoints import privileged_value_model
            settings=yaml.safe_load((ROOT/'experiments/original_gears/privileged_goflow.yaml').read_text())['params']
            builder=A2CBuilder(); builder.load(settings['network'])
            model=ModelA2CContinuousLogStd(builder).build(dict(actions_num=3,input_shape=(105,),num_seqs=n,
                value_size=1,normalize_value=True,normalize_input=False)).to(env.device)
            checkpoint=torch.load(args.checkpoint,map_location=env.device,weights_only=False)
            model.load_state_dict(checkpoint['model']); model.eval()
            critic=privileged_value_model(checkpoint,settings['config']['central_value_config'])
            from goflow.rl_components.my_a2c_common import NormFlowDist
            bounds=list(checkpoint.get('handoff_alignment',{}).get('dr_ranges',cfg.dr_ranges).values())
            flow=NormFlowDist(torch.tensor([b[0] for b in bounds]),torch.tensor([b[1] for b in bounds]),4)
            flow.flow.load_state_dict(checkpoint['goflow_distribution'])
            with torch.no_grad(): initial_log_density=flow.log_prob(initial_context).cpu().numpy()
        goal=IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).multiply(IPose.from_pose(INITIAL_CFG.HOLE_T_PEG_GOAL))
        insert_rows=[]
        for step in range(env.max_episode_length):
            if writer: writer.append_data(env.render())
            observation=obs['policy'].clone()
            if args.insert_controller == 'oracle':
                error=goal.pos-(env.scene['peg'].data.root_pos_w-env.scene.env_origins)
                delta=torch.clamp(10*error,-.1,.1)
                delta[:,2]=torch.minimum(delta[:,2],torch.zeros_like(delta[:,2]))
                delta[torch.linalg.vector_norm(error[:,:2],dim=1)>.001,2]=0
                default=env._compute_default_action('robot1',robot)[:,:3]
                command=(delta-default*env.default_action_scale)/env.trans_action_scale
                value=torch.full((n,),float('nan'),device=env.device)
            else:
                with torch.no_grad():
                    command=model(dict(is_train=False,obs=observation.clamp(-5,5),rnn_states=None))['actions']
                    value=critic(dict(obs=obs['critic'].clamp(-5,5),is_train=False))['values'].flatten()
            action=command.clamp(-1,1)
            obs,reward,terminated,truncated,_=env.step(action)
            insert_rows.append(dict(observation=observation.cpu().numpy(),command=command.cpu().numpy(),
                action=action.cpu().numpy(),reward=reward.cpu().numpy(),goal_distance=env.last_goal_distance.cpu().numpy(),
                connector_pose=env.last_peg_pose.cpu().numpy(),value=value.cpu().numpy(),
                terminated=terminated.cpu().numpy(),truncated=truncated.cpu().numpy()))
            if bool((terminated|truncated).all()): break
        ia={k:np.stack([r[k] for r in insert_rows]) for k in insert_rows[0]}
        np.savez_compressed(args.output/'insert_trajectories.npz',**ia,context=initial_context.cpu().numpy())
        returns=ia['reward'].sum(0); insertion=returns>=50
        matrix=[]
        for fixture_sign in (-1,1):
            for grasp_sign in (-1,1):
                mask=(contexts[:,3].cpu().numpy()*fixture_sign>0)&(contexts[:,1].cpu().numpy()*grasp_sign>0)
                record=dict(fixture_angle=fixture_sign*np.pi/2,grasp='GraspLeft' if grasp_sign<0 else 'GraspRight',
                    trials=int(mask.sum()),grasp_success=float(success[mask].mean()),stage_success=float(stage_success[mask].mean()),
                    insert_success=float(insertion[mask].mean()),end_to_end=float((success&stage_success&insertion)[mask].mean()),
                    mean_return=float(returns[mask].mean()),mean_final_distance_m=float(ia['goal_distance'][-1,mask].mean()))
                if args.insert_controller == 'policy':
                    record.update(mean_value=float(ia['value'][0,mask].mean()),mean_density=float(np.exp(initial_log_density[mask]).mean()))
                matrix.append(record)
        (args.output/'physics_matrix.json').write_text(json.dumps(dict(controller=args.insert_controller,matrix=matrix,
            checkpoint=str(args.checkpoint) if args.insert_controller=='policy' else None),indent=2)+'\n')
        print(json.dumps(matrix,indent=2),flush=True)
    if writer: writer.close()
    env.close()
except Exception:
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
app.close()
