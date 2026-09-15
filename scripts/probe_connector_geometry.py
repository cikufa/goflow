"""Six native geometry cases; oracle pose control only, no learned-policy claim."""
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
parser.add_argument('--output', type=Path, default=ROOT / 'results/custom_connector/geometry_probe')
parser.add_argument('--episode-seconds', type=float, default=6., help='Oracle probe duration, not a training configuration')
parser.add_argument('--fixture-distance-scale', type=float, default=1., help='Move the wall away as an explicit geometry control')
parser.add_argument('--case', type=int, choices=range(6), help='Run one of the six fixed cases')
parser.add_argument('--video', action='store_true')
args = parser.parse_args()
if args.episode_seconds <= 0 or args.fixture_distance_scale <= 0:
    parser.error('Duration and fixture distance scale must be positive')
args.output.mkdir(parents=True, exist_ok=True)
kit = kit_arguments()
from omni.isaac.lab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=args.video, kit_args=kit).app
try:
    import numpy as np
    import torch
    import omni.usd
    from pxr import UsdPhysics
    from experiments.connector_handoff.environment import ConnectorEnv, ConnectorEnvCfg, GRASP_OFFSET
    from goflow.environments.med_gear.direct_panda_position import INITIAL_CFG, IPose
    cfg = ConnectorEnvCfg()
    cfg.scene.num_envs = 6 if args.case is None else 1
    cfg.episode_length_s = args.episode_seconds
    cfg.fixture_distance_scale = args.fixture_distance_scale
    cfg.seed = 0
    env = ConnectorEnv(cfg, render_mode='rgb_array' if args.video else None)
    contexts = torch.tensor([[0., grasp, 0., side * torch.pi / 2]
                              for side in (0, -1, 1) for grasp in (-GRASP_OFFSET, GRASP_OFFSET)], device=env.device)
    if args.case is not None:
        contexts = contexts[args.case:args.case + 1]
    class FixedCases:
        def rsample(self, shape):
            assert shape[0] == len(contexts)
            return contexts.clone()
    env.set_sampling_dist(FixedCases())
    obs, _ = env.reset()
    assert obs['policy'].shape == (env.num_envs, 105)
    assert obs['critic'].shape == (env.num_envs, 109)
    torch.testing.assert_close(obs['critic'][:, :105], obs['policy'])
    torch.testing.assert_close(obs['critic'][:, 105:], env.context)
    joints = [UsdPhysics.FixedJoint(p) for p in omni.usd.get_context().get_stage().Traverse()
              if p.IsA(UsdPhysics.FixedJoint) and any(str(target).endswith('/peg')
                 for target in UsdPhysics.FixedJoint(p).GetBody1Rel().GetTargets())]
    assert len(joints) == env.num_envs
    for i, joint in enumerate(env.robot1_fixed_joints):
        assert [str(p) for p in joint.GetBody1Rel().GetTargets()] == [f'/World/envs/env_{i}/peg']
    writer = None
    if args.video:
        import imageio.v2 as imageio
        env.render()
        for _ in range(8):
            env.sim.render()
        writer = imageio.get_writer(str(args.output / 'probe.mp4'), fps=24)
    goal = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).multiply(IPose.from_pose(INITIAL_CFG.HOLE_T_PEG_GOAL))
    rows = []
    for step in range(env.max_episode_length):
        peg = env.scene['peg'].data.root_pos_w - env.scene.env_origins
        error = goal.pos - peg
        # Use the Gears diagnostic's coarse/proportional target increments;
        # this is an oracle geometry probe, with true pose feedback labeled above.
        delta = torch.clamp(10 * error, -.1, .1)
        delta[:, 2] = torch.minimum(delta[:, 2], torch.zeros_like(delta[:, 2]))
        delta[torch.linalg.vector_norm(error[:, :2], dim=1) > .001, 2] = 0.
        default = env._compute_default_action('robot1', env.robots['robot1'])[:, :3]
        command = (delta - default * env.default_action_scale) / env.trans_action_scale
        action = torch.clamp(command, -1., 1.)
        observation = obs['policy'].clone()
        if writer:
            writer.append_data(env.render())
        obs, reward, terminated, truncated, _ = env.step(action)
        rows.append({'observation': observation.cpu().numpy(), 'action': action.cpu().numpy(),
                     'commanded_action': command.cpu().numpy(),
                     'reward': reward.cpu().numpy(), 'gear_pose': env.last_peg_pose.cpu().numpy(),
                     'goal_distance': env.last_goal_distance.cpu().numpy(),
                     'terminated': terminated.cpu().numpy(), 'truncated': truncated.cpu().numpy()})
        if bool(torch.all(terminated | truncated)):
            break
    if writer:
        writer.close()
    arrays = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
    assert all(np.isfinite(a).all() for a in arrays.values())
    np.savez_compressed(args.output / 'trajectories.npz', contexts=contexts.cpu().numpy(), **arrays)
    summary = {'scope': 'Oracle pose controller for native geometry; no macro or policy competence claim',
               'seed': 0, 'steps': len(rows), 'episode_seconds': args.episode_seconds,
               'fixture_distance_scale': args.fixture_distance_scale,
               'case': args.case, 'video': args.video,
               'actor_width': 105, 'critic_width': 109,
               'contexts': contexts.tolist(),
               'returns': arrays['reward'].sum(0).tolist(),
               'final_goal_distance': arrays['goal_distance'][-1].tolist(),
               'minimum_goal_distance': arrays['goal_distance'].min(0).tolist(),
               'attachment_count': len(joints)}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    env.close()
except Exception:
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
app.close()
