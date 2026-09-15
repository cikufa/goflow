"""Evaluate a real released-code Gears checkpoint with recorded episode traces."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common.runtime import kit_arguments

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=Path)
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--seed-base', type=int, default=10000)
parser.add_argument('--output', type=Path, default=ROOT / 'results/original_gears')
parser.add_argument('--video', action='store_true')
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--sampling', choices=('uniform', 'flow', 'nominal'), default='uniform')
parser.add_argument('--control', choices=('policy', 'zero', 'down'), default='policy',
                    help='zero/down are explicitly labeled physics diagnostics, never policy evaluation')
args = parser.parse_args()
if args.episodes < 1:
    parser.error('--episodes must be positive')
args.output.mkdir(parents=True, exist_ok=True)
sys.argv = sys.argv[:1]
local_kit_arguments = kit_arguments()

from omni.isaac.lab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=args.video, kit_args=local_kit_arguments).app
try:
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import yaml
    from goflow.environments.med_gear.direct_panda_position import MyPandaEnv, MyPandaEnvCfg, INITIAL_CFG, IPose
    from goflow.rl_components.my_models import ModelA2CContinuousLogStd
    from goflow.rl_components.my_network_builder import A2CBuilder
    from goflow.rl_components.my_a2c_common import NormFlowDist, UniformDist

    class EvaluationGears(MyPandaEnv):
        def _get_rewards(self):
            reward = super()._get_rewards()
            target = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).multiply(IPose.from_pose(INITIAL_CFG.HOLE_T_PEG_GOAL))
            self.last_goal_distance = torch.linalg.vector_norm(
                self.scene['peg'].data.root_pos_w - self.scene.env_origins - target.pos, dim=1
            ).detach().clone()
            return reward

    cfg = MyPandaEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed_base
    view_target = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).multiply(
        IPose.from_pose(INITIAL_CFG.HOLE_T_PEG_GOAL)).pos[0].cpu().tolist()
    cfg.viewer.lookat = tuple(view_target)
    cfg.viewer.eye = tuple(p + offset for p, offset in zip(view_target, (.4, .4, .35)))
    env = EvaluationGears(cfg, render_mode='rgb_array' if args.video else None)
    settings = yaml.safe_load((ROOT / 'goflow/environments/med_gear/agents/GOFLOW.yaml').read_text())['params']
    builder = A2CBuilder()
    builder.load(settings['network'])
    model = ModelA2CContinuousLogStd(builder).build({
        'actions_num': cfg.num_actions, 'input_shape': (cfg.num_observations,),
        'num_seqs': 1, 'value_size': 1, 'normalize_value': True, 'normalize_input': False,
    }).to('cuda:0')
    checkpoint = torch.load(args.checkpoint, map_location='cuda:0', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    bounds = list(cfg.dr_ranges.values())
    low, high = torch.tensor([b[0] for b in bounds]), torch.tensor([b[1] for b in bounds])
    flow = NormFlowDist(low, high, len(bounds))
    if 'goflow_distribution' not in checkpoint:
        raise ValueError('Checkpoint has no saved flow; cannot report learned-density checks.')
    flow.flow.load_state_dict(checkpoint['goflow_distribution'])
    class Nominal:
        def rsample(self, shape):
            return ((low + high) / 2).expand(*shape, len(bounds)).clone()
    env.set_sampling_dist({'uniform': UniformDist(low, high), 'flow': flow,
                           'nominal': Nominal()}[args.sampling])
    rows = []
    threshold = settings['config']['dr_method']['success_threshold']
    for episode in range(args.episodes):
        seed = args.seed_base + episode
        env.seed(seed)
        obs, _ = env.reset()
        if args.video:
            # Fill render buffers without advancing physics or policy state.
            for _ in range(8):
                env.sim.render()
        xi = env.context.detach().cpu().numpy()[0].copy()
        initial = {'robot_joints': env.robots['robot1'].data.joint_pos[0].tolist(),
                   'peg_root_state': env.scene['peg'].data.root_state_w[0].tolist(),
                   'policy_observation': obs['policy'][0].tolist()}
        observations, actions, rewards, peg_poses, joint_positions, values = [], [], [], [], [], []
        start = time.time()
        writer = None
        if args.video:
            video_dir = args.output / 'videos'
            video_dir.mkdir(exist_ok=True)
            writer = imageio.get_writer(str(video_dir / f'episode_{episode:03d}.mp4'), fps=24)
        done = False
        while not done:
            policy_obs = torch.clamp(obs['policy'], -5, 5)
            with torch.no_grad():
                out = model({'is_train': False, 'obs': policy_obs, 'rnn_states': None})
            action = torch.clamp(out['mus'] if args.deterministic else out['actions'], -1, 1)
            if args.control != 'policy':
                action = torch.zeros_like(action)
                if args.control == 'down':
                    action[:, 2] = -1
            observations.append(policy_obs.cpu().numpy()[0])
            actions.append(action.cpu().numpy()[0])
            values.append(float(out['values'][0]))
            peg_poses.append(env.scene['peg'].data.root_state_w[0, :7].cpu().numpy().copy())
            joint_positions.append(env.robots['robot1'].data.joint_pos[0].cpu().numpy().copy())
            if writer:
                writer.append_data(env.render())
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward[0]))
            done = bool(terminated[0] or truncated[0])
        if writer:
            writer.close()
        with torch.no_grad():
            log_p = float(flow.log_prob(torch.tensor(xi, device='cuda:0').unsqueeze(0))[0])
        row = {'episode': episode, 'seed': seed, 'xi': json.dumps(xi.tolist()),
               'initial_condition': json.dumps(initial), 'episode_reward': sum(rewards),
               'success': int(sum(rewards) >= threshold), 'success_definition': f'episode return >= {threshold}',
               'final_goal_distance_m': float(env.last_goal_distance[0]),
               'steps': len(rewards), 'duration_s': len(rewards) * env.step_dt,
               'wall_seconds': time.time() - start, 'log_p_phi': log_p,
               'checkpoint': str(args.checkpoint.resolve()), 'deterministic': args.deterministic,
               'sampling': args.sampling, 'control': args.control}
        rows.append(row)
        np.savez_compressed(args.output / f'episode_{episode:03d}.npz', xi=xi,
                            observations=observations, actions=actions, rewards=rewards,
                            peg_poses=peg_poses, joint_positions=joint_positions,
                            nonprivileged_values=values)
        print(json.dumps({k: v for k, v in row.items() if k != 'initial_condition'}), flush=True)
    with (args.output / 'episodes.csv').open('w') as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    summary = {'episodes': len(rows), 'success_rate': float(np.mean([r['success'] for r in rows])),
               'mean_episode_reward': float(np.mean([r['episode_reward'] for r in rows])),
               'mean_final_goal_distance_m': float(np.mean([r['final_goal_distance_m'] for r in rows])),
               'success_definition': f'upstream return >= {threshold}; not a physical seating certificate',
               'seeds': [r['seed'] for r in rows], 'checkpoint': str(args.checkpoint.resolve()),
               'training': checkpoint.get('instrumentation'), 'checkpoint_origin': 'locally trained from released code',
               'privileged_critic': False, 'yaw_randomization_applied': False,
               'sampling': args.sampling,
               'control': args.control,
               'episode_reset_note': 'Upstream pose-history reset behavior preserved; initial observations logged.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    env.close()
except Exception:
    # Kit's native fast shutdown can exit before a pending exception is printed.
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
app.close()
