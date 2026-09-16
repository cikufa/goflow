"""Held-out INSERT critic/density calibration and fixed-state Eq.7 slices."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from goflow.rl_components.my_a2c_common import NormFlowDist
from experiments.common.checkpoints import privileged_value_model
from experiments.common.belief_space import precondition_score

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
parser.add_argument('--milestone', type=int, default=5000000)
parser.add_argument('--prior-run', type=Path, help='Previous session in the same checkpoint lineage')
args = parser.parse_args()
out = args.run/'analysis'/str(args.milestone)
out.mkdir(parents=True, exist_ok=True)
evaluation = args.run/'evaluations'
source = json.loads((evaluation/f'{args.milestone}_flow/summary.json').read_text())
checkpoint = torch.load(source['checkpoint'], map_location='cuda:0', weights_only=False)
config = yaml.safe_load((ROOT/'experiments/original_gears/privileged_goflow.yaml').read_text())['params']['config']
low = torch.tensor([-.12, -.04, -.004, -np.pi])
high = -low
if "handoff_alignment" in checkpoint:
    bounds = list(checkpoint["handoff_alignment"]["dr_ranges"].values())
    low = torch.tensor([b[0] for b in bounds])
    high = torch.tensor([b[1] for b in bounds])
flow = NormFlowDist(low, high, 4)
flow.flow.load_state_dict(checkpoint['goflow_distribution'])
critic = privileged_value_model(checkpoint, config['central_value_config'])
torch.manual_seed(123456)
with torch.no_grad():
    reference = flow.rsample([10000])
    epsilon = float(torch.quantile(flow.log_prob(reference), .05).exp())
report = dict(checkpoint=source['checkpoint'], training=checkpoint['instrumentation'],
              density_threshold=epsilon, threshold_note='5th percentile of 10000 learned-flow samples; released normalized-coordinate density units; no reward tuning',
              success_definition='Released episode return >=50; goal distance reported independently', evaluation={})

def correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else None

def auc(value, success):
    if not success.any() or success.all():
        return None
    delta = value[success, None]-value[None, ~success]
    return float(((delta > 0)+.5*(delta == 0)).mean())

for sampling in ('uniform', 'flow'):
    folder = evaluation/f'{args.milestone}_{sampling}'
    with (folder/'episodes.csv').open() as f:
        rows = list(csv.DictReader(f))
    returns, discounted, values, density, distance, contexts = [], [], [], [], [], []
    for row in rows:
        with np.load(folder/f"episode_{int(row['episode']):03d}.npz") as data:
            reward = data['rewards']
            returns.append(reward.sum())
            discounted.append(reward @ config['gamma']**np.arange(len(reward)))
            values.append(data['privileged_values'][0])
            contexts.append(data['xi'])
            # Fixed observation for all counterfactual contexts in the plot.
            initial_observation = data['observations'][0].copy()
        density.append(float(row['log_p_phi']))
        distance.append(float(row['final_goal_distance_m']))
    returns, discounted, values, density, distance, contexts = map(np.asarray, (returns, discounted, values, density, distance, contexts))
    success = returns >= 50
    joint = precondition_score(values, density, np.ones(len(rows)), return_threshold=50, density_threshold=epsilon)
    accept = np.asarray(joint['value_ok']) & np.asarray(joint['density_ok'])
    metrics = dict(episodes=len(rows), successes=int(success.sum()), mean_return=float(returns.mean()),
                   mean_goal_distance_m=float(distance.mean()), within_3mm=int((distance < .003).sum()),
                   mean_value=float(values.mean()), mean_discounted_return=float(discounted.mean()),
                   value_rmse=float(np.sqrt(np.mean((values-discounted)**2))),
                   value_return_correlation=correlation(values, discounted), value_success_auc=auc(values, success),
                   density_return_correlation=correlation(density, returns),
                   precondition_accepted=int(accept.sum()),
                   precondition_precision=float(success[accept].mean()) if accept.any() else None,
                   precondition_recall=float(accept[success].mean()) if success.any() else None)
    report['evaluation'][sampling] = metrics
    np.savez_compressed(out/f'{sampling}.npz', returns=returns, discounted=discounted, values=values,
                        density=density, contexts=contexts, distance=distance, joint=accept)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(discounted, values, c=success)
    axes[0].set(xlabel='Discounted episode return', ylabel='Initial privileged value')
    axes[1].scatter(density, returns, c=success)
    axes[1].axhline(50, color='gray', ls='--')
    axes[1].set(xlabel='Log density', ylabel='Episode return')
    fig.savefig(out/f'{sampling}_calibration.png', dpi=150)
    plt.close(fig)

x, angle = np.meshgrid(np.linspace(float(low[1]), float(high[1]), 101), np.linspace(-np.pi, np.pi, 101))
xi = torch.tensor(np.column_stack((np.zeros(x.size), x.ravel(), np.zeros(x.size), angle.ravel())), dtype=torch.float32, device='cuda:0')
with torch.no_grad():
    obs = torch.tensor(initial_observation, device='cuda:0').expand(len(xi), -1)
    values = critic({'obs': torch.cat((obs, xi), 1), 'is_train': False})['values'].cpu().numpy().reshape(x.shape)
    lp = flow.log_prob(xi).cpu().numpy().reshape(x.shape)
joint = (values > 50) & (lp > np.log(epsilon))
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for ax, data, title in zip(axes, (lp, values, joint), ('Log density', 'Privileged value', 'Joint precondition')):
    im = ax.pcolormesh(x*1000, angle, data, shading='auto')
    fig.colorbar(im, ax=ax)
    ax.set(xlabel='Grasp x offset (mm)', ylabel='Fixture angle (rad)', title=title)
fig.suptitle('Fixed actor observation; yaw=0 and grasp y=0; JT=50')
fig.savefig(out/'precondition_slice.png', dpi=150)
np.savez_compressed(out/'precondition_slice.npz', x=x, fixture_angle=angle, values=values, log_density=lp,
                    joint=joint, observation=initial_observation)
report['slice_coverage'] = float(joint.mean())
paired = args.run/'grasp_fixture_cases'
if (paired/'episodes.csv').exists():
    with (paired/'episodes.csv').open() as f:
        paired_rows = list(csv.DictReader(f))
    # Do not accidentally associate another milestone's paired test with this report.
    paired_checkpoint = torch.load(paired_rows[0]['checkpoint'], map_location='cuda:0', weights_only=False)
    if all(torch.equal(weight, paired_checkpoint['model'][key]) for key, weight in checkpoint['model'].items()):
        report['paired_fixed_contexts'] = {}
        for case in range(4):
            rows = [row for row in paired_rows if int(row['grasp_fixture_case']) == case]
            report['paired_fixed_contexts'][str(case)] = dict(episodes=len(rows),
                successes=sum(int(row['success']) for row in rows),
                mean_return=float(np.mean([float(row['episode_reward']) for row in rows])),
                mean_goal_distance_m=float(np.mean([float(row['final_goal_distance_m']) for row in rows])))
report['limit'] = 'Calibration and fixed-state slices do not establish a sequential handoff or a belief-space planning result.'
metrics = [json.loads(line) for line in (args.run/'training.jsonl').read_text().splitlines()]
if args.prior_run:
    previous = [json.loads(line) for line in (args.prior_run/'training.jsonl').read_text().splitlines()]
    resume = json.loads((args.run/'resume.json').read_text())
    assert previous[-1]['transitions'] == resume['transitions']
    metrics = previous + metrics
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for validating, label in ((False, 'Flow training'), (True, 'Uniform validation')):
    rows = [row for row in metrics if row['validation'] == validating and 'phase_mean_return' in row]
    axes[0].plot([r['transitions'] for r in rows], [r['phase_mean_return'] for r in rows], label=label)
    axes[1].plot([r['transitions'] for r in rows], [r['phase_success_rate'] for r in rows], label=label)
for ax in axes:
    ax.set_xlabel('Actual transitions (PPO + validation)')
    ax.legend()
axes[0].set_ylabel('Phase mean return')
axes[1].set_ylabel('Phase success rate')
fig.savefig(out/'training.png', dpi=150)
(out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
