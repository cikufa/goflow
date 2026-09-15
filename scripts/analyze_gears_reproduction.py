"""Measured policy, density, privileged-value and Eq.7 checks for one checkpoint."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from goflow.rl_components.my_a2c_common import NormFlowDist
from goflow.rl_components.my_models import ModelA2CContinuousLogStd
from goflow.rl_components.my_network_builder import A2CBuilder
from experiments.common.checkpoints import privileged_value_model
from experiments.common.belief_space import precondition_score

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
parser.add_argument('--milestone', type=int, default=5000000)
parser.add_argument('--reference-evaluation', type=Path,
                    help='Optional same-policy evaluation under the initial flow')
parser.add_argument('--initial-flow-checkpoint', type=Path, default=ROOT /
                    'results/original_gears/privileged_training_64/checkpoints/transitions_000200704.pth')
args = parser.parse_args()
output = args.run / 'analysis' / str(args.milestone)
output.mkdir(parents=True, exist_ok=True)
uniform_dir = args.run / 'evaluations' / f'{args.milestone}_uniform'
summary = json.loads((uniform_dir / 'summary.json').read_text())
checkpoint = torch.load(summary['checkpoint'], map_location='cuda:0', weights_only=False)
params = yaml.safe_load((ROOT / 'experiments/original_gears/privileged_goflow.yaml').read_text())['params']
config = params['config']
builder = A2CBuilder()
builder.load(params['network'])
actor = ModelA2CContinuousLogStd(builder).build({
    'actions_num': 3, 'input_shape': (105,), 'num_seqs': 1, 'value_size': 1,
    'normalize_value': config['normalize_value'], 'normalize_input': config['normalize_input'],
}).to('cuda:0')
actor.load_state_dict(checkpoint['model'])
actor.eval()
flow = NormFlowDist(torch.tensor([-torch.pi, -.02, -.02]), torch.tensor([torch.pi, .02, .02]), 3)
flow.flow.load_state_dict(checkpoint['goflow_distribution'])
initial_checkpoint = torch.load(args.initial_flow_checkpoint, map_location='cuda:0', weights_only=False)
assert initial_checkpoint['instrumentation']['completed_flow_updates'] == 0
initial_flow = NormFlowDist(flow.low, flow.high, 3)
initial_flow.flow.load_state_dict(initial_checkpoint['goflow_distribution'])
critic = privileged_value_model(checkpoint, config['central_value_config'])
torch.manual_seed(123456)
with torch.no_grad():
    reference = flow.rsample([10000])
    reference_lp = flow.log_prob(reference).cpu().numpy()
# Density thresholds use learned-distribution samples, never evaluation rewards.
thresholds = {str(q): float(np.exp(np.quantile(reference_lp, q))) for q in (.01, .05, .10)}

def correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else None

def auc(scores, success):
    positive, negative = scores[success], scores[~success]
    if not len(positive) or not len(negative):
        return None
    difference = positive[:, None] - negative[None, :]
    return float(np.mean((difference > 0) + .5 * (difference == 0)))

report = {'checkpoint': summary['checkpoint'], 'training': checkpoint['instrumentation'],
          'success_definition': 'released episode return >= 50',
          'initial_flow_checkpoint': str(args.initial_flow_checkpoint),
          'flow_parameter_l2_change': float(torch.sqrt(sum(
              torch.sum((v - initial_checkpoint['goflow_distribution'][k])**2)
              for k, v in checkpoint['goflow_distribution'].items() if v.is_floating_point()))),
          'density_thresholds_by_reference_quantile': thresholds,
          'threshold_note': 'Explicit construction assumption: lower 1/5/10 percentiles of 10000 learned-flow log densities; no held-out reward tuning. Upstream normalized-coordinate density units.',
          'value_note': 'Initial denormalized privileged value compared with gamma=.99 finite-episode discounted return, and with released JT=50 for preconditions. These are distinct metrics.',
          'flow_reference_context_mean': reference.mean(0).tolist(),
          'flow_reference_context_std': reference.std(0).tolist(),
          'flow_reference_log_density_range': [float(reference_lp.min()), float(reference_lp.max())],
          'evaluation': {}}
all_rows = []
for sampling in ('uniform', 'flow'):
    directory = args.run / 'evaluations' / f'{args.milestone}_{sampling}'
    with (directory / 'episodes.csv').open() as f:
        episodes = list(csv.DictReader(f))
    returns, values, densities, discounted, contexts, actions = [], [], [], [], [], []
    means, sigmas = [], []
    for row in episodes:
        episode = int(row['episode'])
        with np.load(directory / f'episode_{episode:03d}.npz') as data:
            rewards = data['rewards']
            value = float(data['privileged_values'][0])
            xi = data['xi']
            initial_obs = data['observations'][0]
            actions.append(data['actions'].copy())
            with torch.no_grad():
                prediction = actor({'obs': torch.tensor(data['observations'], device='cuda:0'),
                                    'is_train': False, 'rnn_states': None})
                means.append(prediction['mus'].cpu().numpy())
                sigmas.append(prediction['sigmas'].cpu().numpy())
        ret = float(rewards.sum())
        mc = float(rewards @ config['gamma'] ** np.arange(len(rewards)))
        lp = float(row['log_p_phi'])
        returns.append(ret); values.append(value); densities.append(lp)
        discounted.append(mc); contexts.append(xi)
        all_rows.append(dict(sampling=sampling, episode=episode, seed=int(row['seed']),
                             return_=ret, discounted_return=mc, initial_value=value,
                             log_density=lp, success=int(ret >= 50), xi=json.dumps(xi.tolist())))
    returns, values, densities, discounted = map(np.asarray, (returns, values, densities, discounted))
    success = returns >= 50
    actions = np.concatenate(actions)
    with torch.no_grad():
        initial_lp = initial_flow.log_prob(torch.tensor(np.asarray(contexts), device='cuda:0')).cpu().numpy()
    construction = {}
    for quantile, epsilon in thresholds.items():
        result = precondition_score(values, densities, np.ones(len(values)),
                                    return_threshold=50, density_threshold=epsilon)
        predicted = np.asarray(result['value_ok']) & np.asarray(result['density_ok'])
        result.update(accepted_episodes=int(predicted.sum()),
                      success_precision=float(success[predicted].mean()) if predicted.any() else None,
                      success_recall=float(predicted[success].mean()) if success.any() else None,
                      false_accepts=int(np.sum(predicted & ~success)),
                      false_rejects=int(np.sum(~predicted & success)))
        construction[quantile] = result
    report['evaluation'][sampling] = dict(episodes=len(episodes), successes=int(success.sum()),
        success_rate=float(success.mean()), mean_return=float(returns.mean()),
        mean_executed_residual=actions.mean(0).tolist(),
        mean_unclipped_policy_mu=np.concatenate(means).mean(0).tolist(),
        mean_policy_sigma=np.concatenate(sigmas).mean(0).tolist(),
        residual_upper_clip_fraction=(actions >= .9999).mean(0).tolist(),
        residual_lower_clip_fraction=(actions <= -.9999).mean(0).tolist(),
        mean_initial_value=float(values.mean()), mean_discounted_return=float(discounted.mean()),
        value_mc_rmse=float(np.sqrt(np.mean((values - discounted)**2))),
        value_mc_correlation=correlation(values, discounted),
        density_return_correlation=correlation(densities, returns),
        initial_density_return_correlation=correlation(initial_lp, returns),
        mean_absolute_log_density_change=float(np.mean(np.abs(densities - initial_lp))),
        density_success_auc=auc(densities, success), value_success_auc=auc(values, success),
        precondition=construction)
    np.savez_compressed(output / f'{sampling}_calibration.npz', returns=returns, values=values,
                        discounted_returns=discounted, log_density=densities, contexts=contexts)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(discounted, values, c=success, vmin=0, vmax=1)
    axes[0].set(xlabel='Observed discounted episode return', ylabel='Initial privileged value')
    axes[1].scatter(densities, returns, c=success, vmin=0, vmax=1)
    axes[1].axhline(50, color='gray', ls='--')
    axes[1].set(xlabel='Learned log density (released units)', ylabel='Episode return')
    fig.suptitle(f'{sampling}: held-out seed episodes, stochastic released policy')
    fig.savefig(output / f'{sampling}_calibration.png', dpi=150)
    plt.close(fig)

# Same fixed actor state in all three slices, as in paper-style V(s,xi) construction.
grid = np.linspace(-.02, .02, 101)
x, y = np.meshgrid(grid, grid)
xi = torch.tensor(np.column_stack((np.zeros(x.size), x.ravel(), y.ravel())), dtype=torch.float32, device='cuda:0')
with torch.no_grad():
    lp = flow.log_prob(xi).cpu().numpy().reshape(x.shape)
    obs = torch.tensor(initial_obs, device='cuda:0').expand(len(xi), -1)
    value = critic({'obs': torch.cat((obs, xi), dim=1), 'is_train': False})['values'].cpu().numpy().reshape(x.shape)
mask = (value > 50) & (lp > np.log(thresholds['0.05']))
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for ax, z, title in zip(axes, (lp, value, mask), ('log density', 'Privileged value', 'Joint precondition indicator')):
    im = ax.pcolormesh(x*1000, y*1000, z, shading='auto',
                       **({'vmin': 0, 'vmax': 1} if z.dtype == bool else {}))
    fig.colorbar(im, ax=ax)
    ax.set(xlabel='x offset (mm)', ylabel='y offset (mm)', title=title)
fig.suptitle('Yaw=0, fixed recorded actor observation; JT=50; density reference quantile=5%')
fig.savefig(output / 'precondition_slice.png', dpi=150)
np.savez_compressed(output / 'precondition_slice.npz', x=x, y=y, log_density=lp, value=value,
                    joint_indicator=mask, observation=initial_obs)
report['slice_joint_coverage'] = float(mask.mean())
if args.reference_evaluation:
    reference_summary = json.loads((args.reference_evaluation / 'summary.json').read_text())
    reference_policy = torch.load(reference_summary['checkpoint'], map_location='cuda:0', weights_only=False)
    assert all(torch.equal(value, reference_policy['model'][key]) for key, value in checkpoint['model'].items()), \
        'Reference evaluation must use the identical actor'
    assert Path(reference_summary['sampling_flow_checkpoint']).resolve() == args.initial_flow_checkpoint.resolve()
    learned_summary = json.loads((directory / 'summary.json').read_text())
    assert reference_summary['seeds'] == learned_summary['seeds'], 'Use matching episode seeds'
    report['initial_flow_evaluation'] = {
        'path': str(args.reference_evaluation), 'episodes': reference_summary['episodes'],
        'success_rate': reference_summary['success_rate'],
        'mean_return': reference_summary['mean_episode_reward'],
        'learned_minus_initial_success_rate': learned_summary['success_rate'] - reference_summary['success_rate'],
        'learned_minus_initial_mean_return': learned_summary['mean_episode_reward'] - reference_summary['mean_episode_reward'],
        'scope': 'Same final actor and seed set under different context distributions; no method retraining or causal curriculum ablation.'}
report['slice_observation_source'] = str(directory / f'episode_{episode:03d}.npz')
report['interpretation_limit'] = 'Nonuniform density and context-sensitive values alone do not establish useful skill preconditions. Require held-out success and calibration; null AUC means only one observed outcome class.'
(output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
with (output / 'episodes.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
    writer.writeheader(); writer.writerows(all_rows)
print(json.dumps({k: v for k, v in report.items() if k != 'evaluation'}, indent=2))
for sampling, metrics in report['evaluation'].items():
    print(sampling, json.dumps({k: v for k, v in metrics.items() if k != 'precondition'}))

lines = [f'# Gears checkpoint at {args.milestone:,} transitions', '',
         f"Checkpoint: `{summary['checkpoint']}`", '',
         'Success uses the released episode-return threshold of 50.', '',
         '| Context sampling | Successes | Mean return | Initial V | Discounted return | V RMSE |',
         '|---|---:|---:|---:|---:|---:|']
for sampling, m in report['evaluation'].items():
    lines.append(f"| {sampling} | {m['successes']}/{m['episodes']} | {m['mean_return']:.4f} | "
                 f"{m['mean_initial_value']:.4f} | {m['mean_discounted_return']:.4f} | {m['value_mc_rmse']:.4f} |")
lines += ['', '## Training lineage', '',
          f"Seed {report['training']['seed']}; {report['training']['transitions']:,} cumulative control transitions: "
          f"{report['training']['training_transitions']:,} PPO and {report['training']['validation_transitions']:,} validation.",
          f"Cumulative agent wall time {report['training']['wall_seconds']:.3f} s; "
          f"this continuation {report['training'].get('session_wall_seconds', 0):.3f} s. "
          f"Completed flow updates: {report['training']['completed_flow_updates']}.", '',
          '## Density, critic and precondition', '',
          'Density thresholds are the lower 1%, 5%, 10% quantiles of 10,000 learned-density samples, '
          'chosen without evaluation-return tuning. This is an explicit construction assumption; '
          'the release does not provide a calibrated task-specific density threshold.', '']
for sampling, m in report['evaluation'].items():
    scores = {q: result['score'] for q, result in m['precondition'].items()}
    lines += [f"- {sampling}: learned density/return correlation {m['density_return_correlation']}; "
              f"initial density/return correlation {m['initial_density_return_correlation']}. "
              f"Joint precondition belief scores by quantile: {scores}.",
              f"- {sampling}: mean raw action {m['mean_unclipped_policy_mu']}, "
              f"sigma {m['mean_policy_sigma']}, upper-clipped fractions {m['residual_upper_clip_fraction']}."]
lines += ['', 'The critic checks distinguish low-return calibration from successful skill learning. '
          'Null success AUC/precision/recall values in summary.json denote unavailable outcome classes or empty acceptance sets, not zero-valued estimates.', '',
          '## Artifacts', '',
          '- [Machine-readable summary](summary.json)',
          '- [Per-episode calibration table](episodes.csv)',
          '- [Uniform calibration](uniform_calibration.png)',
          '- [Flow calibration](flow_calibration.png)',
          '- [Density/value/joint precondition slice](precondition_slice.png)', '',
          'Full per-step trajectories and seeds are in ../../evaluations/. '
          'Configuration diagnosis and execution commands are tracked in docs/gears_reproduction_diagnosis.md '
          'and docs/gears_single_seed_continuation.md.']
if 'initial_flow_evaluation' in report:
    r = report['initial_flow_evaluation']
    lines += ['', '## Initial-density reference', '',
              f"The identical final actor under the initial flow achieves {r['success_rate']:.1%} success "
              f"and mean return {r['mean_return']:.3f} over {r['episodes']} episodes. "
              f"Learned minus initial: {r['learned_minus_initial_success_rate']:+.1%} success and "
              f"{r['learned_minus_initial_mean_return']:+.3f} return. Same recorded episode seeds; "
              'this compares sampling regions, not the causal effect of the training curriculum.']
(output / 'report.md').write_text('\n'.join(lines) + '\n')
