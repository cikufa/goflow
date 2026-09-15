"""Recover episode returns and success from complete, ordered online rollouts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
parser.add_argument('--success-threshold', type=float, default=50)
args = parser.parse_args()
counts, returns, xi_initial, steps = None, None, None, None
total, rows = 0, []
resume = args.run / 'resume.json'
transition_offset = json.loads(resume.read_text()).get('transitions', 0) if resume.exists() else 0
discarded_reset_steps, previous_phase = 0, None
metrics = [json.loads(line) for line in (args.run / 'training.jsonl').read_text().splitlines()]
for rollout_index, path in enumerate(sorted((args.run / 'rollouts').glob('*.npz'))):
    with np.load(path) as data:
        phase = metrics[rollout_index]['validation']
        if previous_phase is not None and phase != previous_phase:
            # Released training loop explicitly resets every environment here.
            discarded_reset_steps += int(steps.sum())
            returns.fill(0)
            steps.fill(0)
        previous_phase = phase
        reward, done, xi = data['rewards'], data['dones'], data['xi']
        reward = reward.reshape(reward.shape[:2])
        done = done.reshape(done.shape[:2])
        if counts is None:
            n = reward.shape[1]
            counts, steps = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
            returns, xi_initial = np.zeros(n), xi[0].copy()
        for t in range(len(reward)):
            total += len(returns)
            starting = steps == 0
            xi_initial[starting] = xi[t, starting]
            returns += reward[t]
            steps += 1
            for env_id in np.flatnonzero(done[t]):
                rows.append({'environment': int(env_id), 'episode': int(counts[env_id]),
                             'transition': total + transition_offset, 'xi': json.dumps(xi_initial[env_id].tolist()),
                             'return': returns[env_id], 'steps': int(steps[env_id]),
                             'success': int(returns[env_id] >= args.success_threshold),
                             'validation_at_end': metrics[rollout_index]['validation']})
                counts[env_id] += 1
                returns[env_id] = 0
                steps[env_id] = 0
if not rows:
    raise SystemExit('No complete episodes; nothing to summarize')
with (args.run / 'training_episodes.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = {'transitions': total + transition_offset,
           'transitions_recorded_this_session': total, 'prior_transitions': transition_offset, 'complete_episodes': len(rows),
           'incomplete_episode_transitions': int(steps.sum()),
           'discarded_at_explicit_phase_reset': discarded_reset_steps,
           'success_threshold': args.success_threshold,
           'note': 'Incomplete episodes discarded at explicit train/validation resets; only complete returns included.'}
for phase, flag in [('training', False), ('validation', True)]:
    selected = [r for r in rows if r['validation_at_end'] == flag]
    summary[phase] = {'episodes': len(selected),
                      'success_rate': float(np.mean([r['success'] for r in selected])) if selected else None,
                      'mean_return': float(np.mean([r['return'] for r in selected])) if selected else None}
(args.run / 'rollout_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
