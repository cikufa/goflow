"""Audit actual connector checkpoints, transition counts and evaluation traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
parser.add_argument('--budget', type=int, required=True)
args = parser.parse_args()
run = args.run
counts = json.loads((run/'runtime.json').read_text())
assert counts['transitions'] == counts['training_transitions'] + counts['validation_transitions']
assert args.budget <= counts['transitions'] < args.budget + 32768
checkpoint = torch.load(run/'checkpoints/final.pth', map_location='cpu', weights_only=False)
assert checkpoint['instrumentation']['transitions'] == counts['transitions']
for name in ('model', 'assymetric_vf_nets', 'goflow_distribution'):
    assert all(torch.isfinite(value).all() for value in checkpoint[name].values())
rollouts = sorted((run/'rollouts').glob('*.npz'))
for path in (rollouts[0], rollouts[-1]):
    with np.load(path) as data:
        assert all(np.isfinite(data[key]).all() for key in data.files)
        assert data['observations'].shape[-1] == 105
        assert data['critic_observations'].shape[-1] == 109
        np.testing.assert_allclose(data['observations'], data['critic_observations'][..., :105])
        np.testing.assert_allclose(data['xi'], data['critic_observations'][..., 105:])
episodes = steps = 0
for directory in sorted((run/'evaluations').iterdir()):
    if not directory.is_dir():
        continue
    summary = json.loads((directory/'summary.json').read_text())
    assert summary['task'] in ('connector', 'connector_aligned')
    with (directory/'episodes.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == summary['episodes']
    for row in rows:
        with np.load(directory/f"episode_{int(row['episode']):03d}.npz") as data:
            assert data['xi'].shape == (4,)
            assert data['observations'].shape == (int(row['steps']), 105)
            assert all(np.isfinite(data[key]).all() for key in data.files)
            np.testing.assert_allclose(data['rewards'].sum(), float(row['episode_reward']), rtol=1e-5)
        episodes += 1
        steps += int(row['steps'])
result = dict(status='passed', transitions=counts['transitions'],
              held_out_episodes=episodes, held_out_steps=steps,
              scope='Finite weights/traces, input separation and accounting; no success threshold asserted',
              checkpoint_sha256=hashlib.sha256((run/'checkpoints/final.pth').read_bytes()).hexdigest())
(run/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
