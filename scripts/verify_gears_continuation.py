"""Check recorded budgets, evaluation coverage, finite traces and saved weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
run = args.run
runtime = json.loads((run / 'runtime.json').read_text())
analysis = json.loads((run / 'analysis/5000000/summary.json').read_text())
assert runtime['training_transitions'] + runtime['validation_transitions'] == runtime['transitions'] == 5001216
assert all(x['successes'] == 0 for x in analysis['evaluation'].values())
assert all(p['score'] == 0 for x in analysis['evaluation'].values() for p in x['precondition'].values())
episodes = steps = 0
for directory in (run / 'evaluations').iterdir():
    if not directory.is_dir():
        continue
    with (directory / 'episodes.csv').open() as f:
        rows = list(csv.DictReader(f))
    episodes += len(rows)
    steps += sum(int(x['steps']) for x in rows)
    paths = list(directory.glob('episode_*.npz'))
    assert len(paths) == len(rows)
    for path in paths:
        with np.load(path) as data:
            assert np.isfinite(data['rewards']).all()
            assert np.isfinite(data['privileged_values']).all()
assert episodes == 328 and steps == 15416
with (run / 'privileged_critic_losses.csv').open() as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 752
assert int(rows[-1]['training_transition']) == runtime['training_transitions']
checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (run / 'checkpoints').glob('*.pth')}
final = torch.load(run / 'checkpoints/final.pth', map_location='cpu', weights_only=False)
milestone = torch.load(run / 'checkpoints/transitions_005001216.pth', map_location='cpu', weights_only=False)
# Separate torch.save archives have different filenames/timing metadata. Verify
# model equivalence, not byte identity; retain both exact hashes as provenance.
for name in ('model', 'assymetric_vf_nets', 'goflow_distribution'):
    assert final[name].keys() == milestone[name].keys()
    assert all(torch.equal(value, milestone[name][key]) for key, value in final[name].items())
assert final['instrumentation']['transitions'] == milestone['instrumentation']['transitions']
result = dict(status='passed', held_out_episodes=episodes, held_out_transitions=steps,
              actual_central_updates_this_session=len(rows), final_milestone_weights_equal=True,
              checkpoint_sha256=checksums)
(run / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
