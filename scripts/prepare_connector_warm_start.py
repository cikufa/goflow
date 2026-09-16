"""Preserve learned actor/critic; reset flow normalization and optimizer state."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
weights = torch.load(args.source, map_location='cpu', weights_only=False)
prior = weights.get('instrumentation', {}).copy()
for key in ('goflow_distribution', 'goflow_distribution_optimizer', 'instrumentation',
            'central_value_optimizer', 'central_value_counters'):
    weights.pop(key, None)
weights['epoch'] = weights['frame'] = 0
weights['optimizer']['state'] = {}
weights['last_mean_rewards'] = weights['last_mean_target_rewards'] = -1e9
weights['warm_start_source'] = {
    'path': str(args.source.resolve()),
    'sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
    'inherited_counts': prior,
    'scope': 'Actor/critic weights and normalization retained; fresh optimizers and flow; additional counters start at zero.',
}
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(weights, args.output)
args.output.with_suffix('.json').write_text(json.dumps(weights['warm_start_source'], indent=2) + '\n')
