"""Measure released flow loss scales offline; never update checkpoint weights.

Uses held-out uniform episodes, not the original online update minibatch.
Reports the release's normalized-density / physical-volume convention unchanged.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from goflow.rl_components.my_a2c_common import NormFlowDist

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
parser.add_argument('--milestone', type=int, default=2000000)
parser.add_argument('--initial-checkpoint', type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(123456)
checkpoint = torch.load(args.run/'checkpoints/final.pth', map_location='cuda:0', weights_only=False)
bounds = torch.tensor(list(checkpoint['handoff_alignment']['dr_ranges'].values()), device='cuda:0')
flow = NormFlowDist(bounds[:, 0], bounds[:, 1], 4)
flow.flow.load_state_dict(checkpoint['goflow_distribution'])
config = json.loads((args.run/'effective_config.json').read_text())['config']['dr_method']
contexts, returns = [], []
for path in sorted((args.run/'evaluations'/f'{args.milestone}_uniform').glob('episode_*.npz')):
    with np.load(path) as data:
        contexts.append(data['xi'])
        returns.append(data['rewards'].sum())
assert len(contexts) >= 100
xi = torch.tensor(np.array(contexts), device='cuda:0')
r = torch.tensor(returns, device='cuda:0')
r = (r-r.mean())/(r.std()+1e-8)
volume = (bounds[:, 1]-bounds[:, 0]).prod()
lp = flow.log_prob(xi)
uniform = bounds[:, 0]+torch.rand(10000, 4, device='cuda:0')*(bounds[:, 1]-bounds[:, 0])
ul = flow.log_prob(uniform)
with torch.no_grad():
    reference = flow.rsample((10000,))
    previous_lp = flow.log_prob(reference).detach()
losses = {
    'reward': volume*(r.detach()*lp*lp.exp()).mean(),
    'entropy': config['alpha']*volume*(ul.exp()*ul).mean(),
    'similarity_mc_at_identical_weights': config['beta']*(previous_lp-flow.log_prob(reference)).mean(),
}
metrics = {}
for name, loss in losses.items():
    gradients = torch.autograd.grad(loss, list(flow.flow.parameters()), retain_graph=True)
    norm = torch.sqrt(sum(g.square().sum() for g in gradients))
    metrics[name] = {'loss': float(loss), 'gradient_l2': float(norm)}
with torch.no_grad():
    initial = NormFlowDist(bounds[:, 0], bounds[:, 1], 4)
    initial.flow.load_state_dict(torch.load(args.initial_checkpoint, map_location='cuda:0', weights_only=False)['goflow_distribution'])
    old_lp = initial.log_prob(reference)
    density_change = {'log_density_correlation': float(np.corrcoef(old_lp.cpu(), previous_lp.cpu())[0, 1]),
                      'mean_absolute_log_density_change': float((old_lp-previous_lp).abs().mean())}
report = dict(physical_context_volume=float(volume), normalized_box_volume=flow.scale**4,
    omitted_affine_log_jacobian=float(torch.log(flow.scale/(bounds[:, 1]-bounds[:, 0])).sum()),
    alpha=config['alpha'], beta=config['beta'], loss_probe=metrics, density_change=density_change,
    scope='Offline held-out loss/gradient probe, not exact online updates or a causal intervention. Similarity loss is zero at identical weights; its finite-sample gradient need not be zero. No normalization, objective, optimizer or checkpoint was changed.')
(args.run/'flow_objective_audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
