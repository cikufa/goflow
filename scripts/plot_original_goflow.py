"""Plot measured Gears training and saved density; never fabricate a privileged value."""
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
from goflow.rl_components.my_a2c_common import NormFlowDist

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
output = args.run / 'plots'
output.mkdir(exist_ok=True)
metrics = [json.loads(s) for s in (args.run / 'training.jsonl').read_text().splitlines()]
with (args.run / 'training_episodes.csv').open() as f:
    episodes = list(csv.DictReader(f))
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for validation, label in [('False', 'Flow training'), ('True', 'Uniform validation')]:
    selected = [r for r in episodes if r['validation_at_end'] == validation]
    bins = {}
    for row in selected:
        bins.setdefault(int(row['transition']) // 25000, []).append(row)
    x = [(b+.5)*25000 for b in bins]
    axes[0,0].plot(x, [np.mean([float(r['return']) for r in rows]) for rows in bins.values()], label=label)
    axes[0,1].plot(x, [np.mean([int(r['success']) for r in rows]) for rows in bins.values()], label=label)
axes[0,0].axhline(50, color='gray', ls='--', label='Released success threshold')
axes[0,0].set_ylabel('Mean complete-episode return')
axes[0,1].set_ylabel('Success fraction in 25k-transition bin')
for key in ('actor_loss', 'critic_loss'):
    selected = [r for r in metrics if key in r]
    axes[1,0].plot([r['transitions'] for r in selected], [r[key] for r in selected], label=key)
axes[1,0].set_ylabel('Logged PPO loss')
axes[1,1].plot([r['transitions'] for r in metrics], [r['log_p_phi']['mean'] for r in metrics])
axes[1,1].set_ylabel('Mean log density at rollout ξ')
for ax in axes.flat:
    ax.set_xlabel('Actual control transitions (including validation)')
    ax.grid(alpha=.2)
for ax in (axes[0,0], axes[0,1], axes[1,0]):
    ax.legend(fontsize=8)
fig.suptitle('Measured Gears pilot; training success is not held-out competence')
fig.savefig(output/'training.png', dpi=150)
plt.close(fig)

checkpoint = torch.load(args.run/'checkpoints/final.pth', map_location='cuda:0', weights_only=False)
flow = NormFlowDist(torch.tensor([-torch.pi,-.02,-.02]), torch.tensor([torch.pi,.02,.02]), 3)
flow.flow.load_state_dict(checkpoint['goflow_distribution'])
grid = np.linspace(-.02,.02,121)
x,y = np.meshgrid(grid,grid)
xi = torch.tensor(np.column_stack([np.zeros(x.size),x.ravel(),y.ravel()]),dtype=torch.float32,device='cuda:0')
with torch.no_grad():
    lp = flow.log_prob(xi).cpu().numpy().reshape(x.shape)
fig,ax = plt.subplots(figsize=(6,5),constrained_layout=True)
im=ax.pcolormesh(x*1000,y*1000,lp,shading='auto')
fig.colorbar(im,ax=ax,label='log p_phi (upstream normalized-coordinate convention)')
ax.set(xlabel='x offset (mm)',ylabel='y offset (mm)',title='Saved learned flow, yaw = 0 slice')
fig.savefig(output/'density_yaw_zero.png',dpi=150)
np.savez_compressed(output/'density_yaw_zero.npz',x=x,y=y,log_density=lp)
(output/'limitations.txt').write_text(
    'This released configuration has no privileged critic. No V(s,xi) or published Pre_Insert plot is claimed.\n'
    'Density is plotted in upstream normalized-coordinate units, without a physical-coordinate Jacobian.\n')
print(output)
