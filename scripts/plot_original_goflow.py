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
parser.add_argument('--agent_config', type=Path)
parser.add_argument('--observation', type=Path, help='Recorded episode NPZ providing a fixed actor observation')
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
    label=key+' (shared head)' if key=='critic_loss' and metrics[0]['privileged_critic_enabled'] else key
    axes[1,0].plot([r['transitions'] for r in selected], [r[key] for r in selected], label=label)
if (args.run/'privileged_critic_losses.csv').exists():
    transition_map={r['training_transitions']:r['transitions'] for r in metrics if not r['validation']}
    with (args.run/'privileged_critic_losses.csv').open() as f:
        central=[r for r in csv.DictReader(f) if int(r['training_transition']) in transition_map]
    axes[1,0].plot([transition_map[int(r['training_transition'])] for r in central],
                   [float(r['loss']) for r in central],label='privileged critic loss',alpha=.7)
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
if 'assymetric_vf_nets' in checkpoint and args.agent_config and args.observation:
    import yaml
    from experiments.common.checkpoints import privileged_value_model
    config=yaml.safe_load(args.agent_config.read_text())['params']['config']['central_value_config']
    value_model=privileged_value_model(checkpoint,config)
    with np.load(args.observation) as episode:
        observation=episode['observations'][0]
    states=torch.cat([torch.tensor(observation,device='cuda:0').expand(len(xi),-1),xi],dim=1)
    with torch.no_grad():
        value=value_model({'obs':states,'is_train':False})['values'].cpu().numpy().reshape(x.shape)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,z,title in zip(axes,(value,value>50),('Privileged value V(s,xi)','Value indicator V > 50 (not the full precondition)')):
        im=ax.pcolormesh(x*1000,y*1000,z,shading='auto');fig.colorbar(im,ax=ax)
        ax.set(xlabel='x offset (mm)',ylabel='y offset (mm)',title=title)
    fig.suptitle('Fixed recorded initial observation; yaw = 0; calibration must be assessed separately')
    fig.savefig(output/'privileged_value_yaw_zero.png',dpi=150)
    np.savez_compressed(output/'privileged_value_yaw_zero.npz',x=x,y=y,value=value,observation=observation)
    limitation='Privileged value slice shown; no belief applicability claim without density threshold and calibration.\n'
else:
    limitation='No privileged-value plot: model or explicit configuration/observation unavailable.\n'
(output/'limitations.txt').write_text(limitation+
    'Density uses upstream normalized-coordinate units, without a physical-coordinate Jacobian.\n')
print(output)
