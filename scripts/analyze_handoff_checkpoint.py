"""Analyze 100+ physical handoffs per cell without planner-dependent tuning."""
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

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--handoffs', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
checkpoint = torch.load(args.checkpoint, map_location='cuda:0', weights_only=False)
spec = checkpoint.get('handoff_alignment')
if spec is None:
    bounds = np.array([[-.12,.12],[-.04,.04],[-.004,.004],[-np.pi,np.pi]])
else:
    bounds = np.array(list(spec['dr_ranges'].values()))
flow = NormFlowDist(torch.tensor(bounds[:,0]), torch.tensor(bounds[:,1]), 4)
flow.flow.load_state_dict(checkpoint['goflow_distribution'])
config = yaml.safe_load((ROOT/'experiments/original_gears/privileged_goflow.yaml').read_text())['params']['config']
critic = privileged_value_model(checkpoint, config['central_value_config'])
with np.load(args.handoffs/'insert_trajectories.npz') as data:
    contexts = data['context'].copy()
    observations = data['observation'][0].copy()
    returns = data['reward'].sum(0)
    discounted = config['gamma']**np.arange(len(data['reward'])) @ data['reward']
    distance = data['goal_distance'][-1].copy()
with torch.no_grad():
    xi = torch.tensor(contexts,device='cuda:0')
    obs = torch.tensor(observations,device='cuda:0')
    value = critic(dict(obs=torch.cat((obs,xi),1).clamp(-5,5),is_train=False))['values'].flatten().cpu().numpy()
    log_density = flow.log_prob(xi).cpu().numpy()
# Fixed inherited diagnostic epsilon, chosen before aligned-policy outcomes.
# The paper and release do not supply numerical epsilon/eta for this task.
epsilon = .0002145212929463014
accepted = (value > 50) & (log_density > np.log(epsilon))
success = returns >= 50
rows = []
for i in range(len(contexts)):
    rows.append(dict(trial=i, grasp='GraspLeft' if contexts[i,1]<0 else 'GraspRight',
        fixture_angle=float(contexts[i,3]), success=int(success[i]), episode_return=float(returns[i]),
        discounted_return=float(discounted[i]), final_goal_distance_m=float(distance[i]),
        value=float(value[i]), density=float(np.exp(log_density[i])), accepted=int(accepted[i]),
        value_threshold=50, density_threshold=epsilon))
with (args.output/'samples.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
table=[]
for fixture in (-1,1):
    for grasp in (-1,1):
        mask=(contexts[:,1]*grasp>0)&(contexts[:,3]*fixture>0)
        table.append(dict(grasp='GraspLeft' if grasp<0 else 'GraspRight',fixture_angle=fixture*np.pi/2,
            samples=int(mask.sum()),successes=int(success[mask].sum()), mean_return=float(returns[mask].mean()),
            mean_value=float(value[mask].mean()),mean_density=float(np.exp(log_density[mask]).mean()),
            accepted=int(accepted[mask].sum()),
            precision=float(success[mask & accepted].mean()) if (mask & accepted).any() else None,
            recall=float(accepted[mask & success].mean()) if (mask & success).any() else None))
report=dict(checkpoint=str(args.checkpoint),handoffs=str(args.handoffs),training=checkpoint.get('instrumentation'),
    cases=table,density_threshold=epsilon,
    threshold_fidelity='Fixed inherited custom diagnostic epsilon; published JT=50. Numerical epsilon/eta and a calibration protocol are absent from the release/paper. This is not a verified paper threshold.',
    value_return_correlation=float(np.corrcoef(value,discounted)[0,1]),
    density_return_correlation=float(np.corrcoef(log_density,returns)[0,1]),
    precision=float(success[accepted].mean()) if accepted.any() else None,
    recall=float(accepted[success].mean()) if success.any() else None)
# Each side uses its own observed actor state and empirical yaw/y. A single
# constant-observation slice would hide the different grasp effects on the hand.
fig, axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
slice_arrays={}
for row,sign in enumerate((-1,1)):
    mask=contexts[:,1]*sign>0
    representative=np.flatnonzero(mask)[0]
    gx,ga=np.meshgrid(np.linspace(bounds[1,0],bounds[1,1],81),np.linspace(-np.pi,np.pi,81))
    grid=np.column_stack((np.full(gx.size,np.median(contexts[mask,0])),gx.ravel(),
                          np.full(gx.size,np.median(contexts[mask,2])),ga.ravel())).astype(np.float32)
    with torch.no_grad():
        x=torch.tensor(grid,device='cuda:0')
        o=torch.tensor(observations[representative],device='cuda:0').expand(len(x),-1)
        v=critic(dict(obs=torch.cat((o,x),1).clamp(-5,5),is_train=False))['values'].cpu().numpy().reshape(gx.shape)
        lp=flow.log_prob(x).cpu().numpy().reshape(gx.shape)
    joint=(v>50)&(lp>np.log(epsilon))
    for ax,z,title in zip(axes[row,:3],(lp,v,joint),('Log density','Value','Joint indicator')):
        im=ax.pcolormesh(gx*1000,ga,z,shading='auto');fig.colorbar(im,ax=ax)
        ax.scatter(contexts[mask,1]*1000,contexts[mask,3],s=5,c='red',alpha=.4)
        ax.set(xlabel='Grasp x (mm)',ylabel='Fixture angle',title=title)
    axes[row,3].scatter(contexts[mask,1]*1000,contexts[mask,3],c=success[mask],vmin=0,vmax=1,cmap='RdYlGn')
    axes[row,3].set(title=f'Empirical success: {"Left" if sign<0 else "Right"}',xlabel='Grasp x (mm)',ylabel='Fixture angle')
    slice_arrays[f'{sign}_value']=v; slice_arrays[f'{sign}_log_density']=lp; slice_arrays[f'{sign}_joint']=joint
fig.suptitle('Measured handoff overlays; fixed side-specific actor observations; empirical success measured only at canonical cells')
fig.savefig(args.output/'handoff_precondition_slices.png',dpi=150)
np.savez_compressed(args.output/'slices.npz',x=gx,angle=ga,**slice_arrays)
(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
