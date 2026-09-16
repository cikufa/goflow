"""Preview a proposed context-only flow initialization; fitting is opt-in.

The current experiment has not authorized the additional likelihood objective.
The default command validates its inputs and writes a reviewable proposal only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--proposal',type=Path,default=ROOT/'experiments/connector_handoff/proposed_empirical_flow_start.json')
parser.add_argument('--fit',action='store_true',help='Execute the proposed new initialization objective only after explicit user approval')
args=parser.parse_args()
proposal=json.loads(args.proposal.read_text())
spec=json.loads((ROOT/proposal['alignment_spec']).read_text())
bank_path=ROOT/spec['bank']
assert hashlib.sha256(bank_path.read_bytes()).hexdigest()==spec['bank_sha256']
with np.load(bank_path) as bank: contexts=bank['context'].copy()
assert contexts.shape[1]==4 and np.isfinite(contexts).all()
assert (contexts[:,1]<0).sum()==(contexts[:,1]>0).sum()
weights=torch.load(ROOT/proposal['actor_critic_source'],map_location='cpu',weights_only=False)
output=ROOT/proposal['output']
output.parent.mkdir(parents=True,exist_ok=True)
review=proposal | {'validated_bank_samples':len(contexts),'bank_sha256':spec['bank_sha256'],
                  'executed_fit':args.fit,'training_run_started':False}
if args.fit:
    if output.exists(): raise FileExistsError(output)
    from goflow.rl_components.my_a2c_common import NormFlowDist
    torch.manual_seed(proposal['seed'])
    bounds=torch.tensor(list(spec['dr_ranges'].values()),device='cuda:0',dtype=torch.float32)
    flow=NormFlowDist(bounds[:,0],bounds[:,1],4)
    optimizer=torch.optim.Adam(flow.flow.parameters(),lr=proposal['learning_rate'])
    empirical=torch.tensor(contexts,device='cuda:0')
    jitter=torch.tensor(proposal['jitter_sd_yaw_x_y'],device='cuda:0')
    losses=[]
    for step in range(proposal['fit_steps']):
        index=torch.randint(len(empirical),(proposal['batch_size'],),device='cuda:0')
        target=empirical[index].clone()
        target[:,:3]+=torch.randn_like(target[:,:3])*jitter
        target[:,3].uniform_(float(bounds[3,0]),float(bounds[3,1]))
        target=torch.maximum(torch.minimum(target,bounds[:,1]),bounds[:,0])
        loss=-flow.log_prob(target).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite initialization loss')
        optimizer.zero_grad();loss.backward();optimizer.step()
        losses.append(float(loss))
    weights['goflow_distribution']=flow.flow.state_dict()
    # Do not carry likelihood-fit momentum into the released GoFlow objective.
    weights['goflow_distribution_optimizer']=torch.optim.Adam(flow.flow.parameters(),lr=proposal['learning_rate']).state_dict()
    weights['handoff_alignment']=spec
    weights['empirical_initialization_proposal']=proposal
    torch.save(weights,output)
    review['losses']=losses
    review['checkpoint_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
(output.parent/'proposal_review.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps(review,indent=2))
