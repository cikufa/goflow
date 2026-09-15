"""Check actual actor/critic input separation and trained critic sensitivity to ξ."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from experiments.common.checkpoints import privileged_value_model

parser=argparse.ArgumentParser()
parser.add_argument('run',type=Path)
parser.add_argument('--config',type=Path,default=ROOT/'experiments/original_gears/privileged_goflow.yaml')
args=parser.parse_args()
checkpoint=torch.load(args.run/'checkpoints/final.pth',map_location='cuda:0',weights_only=False)
cfg=yaml.safe_load(args.config.read_text())['params']['config']['central_value_config']
model=privileged_value_model(checkpoint,cfg)
with np.load(sorted((args.run/'rollouts').glob('*.npz'))[-1]) as rollout:
    actor=rollout['observations'][0]
    critic=rollout['critic_observations'][0]
    xi=rollout['xi'][0]
    assert actor.shape[1]==105 and critic.shape[1]==108
    np.testing.assert_allclose(actor,critic[:,:105])
    np.testing.assert_allclose(xi,critic[:,-3:])
states=torch.tensor(critic,device='cuda:0',requires_grad=True)
out=model({'obs':states,'is_train':False})['values']
# The public inference path denormalizes under no_grad; inspect actual functional
# dependence by perturbing only the privileged coordinates at fixed actor input.
with torch.no_grad():
    low=states.detach().clone();high=states.detach().clone()
    low[:,-3:]=torch.tensor([-torch.pi,-.02,-.02],device='cuda:0')
    high[:,-3:]=torch.tensor([torch.pi,.02,.02],device='cuda:0')
    delta=(model({'obs':high,'is_train':False})['values']-
           model({'obs':low,'is_train':False})['values']).abs()
assert torch.isfinite(out).all() and torch.isfinite(delta).all()
assert float(delta.max())>1e-6, 'No measured privileged-parameter dependence'
result={'status':'passed','actor_width':105,'critic_width':108,
        'actor_prefix_matches':True,'privileged_suffix_matches_xi':True,
        'max_value_change_at_fixed_actor_observation':float(delta.max()),
        'mean_value_change_at_fixed_actor_observation':float(delta.mean()),
        'scope':'Input separation and functional dependence; not skill competence or critic calibration'}
(args.run/'critic_access_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
