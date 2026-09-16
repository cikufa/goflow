"""Count actual PPO/validation exposure near the empirical handoff modes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',type=Path)
args=parser.parse_args()
checkpoint=torch.load(args.run/'checkpoints/final.pth',map_location='cpu',weights_only=False)
with np.load(checkpoint['handoff_alignment']['bank']) as bank: contexts=bank['context']
centers=[np.median(contexts[contexts[:,1]*s>0,:3],0) for s in (-1,1)]
rows=[json.loads(line) for line in (args.run/'training.jsonl').read_text().splitlines()]
paths=sorted((args.run/'rollouts').glob('*.npz'))
assert len(rows)==len(paths)
counts={phase:{'total':0,'near_left':0,'near_right':0,'feasible_left_region':0,'feasible_right_region':0}
        for phase in ('ppo','validation')}
for path,row in zip(paths,rows):
    with np.load(path) as data: xi=data['xi'].reshape(-1,4)
    count=counts['validation' if row['validation'] else 'ppo'];count['total']+=len(xi)
    for label,center,phi in zip(('left','right'),centers,(-np.pi/2,np.pi/2)):
        mask=(np.abs(xi[:,:3]-center)<[.001,.0005,.0001]).all(1)
        count['near_'+label]+=int(mask.sum())
        count['feasible_'+label+'_region']+=int((mask&(np.abs(xi[:,3]-phi)<.2)).sum())
result=dict(region_tolerance_yaw_x_y=[.001,.0005,.0001],fixture_tolerance_rad=.2,
            centers=np.array(centers).tolist(),counts=counts,
            scope='Transition exposure; repeated episode contexts are not independent episodes.')
(args.run/'handoff_exposure.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
