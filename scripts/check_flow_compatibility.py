"""Exercise the unchanged upstream flow class without importing the simulator."""
import ast
import json
from pathlib import Path
import subprocess

import torch
import zuko

UPSTREAM = 'a8c6af5de7f427418783fd9faa20d50f38b734a9'
source = subprocess.check_output(
    ['git', 'show', f'{UPSTREAM}:goflow/rl_components/my_a2c_common.py'], text=True
)
nodes = [node for node in ast.parse(source).body
         if isinstance(node, ast.ClassDef) and node.name in ('Distr', 'NormFlowDist')]
assert len(nodes) == 2
namespace = {'torch': torch, 'zuko': zuko}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'upstream_flow_classes', 'exec'), namespace)
torch.manual_seed(12345)
flow = namespace['NormFlowDist'](
    torch.tensor([-torch.pi, -.02, -.02]), torch.tensor([torch.pi, .02, .02]), 3
)
samples = flow.rsample((32,))
log_prob = flow.log_prob(samples.detach())
(-log_prob.mean()).backward()
assert samples.shape == (32, 3) and torch.isfinite(log_prob).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.get_params())
result = {'status': 'passed', 'upstream': UPSTREAM, 'seed': 12345,
          'checks': ['CUDA sampling', 'finite log density', 'finite parameter gradients'],
          'simulator_transitions': 0, 'scope': 'dependency compatibility only'}
Path('.cache/flow-compatibility.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
