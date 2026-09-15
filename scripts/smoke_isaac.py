"""Infrastructure check only: start Isaac Sim and step an empty physics stage."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / '.cache' / 'isaac-smoke.json'
started = time.time()
# License acceptance was explicitly authorized in this conversation.
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
for token in ('cache', 'data', 'logs', 'documents', 'app_documents', 'shared_documents'):
    target = ROOT / '.cache' / 'kit' / token
    target.mkdir(parents=True, exist_ok=True)
    sys.argv.append(f'--/app/tokens/{token}={target}')
sys.argv += [
    '--/app/extensions/registryEnabled=false',
    '--/app/settings/persistent=false',
    '--/crashreporter/enabled=false',
    f'--/app/captureFrame/path={ROOT / ".cache/kit/screenshots"}',
    f'--/persistent/app/captureFrame/path={ROOT / ".cache/kit/screenshots"}',
]

from omni.isaac.lab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app
try:
    from omni.isaac.lab.sim import SimulationContext, SimulationCfg
    import omni.replicator.core
    import torch

    simulation = SimulationContext(SimulationCfg(dt=1 / 120, device='cuda:0'))
    simulation.reset()
    print('GPU physics reset completed', flush=True)
    for _ in range(10):
        simulation.step(render=False)
    report = {
        'status': 'steps_completed',
        'scope': 'empty Isaac Lab GPU physics infrastructure; no GoFlow evaluation',
        'physics_steps': 10,
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(),
        'simulation_device': simulation.device,
        'wall_seconds': time.time() - started,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
