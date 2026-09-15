"""Run the released CLI with project-local Kit paths and no extension downloads."""
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0, str(root))
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'  # explicitly authorized by user
(root / 'goflow/logs').mkdir(parents=True, exist_ok=True)
kit_args = ['--/app/extensions/registryEnabled=false', '--/app/settings/persistent=false',
            '--/crashreporter/enabled=false']
# Optional process-local scheduler cap for intermittent native startup failures.
# Leave physics and learning parameters unchanged; record the override in stdout.
if os.environ.get('GOFLOW_CPU_THREADS'):
    threads = int(os.environ['GOFLOW_CPU_THREADS'])
    if threads < 1:
        raise ValueError('GOFLOW_CPU_THREADS must be positive')
    kit_args.append(f'--/plugins/carb.tasking.plugin/threadCount={threads}')
    print(f'GoFlow process-local Carb thread cap: {threads}', flush=True)
for token in ('cache', 'data', 'logs', 'app_documents', 'shared_documents'):
    path = root / '.cache/kit' / token
    path.mkdir(parents=True, exist_ok=True)
    kit_args.append(f'--/app/tokens/{token}={path}')
for key in ('/app/captureFrame/path', '/persistent/app/captureFrame/path'):
    kit_args.append(f'--{key}={root / ".cache/kit/screenshots"}')
sys.argv = [str(root / 'train_rl.py'), *sys.argv[1:], '--kit_args', ' '.join(kit_args)]
runpy.run_path(str(root / 'train_rl.py'), run_name='__main__')
