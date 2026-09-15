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
for token in ('cache', 'data', 'logs', 'app_documents', 'shared_documents'):
    path = root / '.cache/kit' / token
    path.mkdir(parents=True, exist_ok=True)
    kit_args.append(f'--/app/tokens/{token}={path}')
for key in ('/app/captureFrame/path', '/persistent/app/captureFrame/path'):
    kit_args.append(f'--{key}={root / ".cache/kit/screenshots"}')
sys.argv = [str(root / 'train_rl.py'), *sys.argv[1:], '--kit_args', ' '.join(kit_args)]
runpy.run_path(str(root / 'train_rl.py'), run_name='__main__')
