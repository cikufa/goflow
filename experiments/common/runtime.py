"""Filesystem settings shared by standalone experiment entry points."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def kit_arguments():
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    (ROOT / 'goflow/logs').mkdir(parents=True, exist_ok=True)
    args = ['--/app/extensions/registryEnabled=false', '--/app/settings/persistent=false',
            '--/crashreporter/enabled=false']
    if os.environ.get('GOFLOW_CPU_THREADS'):
        threads = int(os.environ['GOFLOW_CPU_THREADS'])
        if threads < 1:
            raise ValueError('GOFLOW_CPU_THREADS must be positive')
        args.append(f'--/plugins/carb.tasking.plugin/threadCount={threads}')
        print(f'GoFlow process-local Carb thread cap: {threads}', flush=True)
    for token in ('cache', 'data', 'logs', 'app_documents', 'shared_documents'):
        path = ROOT / '.cache/kit' / token
        path.mkdir(parents=True, exist_ok=True)
        args.append(f'--/app/tokens/{token}={path}')
    for prefix in ('/app', '/persistent/app'):
        args.append(f'--{prefix}/captureFrame/path={ROOT / ".cache/kit/screenshots"}')
    return ' '.join(args)
