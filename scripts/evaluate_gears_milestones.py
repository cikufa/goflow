"""Evaluate completed checkpoints from one continuing run, without training sweeps."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
parser.add_argument('--milestones', type=int, nargs='+', default=[3000000, 4000000, 5000000])
args = parser.parse_args()
run = args.run.resolve()
results = run / 'evaluations'
results.mkdir(exist_ok=True)
for milestone in args.milestones:
    while True:
        candidates = sorted((run / 'checkpoints').glob('transitions_*.pth'))
        candidates = [p for p in candidates if int(p.stem.split('_')[1]) >= milestone]
        if candidates:
            checkpoint = candidates[0]
            # Writer saves synchronously at rollout boundary; wait for the next
            # metrics row or final runtime marker, not just path creation.
            lines = (run / 'training.jsonl').read_text().splitlines()
            latest = json.loads(lines[-1])['transitions'] if lines else 0
            if latest > int(checkpoint.stem.split('_')[1]) or (run / 'runtime.json').exists():
                break
        time.sleep(5)
    for sampling, seed in [('uniform', 30000), ('flow', 40000)]:
        output = results / f'{milestone}_{sampling}'
        output.mkdir(exist_ok=True)
        command = [str(ROOT / 'scripts/project_python.sh'), '-u', 'scripts/eval_original_goflow.py',
                   '--checkpoint', str(checkpoint), '--agent_config',
                   'experiments/original_gears/privileged_goflow.yaml', '--episodes',
                   '100' if milestone == args.milestones[-1] else '32',
                   '--seed-base', str(seed), '--sampling', sampling, '--output', str(output)]
        (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        with (output / 'run.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((output / 'summary.json').read_text())
        print(json.dumps({'milestone': milestone, 'sampling': sampling,
                          'success_rate': summary['success_rate'],
                          'mean_return': summary['mean_episode_reward']}), flush=True)
