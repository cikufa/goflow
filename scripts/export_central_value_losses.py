"""Export the actual RL Games central critic losses, distinct from shared-head loss."""
import argparse
import csv
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser=argparse.ArgumentParser()
parser.add_argument('run',type=Path)
parser.add_argument('--tensorboard',type=Path,required=True)
args=parser.parse_args()
events=EventAccumulator(str(args.tensorboard),size_guidance={'scalars':0}).Reload()
losses=events.Scalars('losses/cval_loss')
resume = args.run / 'resume.json'
# RL Games restarts the central-value TensorBoard frame counter on restore.
offset = json.loads(resume.read_text()).get('training_transitions', 0) if resume.exists() else 0
with (args.run/'privileged_critic_losses.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['training_transition','loss','wall_time_unix'])
    writer.writeheader()
    for item in losses:
        writer.writerow({'training_transition':item.step + offset,'loss':item.value,'wall_time_unix':item.wall_time})
print(f'Exported {len(losses)} actual central-value updates')
