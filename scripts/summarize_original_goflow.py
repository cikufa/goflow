"""Generate an evidence-based baseline report; explicitly retain unexecuted stages."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
output=root/'results/original_gears'
runs=[('released_training_64','90a9922','Initial pilot; conflicting cloned constraints'),
      ('corrected_training_64','19390ef','One grasp constraint per environment'),
      ('privileged_training_64','ddc238c','Corrected constraints and explicit privileged critic')]
training=[]
for name,commit,description in runs:
    data=json.loads((output/name/'runtime.json').read_text())
    checkpoint=output/name/'checkpoints/final.pth'
    training.append(data|{'run':name,'source_commit':commit,'description':description,
                         'checkpoint':str(checkpoint.relative_to(root)),
                         'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         'start_utc':datetime.fromtimestamp(data['started_unix'],timezone.utc).isoformat(),
                         'end_utc':datetime.fromtimestamp(data['started_unix']+data['wall_seconds'],timezone.utc).isoformat()})
evaluations={name:json.loads((output/name/'summary.json').read_text()) for name in
             ['corrected_evaluation','privileged_independent_uniform','privileged_independent_flow']}
evaluations['initial_released_pilot']=json.loads((output/'summary.json').read_text())
final=[evaluations[n] for n in ('privileged_independent_uniform','privileged_independent_flow')]
successful=sum(round(r['episodes']*r['success_rate']) for r in final)
status='baseline_competence_unestablished' if successful==0 else 'baseline_requires_further_validation'
report={'status':status,'scientific_conclusion':'Experiment inconclusive',
        'branch':subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
        'report_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'upstream':json.loads((root/'docs/upstream.json').read_text()),
        'training_runs':training,'evaluations':evaluations,
        'total_main_pilot_transitions':sum(r['transitions'] for r in training),
        'total_main_pilot_agent_seconds':sum(r['wall_seconds'] for r in training),
        'custom_connector':{'status':'not_started_baseline_gate_unmet','training_transitions':0,'trials':0,
                            'prospective_inspection_rate':None,'handoff_failure_rate':None,
                            'end_to_end_success':None,'counterfactual_replays':0},
        'original_online_planner_demo':'not_reproduced',
        'bayes3d':'Renderer and documented SMC/grid adapter pass synthetic-image checks; no Isaac RGB-D integration claim'}
(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
with (output/'training_runtime.csv').open('w') as f:
    fields=['run','source_commit','start_utc','end_utc','transitions','training_transitions','validation_transitions',
            'parallel_envs','gpu','completed_flow_updates','wall_seconds','checkpoint','checkpoint_sha256']
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(training)
lines=['# GoFlow baseline execution report','',
       '**Experiment inconclusive. The requested connector experiment is not complete.**','',
       'The baseline competence gate remains unmet. No connector scene, connector training, 20-trial handoff evaluation, '
       'online high-level action sequences, sensing rates, or counterfactual handoff metrics are claimed. '
       'The 20 final episodes below are Gears diagnostics, not the requested balanced connector trials.','',
       '## Actual training','',
       '| Run | Source | Control transitions | PPO transitions | Uniform validation | Agent seconds | Flow updates |',
       '|---|---|---:|---:|---:|---:|---:|']
for r in training:
    lines.append(f"| {r['run']} | {r['source_commit']} | {r['transitions']:,} | {r['training_transitions']:,} | "
                 f"{r['validation_transitions']:,} | {r['wall_seconds']:.3f} | {r['completed_flow_updates']} |")
lines+=['','Times are measured agent runtimes, excluding simulator startup. Some runs overlapped short GPU infrastructure '
        'probes; these are elapsed execution times, not isolated speed benchmarks. Control transitions exclude physics substeps '
        'and reset stabilization. Smoke runs are recorded separately and not pooled into this table.','',
        '## Held-out policy checks','',
        '| Dataset | Sampling | Episodes | Successes | Mean return | Final goal distance (m) |',
        '|---|---|---:|---:|---:|---:|']
for name,r in evaluations.items():
    lines.append(f"| {name} | {r['sampling']} | {r['episodes']} | {round(r['episodes']*r['success_rate'])} | "
                 f"{r['mean_episode_reward']:.4f} | {r['mean_final_goal_distance_m']:.5f} |")
lines+=['','Success uses the released return threshold 50. Independent final episodes clear the pose history before reset '
        'to match synchronous 64-environment training resets. Earlier preserved diagnostics retained the released one-env history bug.','',
        '## What the policy actually did','',
        'The green gear starts attached to the gripper approximately 5 cm above the shaft. The final policy mostly cancels '
        'the default downward motion with a positive vertical residual, makes small lateral movements, and hovers. '
        'It does not complete insertion in these tests. No INSPECT/GRASP/INSERT high-level sequence was executed.','',
        '## Fidelity and remaining limits','',
        '- No public trained checkpoint was found. All evaluated weights were locally trained.',
        '- The released default disables the privileged critic and samples yaw without applying its rotation. Its bounds and duration differ from the paper.',
        '- The initial multi-environment scene had conflicting duplicate grasp constraints. A one-line path correction restores one constraint per environment; the original run is preserved.',
        '- The final profile enables the exact central-critic configuration present in upstream comments. Input separation and value dependence on ξ pass runtime checks.',
        '- PPO and GoFlow distribution objectives remain released code; the flow objective and density units have documented paper/code discrepancies.',
        '- Generic BFS/Equation 7 and Bayes3D SMC/grid inference are implemented and component-tested. They have not been validated together in an online robot planner.',
        '- Current evidence cannot support or refute the proposed handoff gap.','',
        '## Inspect the evidence','',
        '- Final videos: `privileged_independent_flow/videos/episode_000.mp4` through `episode_009.mp4`.',
        '- Contact sheets: `privileged_independent_flow/contact_sheets/`.',
        '- Representative raw frames: `privileged_independent_flow/raw_frames/episode_000/`.',
        '- PPO curves and learned density/value slices: `privileged_training_64/plots/`.',
        '- Full online rollouts, complete episode returns, flow logs and critic losses: `privileged_training_64/`.',
        '- Synthetic perception posterior checks: `../infrastructure/bayes3d-pose-probe/`.',
        '- Audit and exact commands: `../../docs/method_fidelity.md` and `../../experiments/connector_handoff/README.md`.','']
(output/'report.md').write_text('\n'.join(lines))
print(json.dumps({'status':status,'main_pilot_transitions':report['total_main_pilot_transitions'],
                  'main_pilot_agent_seconds':report['total_main_pilot_agent_seconds'],'report':str(output/'report.md')},indent=2))
