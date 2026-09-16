"""Report executed acceptance gates; never invent unrun planner trials."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('results/custom_connector/final_handoff_experiment'))
parser.add_argument('--analysis', type=Path, required=True)
parser.add_argument('--precondition-validated', action='store_true',
                    help='Record a completed review of feasible/blocked applicability; success rates alone do not pass Gate B')
args = parser.parse_args()
root = args.root
if (root/'frozen_experiment.yaml').exists() or list((root/'trials').glob('*/summary.json')):
    raise ValueError('This gate reporter must not overwrite an actual final trial report')
analysis = json.loads(args.analysis.read_text())
cases = analysis['cases']
competence_pass = all((c['successes']/c['samples'] >= .7 if
              (c['grasp']=='GraspLeft') == (c['fixture_angle']<0) else
              c['successes']/c['samples'] < .1) for c in cases)
# Applicability evidence is reported separately; no numerical acceptance target
# for learned-precondition coverage is supplied by the paper.
gate_b = competence_pass and args.precondition_validated
latest_run=Path(analysis['checkpoint']).parents[1]
exposure_path=latest_run/'handoff_exposure.json'
exposure=json.loads(exposure_path.read_text()) if exposure_path.exists() else None
objective_path=latest_run/'flow_objective_audit.json'
objective=json.loads(objective_path.read_text()) if objective_path.exists() else None
calibration_paths=sorted((latest_run/'analysis').glob('*/summary.json'))
calibration=json.loads(calibration_paths[-1].read_text()) if calibration_paths else None
training=[]
for path in sorted((root/'training').glob('*/runtime.json')):
    data=json.loads(path.read_text())
    resume=path.parent/'resume.json'
    prior=json.loads(resume.read_text()) if resume.exists() else {}
    training.append(dict(run=str(path.parent), **data,
        additional_transitions=data['transitions']-prior.get('transitions',0),
        checkpoint_sha256=hashlib.sha256((path.parent/'checkpoints/final.pth').read_bytes()).hexdigest()))
git=dict(upstream_sha='a8c6af5de7f427418783fd9faa20d50f38b734a9',
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
summary=dict(conclusion='INCONCLUSIVE', scientific_trials=0,
    status='low_level_gates_incomplete', gate_a='Physical staging and replay validated in the recorded calibrated domain',
    gate_b=gate_b, competence_pass=competence_pass, exposure=exposure, gate_c='not run: low-level acceptance precedes perception',
    additional_transitions=sum(r['additional_transitions'] for r in training),
    training=training, cases=cases, precondition_analysis=str(args.analysis),
    random_context_calibration=calibration, flow_objective_audit=objective,
    final_frozen_checkpoint=None, latest_diagnostic_checkpoint=analysis['checkpoint'],
    prospective_sensing_rate=None, compatible_grasp_rate=None, handoff_failure_rate=None,
    reactive_recovery_rate=None, end_to_end_success_rate=None, avoidable_failures=None,
    explanation='Rates are undefined because no final planner episodes ran. Low-level diagnostic episodes are not scientific planning trials.',
    git=git)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (root/'summary.csv').open('w') as f:
    csv.writer(f).writerow(['ep','fixture','initial_entropy','pre_grasp_inspect','grasp','oracle_grasp',
                           'Y1','insert_precondition','Y2','recovery','e2e','class'])
with (root/'low_level_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
table='| Fixture angle | Grasp | Policy successes | Mean return | Mean V | Mean density | Joint accepts |\n|---|---|---:|---:|---:|---:|---:|\n'
for c in cases:
    table+=f"| {c['fixture_angle']:.4f} | {c['grasp']} | {c['successes']}/{c['samples']} | {c['mean_return']:.2f} | {c['mean_value']:.2f} | {c['mean_density']:.3g} | {c['accepted']}/{c['samples']} |\n"
sections={
 'Scientific question':'Does the published GoFlow belief-space planner acquire successor-relevant fixture information before choosing between two locally valid grasps? Both a positive and a negative answer require competent low-level skills and real belief updates.',
 'GoFlow fidelity':'Released PPO, GoFlow objective/architecture, 105-input actor, 109-input privileged critic, action scaling, two-second episode and reward are retained. Custom grasp timing, physical staging, empirical reset initialization and the user-approved 1,000-step context-only flow likelihood initialization are explicit adaptations. The extra likelihood objective is absent from the released training procedure. Numerical epsilon/eta and an exact calibration recipe are absent from the released planner materials; diagnostic epsilon is inherited and not retuned. See experiments/connector_handoff/final_experiment.md and docs/method_fidelity.md.',
 'Existing implementation inherited':'The original 10,027,008-transition connector checkpoint and both earlier result directories are preserved. Original checkpoint SHA256: d2e75560f58ac9b4836193e3d437e0817fbcbb7d4eb652680abe40d3be8770e5. It achieved 33/100 flow and 2/100 uniform successes before this work.',
 'Actual grasp terminal distributions':'Initial inherited calibration: 100 executions per grasp, all locally successful; +/-3 mm planar placement, nominal initial yaw. Closure tilted the left-held connector. CSV/statistics/distribution/correlation plots are in calibration/. Revised timing was separately measured on 200 executions per grasp in calibration/aligned_complete_state/. The sampled domain is narrow; no broad six-axis robustness is claimed.',
 'Handoff/staging alignment':'Physical GRASP → elevated translation waypoint → common connector staging point → INSERT is executed without resetting robot or object. The common staging point is 10 mm above the former root initialization (60 mm above the goal). Measured hand orientation and hand-to-connector transform are preserved. The single measured constraint is created before closure in the revised macro. Final calibrated grasp/stage success is 400/400. Maximum stage hand error 0.689 mm; transform translation drift 0.0122 mm; rotation drift 0.000340 rad. Replay preserves authored constraint targets and actuator targets separately from contact-loaded poses. The shorter 30 mm staging probe failed Left and was not adopted.',
 'INSERT retraining':'Executed additional training:\n\n'+ '\n'.join(f"- {r['run']}: {r['additional_transitions']:,} additional transitions; {r['training_transitions']:,} cumulative PPO / {r['validation_transitions']:,} validation; {r['completed_flow_updates']} flow updates." for r in training)+'\n\nThe first tight-bound initialization starved both real grasp clusters. Data-based wider bounds improve initial support while retaining the released normalization and updates. The two earlier attempts and the approved empirical-initialization stage each retain the original actor/critic warm-start weights. The new stage first fits the existing flow to balanced empirical grasp contexts with independent uniform fixture angle, then uses unchanged online GoFlow updates. These are separate documented lineages, not one uninterrupted run.',
 'Learned flow/value/precondition':table+'\n'+analysis['threshold_fidelity']+'\n\nPlots and samples: '+str(args.analysis.parent)+'. No applicability threshold was lowered to admit a case.',
 'Physics reversal validation':'Actual handoffs with the oracle controller: Left/-pi/2 99/100, Right/+pi/2 100/100, both blocked combinations 0/100. Feasible mean terminal root errors are 0.108/0.049 mm; blocked errors are approximately 22–23 mm. Grasp and Stage each pass in all 400 cases. Results: calibration/aligned_complete_state/physics_matrix.json. The released reward is based on the translation of a composed pose error, so held orientation affects reward despite zero explicit rotational weights. Root distance alone does not recover the released return.',
 'Perception and belief validation':'Not run. Gate C follows accepted/frozen low-level skills. Existing synthetic Bayes3D infrastructure checks do not constitute this camera-belief experiment.',
 'Planner validation':'No task-specific final planner validation or online planner episodes ran. Five existing generic BFS/Equation-7 unit tests pass; they do not establish prospective inspection behavior.',
 'Frozen final experiment':'Not frozen. No final seed list or accepted low-level checkpoint exists. The latest checkpoint is a diagnostic candidate, not a frozen final policy.',
 'Trial-by-trial outcomes':'No final scientific trials ran. summary.csv intentionally contains only column headings; low_level_summary.csv contains the actual diagnostic case results.',
 'Prospective vs reactive sensing':'Not measured. T_info, T_grasp and T_insert_check do not exist for unexecuted planner episodes.',
 'Decision-change analysis':'Not measured; no before/after-inspection BFS decision pair exists.',
 'Counterfactual handoff failures':'No online-trial counterfactual replay ran. The four-cell low-level physics matrix must not be reported as counterfactual evidence about a planner decision.',
 'Failure attribution':'The acceptance failure is in low-level learned competence and/or learned support of real handoff states. Previous supported-bounds training had zero measured PPO exposure in either canonical feasible neighborhood. The approved empirical flow initialization has now executed. Latest measured exposure is recorded in the run handoff_exposure.json and the report summary; these are transition counts, not independent episode counts. Feasible physical handoffs exist. Neither prospective-inspection failure nor a GoFlow planning limitation can be inferred. Several library-import failures occurred before simulation; failed launches were preserved and excluded, with unchanged retries. Their system-level cause is not established.',
 'Videos / contact sheets / timelines':'Complete actual oracle GRASP→Stage→INSERT video: calibration/oracle_handoff_video/handoff.mp4. Contact sheet: calibration/oracle_handoff_video/contact_sheet.png. These were visually inspected and show the actual task. Precondition slices are saved alongside the current precondition analysis referenced above. No 20-episode online videos, planner contact sheets or belief timelines exist because the gates have not passed.',
 'Fidelity limitations':'Gravity-disabled object and rigid-constraint grasp approximation; narrow planar grasp calibration; measured contact deflections and replay cannot restore PhysX internal contact caches; empirical box extends/interpolates the two macro modes; simulator/RNG state is not resumed across training sessions; numerical planner thresholds are underspecified by the release; the approved empirical flow likelihood initialization changes the released initialization objective. Earlier original-task reproduction is a functional baseline, not a claim of matching all paper metrics.',
 'Scientific interpretation':'Executed evidence cannot distinguish whether published GoFlow already solves prospective sensing in this two-grasp setting. Low-level acceptance must be resolved before studying the planner. No hidden-state rule, successor-conditioned grasp predicate or inspect-first heuristic has been added.',
 'Conclusion label':'**INCONCLUSIVE**',
}
text='# Connector handoff experiment: acceptance-gate report\n\n'
if calibration:
    sections['Learned flow/value/precondition']+='\n\nHeld-out random-context metrics:\n\n```json\n'+json.dumps(calibration['evaluation'],indent=2)+'\n```'
if objective:
    sections['Failure attribution']+='\n\nOffline released-objective scale probe (not a causal intervention):\n\n```json\n'+json.dumps(objective,indent=2)+'\n```\n\nThe fitted initialization now covers both modes, but density is not established as a learned success region. Inspect the measured loss-scale imbalance before further training; no objective correction was applied.'
if exposure:
    sections['INSERT retraining']+='\n\nLatest measured exposure:\n\n```json\n'+json.dumps(exposure,indent=2)+'\n```'
text+='Final planner trials: **0**. This report records executed diagnostics and explicitly marks unexecuted stages.\n\n'
for i,(name,body) in enumerate(sections.items(),1):text+=f'## {i}. {name}\n\n{body}\n\n'
text+='Git provenance at report generation:\n\n```json\n'+json.dumps(git,indent=2)+'\n```\n'
(root/'report.md').write_text(text)
print(json.dumps(summary,indent=2))
