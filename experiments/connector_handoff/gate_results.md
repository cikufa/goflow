# Connector handoff experiment: acceptance-gate report

Final planner trials: **0**. This report records executed diagnostics and explicitly marks unexecuted stages.

## 1. Scientific question

Does the published GoFlow belief-space planner acquire successor-relevant fixture information before choosing between two locally valid grasps? Both a positive and a negative answer require competent low-level skills and real belief updates.

## 2. GoFlow fidelity

Released PPO, GoFlow objective/architecture, 105-input actor, 109-input privileged critic, action scaling, two-second episode and reward are retained. Custom grasp timing, physical staging and empirical reset initialization are explicit adaptations. Numerical epsilon/eta and an exact calibration recipe are absent from the released planner materials; diagnostic epsilon is inherited and not retuned. See experiments/connector_handoff/final_experiment.md and docs/method_fidelity.md.

## 3. Existing implementation inherited

The original 10,027,008-transition connector checkpoint and both earlier result directories are preserved. Original checkpoint SHA256: d2e75560f58ac9b4836193e3d437e0817fbcbb7d4eb652680abe40d3be8770e5. It achieved 33/100 flow and 2/100 uniform successes before this work.

## 4. Actual grasp terminal distributions

Initial inherited calibration: 100 executions per grasp, all locally successful; +/-3 mm planar placement, nominal initial yaw. Closure tilted the left-held connector. CSV/statistics/distribution/correlation plots are in calibration/. Revised timing was separately measured on 200 executions per grasp in calibration/aligned_complete_state/. The sampled domain is narrow; no broad six-axis robustness is claimed.

## 5. Handoff/staging alignment

Physical GRASP → elevated translation waypoint → common connector staging point → INSERT is executed without resetting robot or object. The common staging point is 10 mm above the former root initialization (60 mm above the goal). Measured hand orientation and hand-to-connector transform are preserved. The single measured constraint is created before closure in the revised macro. Final calibrated grasp/stage success is 400/400. Maximum stage hand error 0.689 mm; transform translation drift 0.0122 mm; rotation drift 0.000340 rad. Replay preserves authored constraint targets and actuator targets separately from contact-loaded poses. The shorter 30 mm staging probe failed Left and was not adopted.

## 6. INSERT retraining

Executed additional training:

- results/custom_connector/final_handoff_experiment/training/aligned_2m: 2,031,616 additional transitions; 1,048,576 cumulative PPO / 983,040 validation; 5 flow updates.
- results/custom_connector/final_handoff_experiment/training/supported_3m: 3,014,656 additional transitions; 1,572,864 cumulative PPO / 1,441,792 validation; 7 flow updates.

The first tight-bound initialization starved both real grasp clusters. Data-based wider bounds improve initial support while retaining the released normalization and updates. Both warm-start experiments retain original actor/critic weights and start a fresh flow because their normalization domains differ. They are separate documented lineages, not one uninterrupted run.

## 7. Learned flow/value/precondition

| Fixture angle | Grasp | Policy successes | Mean return | Mean V | Mean density | Joint accepts |
|---|---|---:|---:|---:|---:|---:|
| -1.5708 | GraspLeft | 6/100 | 22.69 | 55.40 | 1.6e-05 | 0/100 |
| -1.5708 | GraspRight | 0/100 | 17.94 | 18.33 | 0.000165 | 0/100 |
| 1.5708 | GraspLeft | 0/100 | 24.84 | 7.62 | 1.65e-05 | 0/100 |
| 1.5708 | GraspRight | 67/100 | 57.63 | 34.57 | 4.06e-05 | 0/100 |

Fixed inherited custom diagnostic epsilon; published JT=50. Numerical epsilon/eta and a calibration protocol are absent from the release/paper. This is not a verified paper threshold.

Plots and samples: results/custom_connector/final_handoff_experiment/precondition_analysis/supported_3m. No applicability threshold was lowered to admit a case.

## 8. Physics reversal validation

Actual handoffs with the oracle controller: Left/-pi/2 99/100, Right/+pi/2 100/100, both blocked combinations 0/100. Feasible mean terminal root errors are 0.108/0.049 mm; blocked errors are approximately 22–23 mm. Grasp and Stage each pass in all 400 cases. Results: calibration/aligned_complete_state/physics_matrix.json. The released reward is based on the translation of a composed pose error, so held orientation affects reward despite zero explicit rotational weights. Root distance alone does not recover the released return.

## 9. Perception and belief validation

Not run. Gate C follows accepted/frozen low-level skills. Existing synthetic Bayes3D infrastructure checks do not constitute this camera-belief experiment.

## 10. Planner validation

No task-specific final planner validation or online planner episodes ran. Five existing generic BFS/Equation-7 unit tests pass; they do not establish prospective inspection behavior.

## 11. Frozen final experiment

Not frozen. No final seed list or accepted low-level checkpoint exists. The latest checkpoint is a diagnostic candidate, not a frozen final policy.

## 12. Trial-by-trial outcomes

No final scientific trials ran. summary.csv intentionally contains only column headings; low_level_summary.csv contains the actual diagnostic case results.

## 13. Prospective vs reactive sensing

Not measured. T_info, T_grasp and T_insert_check do not exist for unexecuted planner episodes.

## 14. Decision-change analysis

Not measured; no before/after-inspection BFS decision pair exists.

## 15. Counterfactual handoff failures

No online-trial counterfactual replay ran. The four-cell low-level physics matrix must not be reported as counterfactual evidence about a planner decision.

## 16. Failure attribution

The acceptance failure is in low-level learned competence and/or learned support of real handoff states. Sampling starvation was directly measured: in the supported 3M run, PPO saw 47/713 transitions near measured Left/Right grasp modes and zero in either canonical feasible neighborhood when fixture angle is also constrained to within 0.2 rad. An additional context-only likelihood initialization has been prepared but NOT executed; it requires approval because that objective is absent from the release. Feasible physical handoffs exist. Neither prospective-inspection failure nor a GoFlow planning limitation can be inferred. Several library-import failures occurred before simulation; failed launches were preserved and excluded, with unchanged retries. Their system-level cause is not established.

## 17. Videos / contact sheets / timelines

Complete actual oracle GRASP→Stage→INSERT video: calibration/oracle_handoff_video/handoff.mp4. Contact sheet: calibration/oracle_handoff_video/contact_sheet.png. These were visually inspected and show the actual task. Precondition slices: precondition_analysis/supported_3m/handoff_precondition_slices.png. No 20-episode online videos, planner contact sheets or belief timelines exist because the gates have not passed.

## 18. Fidelity limitations

Gravity-disabled object and rigid-constraint grasp approximation; narrow planar grasp calibration; measured contact deflections and replay cannot restore PhysX internal contact caches; empirical box extends/interpolates the two macro modes; simulator/RNG state is not resumed across training sessions; numerical planner thresholds are underspecified by the release. Earlier original-task reproduction is a functional baseline, not a claim of matching all paper metrics.

## 19. Scientific interpretation

Executed evidence cannot distinguish whether published GoFlow already solves prospective sensing in this two-grasp setting. Low-level acceptance must be resolved before studying the planner. No hidden-state rule, successor-conditioned grasp predicate or inspect-first heuristic has been added.

## 20. Conclusion label

**INCONCLUSIVE**

Git provenance at report generation:

```json
{
  "upstream_sha": "a8c6af5de7f427418783fd9faa20d50f38b734a9",
  "branch": "goflow-handoff-experiment",
  "head": "e8735a0f9018574de758cca95f10c9b39e053436"
}
```
