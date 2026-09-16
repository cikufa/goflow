# Connector handoff experiment: acceptance-gate report

Final planner trials: **0**. This report records executed diagnostics and explicitly marks unexecuted stages.

## 1. Scientific question

Does the published GoFlow belief-space planner acquire successor-relevant fixture information before choosing between two locally valid grasps? Both a positive and a negative answer require competent low-level skills and real belief updates.

## 2. GoFlow fidelity

Released PPO, GoFlow objective/architecture, 105-input actor, 109-input privileged critic, action scaling, two-second episode and reward are retained. Custom grasp timing, physical staging, empirical reset initialization and the user-approved 1,000-step context-only flow likelihood initialization are explicit adaptations. The extra likelihood objective is absent from the released training procedure. Numerical epsilon/eta and an exact calibration recipe are absent from the released planner materials; diagnostic epsilon is inherited and not retuned. See experiments/connector_handoff/final_experiment.md and docs/method_fidelity.md.

## 3. Existing implementation inherited

The original 10,027,008-transition connector checkpoint and both earlier result directories are preserved. Original checkpoint SHA256: d2e75560f58ac9b4836193e3d437e0817fbcbb7d4eb652680abe40d3be8770e5. It achieved 33/100 flow and 2/100 uniform successes before this work.

## 4. Actual grasp terminal distributions

Initial inherited calibration: 100 executions per grasp, all locally successful; +/-3 mm planar placement, nominal initial yaw. Closure tilted the left-held connector. CSV/statistics/distribution/correlation plots are in calibration/. Revised timing was separately measured on 200 executions per grasp in calibration/aligned_complete_state/. The sampled domain is narrow; no broad six-axis robustness is claimed.

## 5. Handoff/staging alignment

Physical GRASP → elevated translation waypoint → common connector staging point → INSERT is executed without resetting robot or object. The common staging point is 10 mm above the former root initialization (60 mm above the goal). Measured hand orientation and hand-to-connector transform are preserved. The single measured constraint is created before closure in the revised macro. Final calibrated grasp/stage success is 400/400. Maximum stage hand error 0.689 mm; transform translation drift 0.0122 mm; rotation drift 0.000340 rad. Replay preserves authored constraint targets and actuator targets separately from contact-loaded poses. The shorter 30 mm staging probe failed Left and was not adopted.

## 6. INSERT retraining

Executed additional training:

- results/custom_connector/final_handoff_experiment/training/aligned_2m: 2,031,616 additional transitions; 1,048,576 cumulative PPO / 983,040 validation; 5 flow updates.
- results/custom_connector/final_handoff_experiment/training/empirical_init_2m: 2,031,616 additional transitions; 1,048,576 cumulative PPO / 983,040 validation; 5 flow updates.
- results/custom_connector/final_handoff_experiment/training/supported_3m: 3,014,656 additional transitions; 1,572,864 cumulative PPO / 1,441,792 validation; 7 flow updates.

The first tight-bound initialization starved both real grasp clusters. Data-based wider bounds improve initial support while retaining the released normalization and updates. The two earlier attempts and the approved empirical-initialization stage each retain the original actor/critic warm-start weights. The new stage first fits the existing flow to balanced empirical grasp contexts with independent uniform fixture angle, then uses unchanged online GoFlow updates. These are separate documented lineages, not one uninterrupted run.

Latest measured exposure:

```json
{
  "region_tolerance_yaw_x_y": [
    0.001,
    0.0005,
    0.0001
  ],
  "fixture_tolerance_rad": 0.2,
  "centers": [
    [
      -0.008369268849492073,
      -0.012018948793411255,
      -0.00041747093200683594
    ],
    [
      -0.0008484378922730684,
      0.011927545070648193,
      9.649991989135742e-05
    ]
  ],
  "counts": {
    "ppo": {
      "total": 1048576,
      "near_left": 150153,
      "near_right": 171181,
      "feasible_left_region": 9254,
      "feasible_right_region": 10658
    },
    "validation": {
      "total": 983040,
      "near_left": 188,
      "near_right": 51,
      "feasible_left_region": 0,
      "feasible_right_region": 0
    }
  },
  "scope": "Transition exposure; repeated episode contexts are not independent episodes."
}
```

## 7. Learned flow/value/precondition

| Fixture angle | Grasp | Policy successes | Mean return | Mean V | Mean density | Joint accepts |
|---|---|---:|---:|---:|---:|---:|
| -1.5708 | GraspLeft | 36/100 | 46.91 | 17.76 | 0.319 | 0/100 |
| -1.5708 | GraspRight | 0/100 | 15.35 | 15.78 | 0.449 | 0/100 |
| 1.5708 | GraspLeft | 0/100 | 11.96 | 12.85 | 0.365 | 0/100 |
| 1.5708 | GraspRight | 95/100 | 149.60 | 102.54 | 0.388 | 100/100 |

Fixed inherited custom diagnostic epsilon; published JT=50. Numerical epsilon/eta and a calibration protocol are absent from the release/paper. This is not a verified paper threshold.

Plots and samples: results/custom_connector/final_handoff_experiment/precondition_analysis/empirical_2m. No applicability threshold was lowered to admit a case.

Held-out random-context metrics:

```json
{
  "uniform": {
    "episodes": 100,
    "successes": 7,
    "mean_return": 22.098605793267488,
    "mean_goal_distance_m": 0.031458276361227035,
    "within_3mm": 7,
    "mean_value": 19.299399757385252,
    "mean_discounted_return": 16.463804489913375,
    "value_rmse": 27.6041264896238,
    "value_return_correlation": 0.5211477109432229,
    "value_success_auc": 0.8878648233486943,
    "density_return_correlation": 0.2569345612282985,
    "precondition_accepted": 0,
    "precondition_precision": null,
    "precondition_recall": 0.0
  },
  "flow": {
    "episodes": 100,
    "successes": 46,
    "mean_return": 65.04304470092057,
    "mean_goal_distance_m": 0.022280640999088063,
    "within_3mm": 46,
    "mean_value": 42.262586765289306,
    "mean_discounted_return": 45.80772550593956,
    "value_rmse": 26.902572225839535,
    "value_return_correlation": 0.7979494685939055,
    "value_success_auc": 0.8796296296296297,
    "density_return_correlation": -0.14402790773232513,
    "precondition_accepted": 32,
    "precondition_precision": 0.9375,
    "precondition_recall": 0.6521739130434783
  }
}
```

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

The acceptance failure is in low-level learned competence and/or learned support of real handoff states. Previous supported-bounds training had zero measured PPO exposure in either canonical feasible neighborhood. The approved empirical flow initialization has now executed. Latest measured exposure is recorded in the run handoff_exposure.json and the report summary; these are transition counts, not independent episode counts. Feasible physical handoffs exist. Neither prospective-inspection failure nor a GoFlow planning limitation can be inferred. Several library-import failures occurred before simulation; failed launches were preserved and excluded, with unchanged retries. Their system-level cause is not established.

Offline released-objective scale probe (not a causal intervention):

```json
{
  "physical_context_volume": 3.633291635196656e-05,
  "normalized_box_volume": 10000.0,
  "omitted_affine_log_jacobian": 19.43312644958496,
  "alpha": 0.5,
  "beta": 1.0,
  "loss_probe": {
    "reward": {
      "loss": 4.064839868130166e-11,
      "gradient_l2": 5.892852872335652e-09
    },
    "entropy": {
      "loss": -2.755468742066114e-09,
      "gradient_l2": 2.0230233488405247e-08
    },
    "similarity_mc_at_identical_weights": {
      "loss": 0.0,
      "gradient_l2": 0.5156211853027344
    }
  },
  "density_change": {
    "log_density_correlation": 0.98234118693842,
    "mean_absolute_log_density_change": 0.16110502183437347
  },
  "scope": "Offline held-out loss/gradient probe, not exact online updates or a causal intervention. Similarity loss is zero at identical weights; its finite-sample gradient need not be zero. No normalization, objective, optimizer or checkpoint was changed."
}
```

The fitted initialization now covers both modes, but density is not established as a learned success region. Inspect the measured loss-scale imbalance before further training; no objective correction was applied.

## 17. Videos / contact sheets / timelines

Complete actual oracle GRASP→Stage→INSERT video: calibration/oracle_handoff_video/handoff.mp4. Contact sheet: calibration/oracle_handoff_video/contact_sheet.png. These were visually inspected and show the actual task. Precondition slices are saved alongside the current precondition analysis referenced above. No 20-episode online videos, planner contact sheets or belief timelines exist because the gates have not passed.

## 18. Fidelity limitations

Gravity-disabled object and rigid-constraint grasp approximation; narrow planar grasp calibration; measured contact deflections and replay cannot restore PhysX internal contact caches; empirical box extends/interpolates the two macro modes; simulator/RNG state is not resumed across training sessions; numerical planner thresholds are underspecified by the release; the approved empirical flow likelihood initialization changes the released initialization objective. Earlier original-task reproduction is a functional baseline, not a claim of matching all paper metrics.

## 19. Scientific interpretation

Executed evidence cannot distinguish whether published GoFlow already solves prospective sensing in this two-grasp setting. Low-level acceptance must be resolved before studying the planner. No hidden-state rule, successor-conditioned grasp predicate or inspect-first heuristic has been added.

## 20. Conclusion label

**INCONCLUSIVE**

Git provenance at report generation:

```json
{
  "upstream_sha": "a8c6af5de7f427418783fd9faa20d50f38b734a9",
  "branch": "goflow-handoff-experiment",
  "head": "5bd85a5a91d9e37321453be3af23f7d6c3c11cd2"
}
```
