# Connector grasp gate and INSERT training

The predecessor actions are visible-pose IK macros, as requested in the original
specification. They are not separately trained grasp policies. The original
Gears baseline is recorded at a4e09aa; the first custom geometry commit is 10f704b.

`scripts/validate_connector_grasps.py` executes retreat/approach, finger closure,
measured attachment, lift, and hold. It disables the initial constraint before
approach and attaches only when the actual relative translation is within 3 mm
of the selected grasp transform. Success also requires >30 mm lift and <2 mm
relative translation drift throughout the final two-second hold. The fingers
close to 12 mm per joint. Both choices use identical local criteria; fixture
clearance never enters the grasp predicate. NPZ outputs expose the measured
relative pose as the action effect, plus hand/object poses and finger positions.

This uses the released gravity-disabled connector and fixed-grasp approximation;
it is not a frictional pickup benchmark. Object placement is visible, with
uniform +/-3 mm planar jitter, and yaw is nominal for this local macro gate.
The first 4-trial attempt used insufficient approach settling and failed all
four accuracy gates (~9 mm error). Increasing settling under unchanged gains
passed the four-trial smoke and then 50/50 trials for each grasp (seed 41000).
Outputs: `results/custom_connector/grasp_validation/`.

The closed-gripper geometry check exposed an intrinsic finger/socket-rim
collision: nominal cases bottomed out ~8 mm above the root goal. The recorded
video is in `results/custom_connector/geometry_closed_visual/`. The custom
connector now has a 24 mm longer exposed insertion tip (48 mm total axial body
length), and the socket is 24 mm lower. The grip region, root goal, fixture,
reward and context bounds stay the same. This geometry correction is explicitly
custom; it is not an original Gears modification. Revalidate the grasp and
clearance gates on this geometry before any PPO run.

The new tip geometry again passes 50/50 trials for each grasp, saved in
`results/custom_connector/grasp_validation_tip/`. Closed-finger clearance:
central both reach <0.014 mm; angle -pi/2 allows GraspLeft and blocks GraspRight
at 22.76 mm; angle +pi/2 blocks GraspLeft at 23.26 mm and allows GraspRight.
This side mapping supersedes the original open-finger prototype mapping.
Moving only the wall three times farther away lets all six closed-gripper cases
reach the goal (<0.090 mm final root error). The final macro gate's worst grasp
error is 2.522 mm; minimum lift 37.615 mm; worst hold translation drift 0.010 mm;
worst quaternion-derived angular drift ~0.089 degrees. All saved arrays are finite.

A further bounded check uses the actual two-second training duration:
`geometry_closed_tip_2s`. Both central cases and the two feasible side cases
reach <0.432 mm root error; the blocked counterparts remain ~25 mm away. Thus
the current geometry is solvable within the released training horizon by the
explicitly oracle pose controller; it does not establish learned competence.

INSERT uses `Connector-GOFLOW-v0`, the same privileged Gears agent YAML, and a
new seed-0 actor/critic/flow. Explicit custom changes: keyed connector/socket and
fixture; four context components (yaw +/-0.12 rad, grasp x +/-40 mm, grasp y
+/-4 mm, fixture angle +/-pi); yaw is physically applied; fingers initialize
closed at 12 mm to match macro closure. It retains the released two-second
episode, translational residual action scaling, position reward, actor 256/128/64,
and GoFlow update objectives/settings. Critic input grows from 108 to 109 due to
the extra context component; actor input remains 105. Grasp effects include small
measured errors beyond these four nominal context parameters. Full handoff
execution must measure their consequence, not reset them away and call it a trial.

Commands (project root):

```bash
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/validate_connector_grasps.py
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/probe_connector_geometry.py \
  --closed-gripper --output results/custom_connector/geometry_closed
GOFLOW_CPU_THREADS=16 GOFLOW_RUN_DIR=results/custom_connector/insert_seed0 \
GOFLOW_TRANSITION_BUDGET=5000000 GOFLOW_SAVE_EVERY=1000000 \
scripts/project_python.sh -u scripts/run_goflow.py --headless \
  --task Connector-GOFLOW-v0 --agent_config experiments/original_gears/privileged_goflow.yaml \
  --num_envs 1024 --seed 0 --exp_name connector_insert_seed0
scripts/project_python.sh -m unittest discover -s tests -v
```

The five existing component tests pass. An initial test launch failed inside
Python's import machinery while loading NumPy (`from_bytes` AttributeError);
an unchanged retry passed. This resembles earlier intermittent runtime failures
but its root cause is not established. No packages or system settings changed.

The first saved INSERT rollout verifies (32,1024,105) actor and (32,1024,109)
critic tensors: the critic prefix equals actor input, and its final four features
equal the sampled context. Every array is finite. Effective training rollout is
32768 transitions with minibatch 2048 and an enabled privileged critic.

Checkpoint evaluation uses the same stochastic actor and manually cleared
history as the working original-task evaluator:

```bash
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/evaluate_gears_milestones.py \
  results/custom_connector/insert_seed0 --task connector --milestones 1000000 3000000 5000000
scripts/project_python.sh scripts/analyze_connector_policy.py results/custom_connector/insert_seed0
```

`--grasp-fixture-cases --task connector --sampling nominal` in the evaluator
cycles four fixed grasp/fixture cases to check skill competence at the nominal
grasp transforms. This is an isolated policy diagnostic, not an executed
predecessor-to-successor handoff and not a planner experiment.

## First 5M result and bounded continuation

Seed 0 completed 5,013,504 transitions (2,555,904 PPO; 2,457,600 uniform validation)
in 554.371 agent seconds, with 12 flow updates. Training source: `197e15c`.
Final checkpoint SHA256:
`f1199121885254a615816871140b875a6575ddb7498a9f084ad8f17b4ecd369e`.

| Checkpoint | Uniform success | Flow success | Flow mean return |
|---|---:|---:|---:|
| ~1M | 0/32 | 2/32 | 23.167 |
| ~3M | 0/32 | 5/32 | 38.703 |
| ~5M | 0/100 | 18/100 | 34.598 |

The final training-phase success rate is 17.3%, versus ~5.6% near 1M and ~13.8%
near 3.3M. This is improving but weak, motivating one continuation toward 10M
without changing the task, objectives, seed or network. The held-out 3M/5M
sample sizes differ, so the small success increase is not a significance claim.
Paired nominal cases at 5M: allowed-left 1/10; blocked-right 0/10;
blocked-left 0/10; allowed-right 0/10. This fails the useful handoff-skill gate.

Weights are finite, true transition counters sum correctly, and the final
critic input check passes 105/109 separation and nonzero context sensitivity.
Complete trajectories/calibration are saved under the run directory. These
checks do not turn low success into a successful policy/planning result.

Continuation command (new output directory, cumulative budget):

```bash
GOFLOW_CPU_THREADS=16 GOFLOW_RUN_DIR=results/custom_connector/insert_seed0_to10m \
GOFLOW_TRANSITION_BUDGET=10000000 GOFLOW_SAVE_EVERY=2000000 \
scripts/project_python.sh -u scripts/run_goflow.py --headless \
  --task Connector-GOFLOW-v0 --agent_config experiments/original_gears/privileged_goflow.yaml \
  --num_envs 1024 --seed 0 --exp_name connector_insert_seed0_to10m \
  --checkpoint results/custom_connector/insert_seed0/checkpoints/final.pth
```

The resume path restores actor/critic/flow and optimizer state and counters.
Simulator state, random-generator progression and a partially completed update
phase restart, as explicitly documented for the original Gears continuation.

## Final 10M result: learned region, handoff gate not passed

The same seed finished at 10,027,008 cumulative transitions: 5,111,808 PPO and
4,915,200 validation, with 24 flow updates. Cumulative agent time is 1112.932 s
(18.55 minutes), excluding simulator startup and independent evaluation. Final
checkpoint: `results/custom_connector/insert_seed0_to10m/checkpoints/final.pth`.
SHA256: `d2e75560f58ac9b4836193e3d437e0817fbcbb7d4eb652680abe40d3be8770e5`.

| Checkpoint | Uniform success | Flow success | Flow mean return |
|---|---:|---:|---:|
| ~6M | 0/32 | 6/32 | 42.464 |
| ~8M | 1/32 | 5/32 | 28.326 |
| ~10M | 2/100 | 33/100 | 53.271 |

Final flow evaluation: privileged value/discounted-return correlation 0.853,
success AUC 0.952, RMSE 22.439. Learned log-density/return correlation is 0.412.
The joint value/density indicator accepts 28 episodes, of which 23 succeed:
82.1% precision and 69.7% recall. The density threshold is 0.0002145213 in
released normalized-coordinate units, the fifth percentile of 10,000 flow
samples selected without reward tuning. Uniform evaluation accepts no episodes.
These measurements support a useful learned region, not uniform competence or
a controlled causal claim that flow learning caused the improvement.

The paired nominal grasp/fixture checks use ten shared policy-noise seeds:

| Fixture angle | Grasp | Oracle feasible? | Stochastic policy | Mean-action policy (one case) |
|---|---|---|---:|---:|
| -pi/2 | Left | yes | 2/10 | failure, return 23.375 |
| -pi/2 | Right | no | 0/10 | failure, return 16.338 |
| +pi/2 | Left | no | 0/10 | failure, return 39.103 |
| +pi/2 | Right | yes | 2/10 | success, return 67.396 |

Mean-action evaluation is explicitly a diagnostic. The main evaluation and
trained critic concern the released stochastic policy; substituting mean-action
execution silently would change the policy whose precondition is being tested.
Four recorded mean-action videos and success/failure contact sheets are under
`insert_seed0_to10m/grasp_fixture_deterministic/`.

All four paired cases have initial values below JT=50: 32.72, 11.39, 11.81,
16.97, respectively. Thus the current joint precondition rejects these nominal
handoff contexts. Calling this a failure of prospective inspection or BFS would
confound the planner with inadequate low-level INSERT competence.

### Targeted diagnosis and next gate

At 5M, the centered nominal configuration succeeds in 4/4 stochastic episodes
(mean return 123.046), while the mean-action policy fails at either +/-12 mm
grasp offset even with the fixture in its clear central position (returns
13.569 and 17.298; ~24.7 mm final error). The final policy has improved, but
the nominal paired cases remain unreliable. The oracle controller reaches the
same feasible cases within two seconds, so an impossible horizon, missing
attachment or inherently blocked feasible geometry does not explain those
failures. The remaining issue is learned grasp-offset correction and coverage;
the exact optimization/observation cause is not yet isolated.

Next: validate actual macro terminal states against the INSERT reset distribution
and run a bounded offset-only diagnostic with a clear fixture. In particular,
the current macro ends after a 40 mm lift, whereas INSERT training resets to
the original pre-insertion hand pose; a transport/handoff stage has not yet
been implemented. Do not silently reset away that difference in an end-to-end
trial. Resolve this starting-state/offset-learning gate before more training or
online planner scoring. No learning beyond 10M, seed sweep, online INSPECT,
Bayes3D handoff integration or planner outcome is claimed here.

The grasp-validation rerun now starts both choices from the **same nominal
visible connector placement**, with +/-3 mm planar jitter. Earlier runs inherited
the grasp-dependent object placements from the INSERT reset and are preserved
as narrower checks. The corrected run again succeeds 50/50 for each grasp;
maximum grasp error 2.525 mm, minimum lift 37.604 mm, maximum hold drift 0.010 mm.
Outputs: `results/custom_connector/grasp_validation_common_pose/`.

Final commands, in addition to the training commands above:

```bash
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/evaluate_gears_milestones.py \
  results/custom_connector/insert_seed0_to10m --task connector \
  --milestones 6000000 8000000 10000000
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/eval_original_goflow.py \
  --task connector --checkpoint results/custom_connector/insert_seed0_to10m/checkpoints/final.pth \
  --agent_config experiments/original_gears/privileged_goflow.yaml --sampling nominal \
  --grasp-fixture-cases --episodes 40 --seed-base 50000 \
  --output results/custom_connector/insert_seed0_to10m/grasp_fixture_cases
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/eval_original_goflow.py \
  --task connector --checkpoint results/custom_connector/insert_seed0_to10m/checkpoints/final.pth \
  --agent_config experiments/original_gears/privileged_goflow.yaml --sampling nominal \
  --grasp-fixture-cases --deterministic --video --episodes 4 --seed-base 50000 \
  --output results/custom_connector/insert_seed0_to10m/grasp_fixture_deterministic
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/validate_connector_grasps.py \
  --output results/custom_connector/grasp_validation_common_pose
scripts/project_python.sh scripts/analyze_connector_policy.py \
  results/custom_connector/insert_seed0_to10m --milestone 10000000 \
  --prior-run results/custom_connector/insert_seed0
scripts/project_python.sh scripts/check_privileged_critic.py \
  results/custom_connector/insert_seed0_to10m --task connector
scripts/project_python.sh scripts/verify_connector_run.py \
  results/custom_connector/insert_seed0_to10m --budget 10000000
scripts/project_python.sh -m unittest discover -s tests -v
```

The five unit tests pass. Each session's artifact audit passes finite weights,
first/last rollout input separation, transition accounting, and all 328 held-out
evaluation traces (15,416 steps per session; excludes additional local probes).
Large outputs/checkpoints/videos remain ignored; code and findings are committed.
