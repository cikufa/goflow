# Final handoff experiment: acceptance gates

Result root: `results/custom_connector/final_handoff_experiment/`. The earlier
5M and 10M INSERT runs are preserved. No planner outcome is inferred until
physical handoff, learned competence/precondition, and real perception all pass.
No successor-aware grasp predicate or inspect-before-grasp rule is permitted.

## Measured predecessor and staging

`scripts/calibrate_connector_handoff.py` first measured the inherited macros on
100 executions per grasp, seed 61000, independent uniform +/-3 mm planar object
placement with nominal visible yaw. Both grasps succeed 100/100. CSV/NPZ records
contain actual hand/object poses, hand-to-connector transforms, Euler angles,
joint state, object velocity, drift, and control trajectories. Coordinates named
`hand_local` are environment-relative; world origins are saved in the NPZ.
This is a narrow planar placement calibration, not broad six-axis robustness.

The inherited left macro closes the fingers before constraining the object.
Closure produces approximately -1.43 mm lateral error, -1.46 degree roll and
-0.37 degree yaw. Even with the fixture moved away, the oracle translation-only
controller fails from this held pose. Grasp hold success alone was insufficient.

A separately labeled rigid-grasp timing correction enables the single constraint
at the **measured approach pose**, after the same <3 mm local accuracy check,
before closure. It never snaps to the ideal grasp transform. This changes the
custom macro's timing, not the original Gears environment or GoFlow algorithm.
It retains the existing gravity-disabled object/fixed-constraint approximation;
it is not a frictional pickup validation. Finger/object contacts remain active.
Residual constraint deflection and roll/pitch/z errors are measured, not erased.

Neutral staging translates the held connector toward the known common starting
connector position, 10 mm above the previous INSERT root initialization. The hand
target is computed from the measured held transform. Its orientation remains the
measured grasp-terminal hand orientation. It moves via a waypoint 40 mm above
the target, descends, and holds for 192 control steps. Both choices use identical
logic. No fixture state is read by this primitive. The two grasps retain distinct
hand-to-connector transforms and therefore distinct hand positions at staging.
Collision checking remains active throughout. Common-hand-position alternatives
are preserved as diagnostics: they caused connector/fixture contact at staging.

Final calibration: 200 executions per grasp, seed 62000, with 100 trials in each
grasp/fixture cell. Every grasp and stage succeeds. Oracle return>=50 success is
99/100 for Left at -pi/2, 100/100 for Right at +pi/2, and 0/100 in both opposite
pairings. Mean final goal distances are 0.108 mm, 0.049 mm, and about 22--23 mm
for blocked cases. Left is physically feasible but has little return margin.
Existing 10M stochastic policy on the same physical handoffs: 0/100 feasible Left,
79/100 feasible Right, 0/100 in both blocked cells. No new PPO preceded this test.

Reproduction commands, from the project root:

```bash
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/calibrate_connector_handoff.py
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/calibrate_connector_handoff.py \
  --trials-per-grasp 200 --seed 62000 --attach-before-close \
  --stage-frame connector --stage-height .01 --stage-hold-steps 192 \
  --insert-controller oracle \
  --output results/custom_connector/final_handoff_experiment/calibration/aligned_oracle
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/calibrate_connector_handoff.py \
  --trials-per-grasp 200 --seed 62000 --attach-before-close \
  --stage-frame connector --stage-height .01 --stage-hold-steps 192 \
  --insert-controller policy \
  --output results/custom_connector/final_handoff_experiment/calibration/aligned_existing_policy
```

Completed calibration directories refuse overwrite; choose a new output for a rerun.

## Empirical INSERT initialization

`aligned_initialization.json` records the 400-state bank hash and new context
bounds: empirical extrema plus 5% padding (minimum 0.1 mrad yaw / 0.05 mm x,y).
The continuous bounding box includes intermediate grasp offsets absent from the
two macro clusters. That extension is explicit: the released continuous flow
does not directly represent two discrete grasp identities. Fixture angle retains
[-pi,pi] and the exact existing elliptical-position/rotating-wall geometry. At
-pi/2 the wall is on negative world y; at +pi/2 on positive world y. The geometric
reversal was measured, not inserted as a symbolic rule.

`AlignedConnectorEnv` draws context using the released distribution, chooses
uniformly among the 16 nearest empirical contexts in normalized coordinates,
and restores the associated robot configuration/velocity and residual roll,
pitch, axial offset and object velocity. The sampled context supplies actual
relative yaw/x/y. The object root is reconstructed consistently with the saved
hand pose and this relative transform. These are physical reset states; no
trajectories or demonstrations are used as a learning target. Actor input remains
105 and excludes context; critic remains 109. Reward, 2-second episode, action
scaling, PPO architecture/settings and GoFlow objectives remain released.

Warm-start preserves the existing actor/critic weights and normalization state.
The changed context bounds change flow normalization, so the old flow is **not**
reinterpreted under new bounds. Flow and optimizers start fresh; additional
transition counters start at zero and the inherited 10,027,008 are recorded
separately. The old checkpoint is preserved. This is an aligned warm-start
experiment, not an uninterrupted continuation under identical initialization.

Gate B requires approximately 70%+ feasible policy success and <10% blocked
success, plus meaningful learned preconditions. Gate C requires real rendered
observations and validated pose/fixture beliefs. Scientific trials remain gated.

### Reset replay correction

A narrow replay exposed a consequential distinction: measured held pose differs
from the authored joint rest transform under finger contact. Using the measured
pose as the new rest transform doubled the deflection and made feasible Left
fail. The first training process was stopped during simulator startup, before
PPO. `calibration/aligned_complete_state` now also records joint rest transforms
and actuator targets. Aligned reset preserves both, applying only the sampled
context delta. The reset probe now reaches 0.107 mm and 0.050 mm in feasible
Left/Right cases, returns 53.46/96.03, while blocked cases remain >22 mm away.
Its purpose is alignment validation, not a second handoff success estimate.

```bash
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/probe_connector_geometry.py \
  --aligned-init --episode-seconds 2 \
  --output results/custom_connector/final_handoff_experiment/calibration/reset_complete_state
scripts/project_python.sh scripts/prepare_connector_warm_start.py \
  --source results/custom_connector/insert_seed0_to10m/checkpoints/final.pth \
  --output results/custom_connector/final_handoff_experiment/training/warm_start.pth
GOFLOW_CPU_THREADS=16 \
GOFLOW_RUN_DIR=results/custom_connector/final_handoff_experiment/training/aligned_2m \
GOFLOW_TRANSITION_BUDGET=2000000 GOFLOW_SAVE_EVERY=1000000 \
scripts/project_python.sh -u scripts/run_goflow.py --headless \
  --task ConnectorAligned-GOFLOW-v0 \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --num_envs 1024 --seed 0 --exp_name connector_handoff_aligned_2m \
  --checkpoint results/custom_connector/final_handoff_experiment/training/warm_start.pth
```

### Bounded 2M result and initial sampling support correction

The first aligned stage completed 2,031,616 additional transitions (1,048,576 PPO,
983,040 uniform validation). Its actual grasp clusters received **zero** nearby
PPO transitions, versus 1,685 Left and 2,433 Right in validation. Near means
within 1 mrad yaw, 0.5 mm x, and 0.1 mm y of a measured cluster median, with fixture
unrestricted. This is a sampling configuration failure: released normalization
maps a context box to width 10 while the initial base distribution has unit SD.
A tight bounding box puts both measured modes near its corners, with initial
joint densities around 1e-14. The final 2M policy reaches 22/100 feasible Left, 0/100 feasible Right, and 0/100 in both blocked cells; all 400 joint preconditions reject. At 1M, all four cells scored 0/10. Flow/uniform held-out success is 41/100 and 27/100, which does not establish handoff competence.

`aligned_initialization_supported.json` instead derives bounds from pooled
empirical mean +/-5 SD, so the released width/10 sampling SD reflects the measured
spread. Grasp x is capped at +/-30 mm within the 70 mm connector body. This widens
the continuous training extension, explicitly beyond the observed two clusters;
it does not modify the released flow normalization, base distribution or updates.
The same 100,000-sample initialization probe now visits 8/59 nearby Left/Right
contexts, versus 0/0 before. This is improved support, not balanced sampling or a
competence claim. Six reset oracle cases remain unchanged after this correction.

The next bounded stage uses 3M transitions under these bounds. It restarts from
the original 10M actor/critic rather than the 2M checkpoint which lost the
right-handoff skill. Flow/optimizers are fresh as previously documented. All
versions, checkpoints and results remain available. Total additional compute
includes both runs; they are not misrepresented as one uninterrupted lineage.
No planner or perception evaluation is authorized by these preliminary outcomes.

The diagnostic Equation-7 analysis retains epsilon=0.0002145212929463014 from
the prior custom analysis, without fitting it to aligned outcomes. JT remains 50.
The paper and release do not give numerical epsilon/eta or an exact calibration
protocol for this task. Thus this epsilon is an explicit adaptation assumption,
not a claimed published threshold. Planner thresholds remain unresolved and no
final trials are run with a silently chosen value.


## Latest bounded result and stop decision

The supported-bounds stage completed 3,014,656 transitions (1,572,864 PPO;
1,441,792 validation), seven flow updates, 327.40 agent seconds. Across both
aligned attempts, additional compute totals 5,046,272 transitions and 535.99 agent
seconds, excluding startup and independent evaluation. The latest checkpoint is
`training/supported_3m/checkpoints/final.pth` under the final-experiment root,
SHA256 `581a532b2344d4dc704a3ad3dcbd727b72d2bcb63c4cfc60100c6d3861d581e4`.
It is a diagnostic checkpoint, not an accepted/frozen low-level system.

| Stage | Feasible Left | Feasible Right | Blocked pairings |
|---|---:|---:|---:|
| Supported 1M | 3/10 | 1/10 | 0/10 each |
| Supported 2M | 1/10 | 7/10 | 0/10 each |
| Supported 3M | 6/100 | 67/100 | 0/100 each |

There is no sustained Left improvement, so the optional additional 5M was not
launched. Held-out flow/uniform success is 43/100 and 12/100. Flow-evaluation
value/discounted-return correlation is 0.801, success AUC 0.911, log-density/return
correlation 0.209. The existing five-percentile diagnostic calibration accepts
43 flow episodes, with 86.0% precision/recall. This useful region does not cover
the real grasp handoffs: all 400 are rejected at both the fixed inherited epsilon
0.0002145213 and the independent five-percentile estimate 0.0002254338.

Near the actual Left/Right grasp modes, PPO saw 47/713 transitions; when fixture
angle is also within 0.2 rad of each feasible canonical case, both counts are
zero. Near-mode tolerances: yaw 1 mrad, x 0.5 mm, y 0.1 mm. These are diagnostic
neighborhoods, not proof that no generalization is possible. They explain why
more raw transitions alone are not a justified next step. The critic predicts
55.40 on feasible Left despite only 6% success; low density appropriately excludes
this unreliable prediction. Mean V for feasible Right is 34.57, below JT=50.

A lower, 30-mm staging point was physically tested and made Left infeasible; it
was not adopted. The current scene, reward, threshold and validated 60-mm staging
remain unchanged. The complete oracle handoff video is in
`calibration/oracle_handoff_video/handoff.mp4`, with `contact_sheet.png`.
The earlier single-sample correlation plot and off-scene camera attempts remain
as debug artifacts; neither is presented as a successful visualization.

`proposed_empirical_flow_start.json` and `scripts/prepare_empirical_flow_start.py`
make the next possible adaptation reviewable. Default execution only validates
inputs and writes a proposal. No likelihood fit or subsequent PPO has executed.
It would initialize the **existing** flow with balanced empirical grasp contexts
and independent uniform fixture angle, using no success labels or planner
outcomes. Its additional likelihood objective is absent from the released
procedure; user approval was requested before executing it. No automatic
inspection rule, successor model, actor privilege or new planner objective is
part of the proposal.

The five unit tests and both checkpoint/trajectory audits pass. Each random
held-out evaluation audit covers 200 episodes / 9,400 steps, in addition to
physical handoff diagnostics. Native library-import failures were preserved and
retried, excluded from success denominators; their system-level cause is not
established. No system changes or installations were made.

## Approved empirical initialization stage

The user explicitly approved the preceding proposal. Execute 1,000 context-only
likelihood steps, check finite flow samples and coverage of both empirical modes,
then run a new bounded 2M-transition PPO/GoFlow stage. Actor/critic weights come
from the preserved original custom 10M warm start, not either failed aligned run.
The fixture angle remains independent and uniform in the initialization target.
No success labels or feasible-pair labels enter fitting. All prior results remain
intact. Evaluate actual four-cell handoffs and random contexts before Gate B.

```bash
GOFLOW_CPU_THREADS=16 scripts/project_python.sh scripts/prepare_empirical_flow_start.py --fit
GOFLOW_HANDOFF_SPEC=experiments/connector_handoff/aligned_initialization_supported.json \
GOFLOW_CPU_THREADS=16 \
GOFLOW_RUN_DIR=results/custom_connector/final_handoff_experiment/training/empirical_init_2m \
GOFLOW_TRANSITION_BUDGET=2000000 GOFLOW_SAVE_EVERY=1000000 \
scripts/project_python.sh -u scripts/run_goflow.py --headless \
  --task ConnectorAligned-GOFLOW-v0 \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --num_envs 1024 --seed 0 --exp_name connector_handoff_empirical_init_2m \
  --checkpoint results/custom_connector/final_handoff_experiment/proposal/empirical_flow_start.pth
```
