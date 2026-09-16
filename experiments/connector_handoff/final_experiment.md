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
