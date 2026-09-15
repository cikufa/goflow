# Connector diagnostic: implementation contract

Current continuation: [grasp validation and INSERT training](training.md). The
geometry results below describe the original open-gripper prototype. Closed
fingers required a longer exposed connector tip; the updated geometry and macro
measurements are documented in the continuation.

Baseline rollback point: `a4e09aa6b125ea161c76ea98134dab45ae4b07e1`.
See `docs/gears_baseline.md` for the completed one-seed Gears gate and limits.

Begin with a native Isaac Lab scene and a bounded geometry/macro diagnostic,
before training INSERT. The existing Gears implementation and checkpoint remain
the reference for PPO, flow, observations, privileged critic and attachments.

## Geometry and controls

Use the released Franka, differential IK, history observations and one fixed
grasp constraint per environment. A keyed rectangular connector and matching
panel socket replace the gear and taskboard geometry. A neighboring fixture
wall has a continuous lateral position; it must physically obstruct gripper
clearance for one stable grasp while allowing the other. No collision penalty
or hidden-state rule may manufacture this asymmetry.

`GraspLeft` and `GraspRight` use visible-object IK macros, close the gripper and
produce different connector-to-hand transforms. Their preconditions concern
only reaching and holding the connector. Fixed constraints approximate stable
grasps consistently with the released simulator; they are not evidence of a
frictional grasp-quality model. Verify local grasp stability separately.

INSERT retains the released PPO/flow hyperparameters and actor architecture.
Its actor observes proprioceptive history; context is supplied only to the
critic. Context contains connector-to-hand x/y/yaw plus the physical fixture
position. Geometry, context bounds, duration and reward choices must be written
explicitly in its config, without changing the original Gears task.

## Gates before training and planning

1. Native scene loads and steps with finite state and one grasp constraint.
2. Both grasp macros achieve at least approximately 90% local stable-hold success.
3. With known geometry, switching grasps changes physical insertion feasibility
   in both fixture cases. A symmetric failure or success is a task-design defect.
4. A new INSERT policy learns a useful region before planner outcomes are scored.
5. Learned V and density produce the joint Equation-7 precondition; no manual
   successor-aware grasp predicate is allowed.

The planner uses the existing literal FIFO BFS and weighted joint indicator.
Belief/effect and observation assumptions must be documented. INSPECT updates
uncertainty through a rendered observation; the planner never receives the true
fixture side. Use the existing Bayes3D adapter when feasible, and distinguish an
explicit paper-observation-model fallback if an integration blocker remains.

This document starts the custom diagnostic after the committed baseline. It
does not report connector training, grasp success or a planning result.

## Implemented native scene and measured geometry gate

`environment.py` subclasses the released Gears configuration and environment.
It builds local compound USD colliders for a 70 x 24 x 24 mm rectangular
connector with an asymmetric key, a matching socket with 0.6 mm nominal side
clearance, and a 12 x 160 x 150 mm neighboring fixture wall. It retains the
released robot, actuator settings, IK/residual controller, observation history,
position reward and default two-second episode. No Gears source globals change.
Generated USD files live under `.cache/connector_assets/`, keyed by source hash.

The wall's continuous angle controls its position and orientation around the
socket: x=120*cos(angle) mm, y=50*sin(angle) mm. This provides realistic side
clearance variation while avoiding an intrinsically blocked central placement.
The two diagnostic grasp transforms displace the hand by +/-12 mm along the
connector length. These are pre-attached transforms, **not yet executed pickup
macros**. The fixed-grasp approximation and original finger configuration remain
in this geometry test; physical macro execution still needs separate validation.

Explicit custom domain, in order: yaw +/-0.12 rad, grasp x +/-0.04 m, grasp y
+/-0.004 m, fixture angle +/-pi rad. Unlike released Gears, sampled yaw is applied
to the connector attachment. The actor remains 105-dimensional with three
translational actions; the privileged critic input adds these four parameters
for 109 dimensions. The geometry probe checks actual input separation.

The bounded oracle probe first aligns laterally above the socket and then
descends. It uses true connector pose solely to test geometry. Its explicit
six-second diagnostic allowance is not a PPO training configuration and does
not change the original two-second Gears task. It records all 143 control steps,
commands/clipped actions, observations, rewards, poses, distances and done flags.

| Fixture placement | Hand along -Y (GraspLeft transform) | Hand along +Y (GraspRight transform) |
|---|---:|---:|
| Central, angle 0 | 0.011 mm final error | 0.012 mm |
| Left-constrained, angle -pi/2 | 23.035 mm | 0.012 mm |
| Right-constrained, angle +pi/2 | 0.015 mm | 23.402 mm |

Moving only the fixture three times farther away makes all six cases reach the
goal (maximum final error 0.021 mm). This supports a physical fixture-clearance
effect rather than an intrinsically bad grasp transform or an imposed failure
predicate. It does not establish 90% pickup-macro reliability, policy competence,
perception validity or any high-level planning result.

Exact commands from the project root:

```bash
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/probe_connector_geometry.py \
  --episode-seconds 6 --output results/custom_connector/geometry_probe_v2
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/probe_connector_geometry.py \
  --episode-seconds 6 --fixture-distance-scale 3 \
  --output results/custom_connector/geometry_probe_wall_away
GOFLOW_CPU_THREADS=16 scripts/project_python.sh -u scripts/probe_connector_geometry.py \
  --episode-seconds 6 --case 3 --video --output results/custom_connector/geometry_visual
```

Each run saves `summary.json`, `trajectories.npz` and its shell-redirected
`run.log`; the visual run also saves `probe.mp4`. Launches used a 180-second
process timeout. Earlier prototypes are retained separately: a two-second
unstaged controller failed to isolate geometry; a first six-second layout
showed the desired side pattern but an obstructed central placement. Those
findings led to lateral staging and the explicit elliptical wall placement.

Next required gate: execute and validate both visible-object grasp macros, then
train INSERT with the restored Gears PPO/flow pipeline. No INSERT PPO run has
started, and no planner conclusion follows from this geometry test.

The final six-case rerun in `results/custom_connector/geometry_validation`
matches the v2 distances exactly and passes actual 105/109 input separation and
one-grasp-joint-per-environment checks. Robot-internal fixed joints are excluded
from the grasp count. The one-case visual probe independently reproduces the
blocked case at 23.034 mm final error and saves a video/contact sheet. The five
existing unit tests pass; these are component tests, not grasp-macro validation.
