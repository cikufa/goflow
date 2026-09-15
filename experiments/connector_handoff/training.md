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
