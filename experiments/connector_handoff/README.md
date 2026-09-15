# Active clean clone

All reproduction work and commits now use `/home/shekoufeh/goflow/reproduction` on `goflow-handoff-experiment`. The pre-existing parent checkout and its deletions are preserved. Official upstream is recorded in `docs/upstream.json`. The compatible environment at `/home/shekoufeh/goflow/.conda/envs/goflow-repro` and its existing Isaac Lab dependency are reused. `scripts/project_python.sh` writes only inside this clone and the shared environment; it keeps the rest of the host read-only.

From the clean clone, install GoFlow without reinstalling dependencies:

```bash
scripts/project_python.sh -m pip install --no-deps --no-build-isolation -e .
```

The sections below retain the initial environment installation record. Their environment-creation commands were run in the parent workspace before this clean clone was selected. Simulator/check scripts must now be run from the clean clone.

# Connector handoff reproduction

**Status:** the minimal Gears baseline gate passes: 6/100 uniform and 59/100 learned-flow successes at 10M; the joint precondition selects 54 flow episodes with 49 successes. See [baseline and limitations](../../docs/gears_baseline.md) and [exact recovery commands](../../docs/gears_recovery.md). Native connector geometry and oracle clearance tests now run; see [design, commands and measured gate](design.md). Pickup macros, INSERT training, online planning and balanced handoff trials remain pending. The failed 64-env [continuation](../../docs/gears_single_seed_continuation.md) and earlier pilots are historical.

The initial working tree contained deletions of almost all upstream files. The user selected a fresh subdirectory clone; those deletions remain untouched.

## Versions

- Official GoFlow main: `a8c6af5de7f427418783fd9faa20d50f38b734a9`.
- Isaac Lab v1.4.1: `a520a883ce996d855cc9d5255d71fd1c1307633f`.
- Isaac Sim 4.2.0.2, Python 3.10.21, Torch 2.4.0+cu118, torchvision 0.19.0+cu118.
- RL Games 1.6.1, Zuko 1.3.1, NumPy 1.26.4.
- GPU: RTX 4090; existing NVIDIA driver 580.105.08 unchanged.

GoFlow does not pin its original environment. These compatibility selections preserve its older IsaacLab API. See [setup audit](../../docs/environment_setup_plan.md), [fidelity audit](../../docs/method_fidelity.md), and [validation record](../../docs/environment_validation.json).

## Environment creation and activation

All commands run from `/home/shekoufeh/goflow`. The following creation command was executed after a dry run showing about 40 MB of Conda package downloads:

```bash
mkdir -p .cache/conda/pkgs .cache/tmp .cache/pip .conda/envs
CONDA_REGISTER_ENVS=false \
CONDA_PKGS_DIRS=/home/shekoufeh/goflow/.cache/conda/pkgs \
CONDA_ENVS_PATH=/home/shekoufeh/goflow/.conda/envs \
TMPDIR=/home/shekoufeh/goflow/.cache/tmp \
PYTHONDONTWRITEBYTECODE=1 CONDA_NO_PLUGINS=true \
/home/shekoufeh/miniconda3/bin/conda create -y --json --solver classic \
  --override-channels -c conda-forge \
  -p /home/shekoufeh/goflow/.conda/envs/goflow-repro \
  python=3.10.21 pip=26.2.1 > .cache/conda-create.json
```

Activation for interactive use:

```bash
source /home/shekoufeh/miniconda3/etc/profile.d/conda.sh
conda activate /home/shekoufeh/goflow/.conda/envs/goflow-repro
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

For all installation and runtime commands below, use `scripts/project_python.sh`. It selects the prefix explicitly, removes inherited ROS/user-site imports, redirects caches, and uses the existing bubblewrap executable to prevent host writes outside the project. No base environment, driver, or system CUDA changes are needed. The environment is intentionally not registered in the external Conda environment list.

## Executed installation commands

These commands produced the installed stack. `constraints.txt` includes the AWS compatibility pins discovered during startup; failed attempts and their fixes are documented in [compatibility changes](../../docs/compatibility_changes.md).

```bash
mkdir -p .deps
git clone --depth 1 --branch v1.4.1 https://github.com/isaac-sim/IsaacLab.git .deps/IsaacLab
scripts/project_python.sh -m pip install -c constraints.txt \
  torch==2.4.0 torchvision==0.19.0 numpy pillow sympy \
  --index-url https://download.pytorch.org/whl/cu118
scripts/project_python.sh -B scripts/audit_isaac_downloads.py
scripts/project_python.sh -m pip install -c constraints.txt --only-binary=:all: \
  isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 \
  isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 \
  --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple
scripts/project_python.sh -m pip install -c constraints.txt \
  boto3 botocore s3transfer setuptools==69.5.1 toml
scripts/project_python.sh -m pip install poetry-core==1.9.1
scripts/project_python.sh -m pip install -c constraints.txt --no-build-isolation \
  -e .deps/IsaacLab/source/extensions/omni.isaac.lab \
  -e .deps/IsaacLab/source/extensions/omni.isaac.lab_assets \
  -e '.deps/IsaacLab/source/extensions/omni.isaac.lab_tasks[rl-games]' \
  zuko pybullet tensorboardX imageio-ffmpeg
```

The NVIDIA payload audit found 6.013 GB of Isaac Sim wheels plus a 63.4 MB Omniverse Kit wheel. No full asset pack was downloaded. The user explicitly accepted the applicable Isaac Sim license in this session. The smoke script records that authorization through the required process environment variable.

## Validated checks

```bash
scripts/project_python.sh -m pip check
scripts/project_python.sh scripts/check_flow_compatibility.py
scripts/project_python.sh -u scripts/smoke_isaac.py
```

Results: no broken requirements; unchanged upstream flow classes pass GPU sampling/density/gradient checks; AppLauncher completes 10 GPU physics steps on cuda:0 and exits 0. The smoke marker says `steps_completed`; also check process exit status and logs because default Kit shutdown can terminate Python before code following app.close runs. These are infrastructure checks, not policy success measurements.

## Environment exports

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 CONDA_REGISTER_ENVS=false \
  CONDA_PKGS_DIRS=/home/shekoufeh/goflow/.cache/conda/pkgs \
  CONDA_NO_PLUGINS=true PYTHONDONTWRITEBYTECODE=1 \
  /home/shekoufeh/miniconda3/bin/conda env export \
  -p /home/shekoufeh/goflow/.conda/envs/goflow-repro > environment.yml
scripts/project_python.sh -m pip freeze --all > requirements_frozen.txt
```

The exports record the installed state, including editable Lab paths. Some Conda packages report build-time file URLs in pip freeze; use the Conda versions and installation commands above to recreate those packages. Nothing was installed with system pip.

## Original Gears commands executed in the clean clone

```bash
GOFLOW_RUN_DIR=results/original_gears/smoke GOFLOW_TRANSITION_BUDGET=1024 \
  scripts/project_python.sh -u scripts/run_goflow.py --headless \
  --task Gears-GOFLOW-v0 --num_envs 4 --video --enable_cameras \
  --video_length 200 --video_interval 100000 --exp_name smoke

GOFLOW_RUN_DIR=results/original_gears/released_training_64 \
GOFLOW_TRANSITION_BUDGET=1000000 GOFLOW_SAVE_EVERY=100000 \
  scripts/project_python.sh -u scripts/run_goflow.py --headless \
  --task Gears-GOFLOW-v0 --num_envs 64 --seed 0 --exp_name released_baseline_64

scripts/project_python.sh scripts/summarize_training_rollouts.py \
  results/original_gears/released_training_64
scripts/project_python.sh -u scripts/eval_original_goflow.py \
  --checkpoint results/original_gears/released_training_64/checkpoints/final.pth \
  --episodes 10 --output results/original_gears
scripts/project_python.sh -u scripts/eval_original_goflow.py \
  --checkpoint results/original_gears/released_training_64/checkpoints/final.pth \
  --episodes 3 --sampling nominal --video --output results/original_gears/nominal_visual
scripts/project_python.sh -u scripts/eval_original_goflow.py \
  --checkpoint results/original_gears/released_training_64/checkpoints/final.pth \
  --episodes 1 --sampling nominal --control down --video \
  --output results/original_gears/diagnostic_down
scripts/project_python.sh -u scripts/check_gears_attachments.py
scripts/project_python.sh -m unittest discover -s tests -v
```

The failed 1,024-env attempt used the same training command with `--num_envs 1024`, output `results/original_gears/released_training`, and experiment name `released_baseline`. It produced zero transitions.

## Next execution gate

The original Gears sanity baseline is committed at `a4e09aa`. The custom native
geometry gate now shows the intended grasp/fixture asymmetry, including a
wall-away control. The next gate is reliable execution of both pickup macros,
followed by INSERT learning. Do not infer a handoff-planning result from the
geometry probe. Initial rollback points remain `b0a6ae1` (released-code pilot),
`19390ef` (constraint fix) and `ddc238c` (critic profile).

Connector training and balanced handoff evaluation commands do not yet exist because those stages have not been implemented. The available original-task and component-check commands are documented below; no custom results are fabricated.

## Corrected and privileged pilots

```bash
GOFLOW_ATTACHMENT_REPORT=attachments_after_fix.json scripts/project_python.sh -u scripts/check_gears_attachments.py
GOFLOW_RUN_DIR=results/original_gears/corrected_training_64 \
GOFLOW_TRANSITION_BUDGET=1000000 GOFLOW_SAVE_EVERY=100000 \
  scripts/project_python.sh -u scripts/run_goflow.py --headless --task Gears-GOFLOW-v0 \
  --num_envs 64 --seed 0 --exp_name corrected_baseline_64
scripts/project_python.sh -u scripts/eval_original_goflow.py \
  --checkpoint results/original_gears/corrected_training_64/checkpoints/final.pth \
  --episodes 10 --released_history_reset --output results/original_gears/corrected_evaluation

GOFLOW_RUN_DIR=results/original_gears/privileged_smoke_64 \
GOFLOW_TRANSITION_BUDGET=4096 GOFLOW_SAVE_EVERY=4096 \
  scripts/project_python.sh -u scripts/run_goflow.py --headless --task Gears-GOFLOW-v0 \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --num_envs 64 --seed 0 --exp_name privileged_smoke_64
GOFLOW_RUN_DIR=results/original_gears/privileged_training_64 \
GOFLOW_TRANSITION_BUDGET=2000000 GOFLOW_SAVE_EVERY=200000 \
  scripts/project_python.sh -u scripts/run_goflow.py --headless --task Gears-GOFLOW-v0 \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --num_envs 64 --seed 0 --exp_name privileged_baseline_64
scripts/project_python.sh scripts/check_privileged_critic.py results/original_gears/privileged_training_64
```

## Final independent Gears evaluations and visualizations

```bash
scripts/project_python.sh -u scripts/eval_original_goflow.py \
  --checkpoint results/original_gears/privileged_training_64/checkpoints/final.pth \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --episodes 10 --output results/original_gears/privileged_independent_uniform
scripts/project_python.sh -u scripts/eval_original_goflow.py \
  --checkpoint results/original_gears/privileged_training_64/checkpoints/final.pth \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --episodes 10 --sampling flow --seed-base 20000 --video \
  --output results/original_gears/privileged_independent_flow
scripts/project_python.sh scripts/summarize_training_rollouts.py results/original_gears/privileged_training_64
scripts/project_python.sh scripts/export_central_value_losses.py results/original_gears/privileged_training_64 \
  --tensorboard 'logs/2026-09-15_00-58-54_privileged_baseline_64_Gears-GOFLOW-v0_GOFLOW_0_[]/summaries'
scripts/project_python.sh scripts/plot_original_goflow.py results/original_gears/privileged_training_64 \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --observation results/original_gears/privileged_independent_uniform/episode_000.npz
for video in results/original_gears/privileged_independent_flow/videos/*.mp4; do
  scripts/project_python.sh scripts/make_video_contact_sheet.py "$video" \
    --output "results/original_gears/privileged_independent_flow/contact_sheets/$(basename "${video%.mp4}").png" \
    --title 'Independent Gears evaluation; not a connector trial'
done
scripts/project_python.sh scripts/make_video_contact_sheet.py \
  results/original_gears/privileged_independent_flow/videos/episode_000.mp4 \
  --output results/original_gears/privileged_independent_flow/contact_sheets/episode_000.png \
  --raw-dir results/original_gears/privileged_independent_flow/raw_frames/episode_000
scripts/project_python.sh scripts/summarize_original_goflow.py
```

The TensorBoard path above is the actual executed run; a rerun prints its new timestamped log path. Earlier evaluations retained the upstream history-reset bug and are preserved separately. `--released_history_reset` reproduces that behavior; final independent evaluations match training reset history. These are Gears tests, not the requested 20 balanced connector trials.

## Bayes3D and planner component checks

The separate perception environment and exact installation commands are documented in [Bayes3D audit](../../docs/bayes3d_feasibility.md). Its complete installed exports are `environment_bayes3d.yml` and `requirements_bayes3d_frozen.txt`. Three declared optional-library dependencies remain omitted; the renderer/inference subset is explicitly identified in the audit. The Isaac environment still passes `pip check`.

```bash
scripts/bayes3d_python.sh -u scripts/smoke_bayes3d.py
scripts/bayes3d_python.sh -u scripts/check_bayes3d_pose_inference.py
scripts/project_python.sh scripts/plot_bayes3d_probe.py
scripts/project_python.sh -m unittest discover -s tests -v
```

Synthetic pose-inference evidence is also preserved under `results/infrastructure/`. These tests do not constitute an online original planning/inspection reproduction. Generic BFS and Equation 7 are in `experiments/common/belief_space.py`; task-specific effects, belief equality/threshold calibration and camera integration remain outstanding.

## Deterministic Gears solvability diagnostic

Run from the clean reproduction clone (no checkpoint or training needed):

```bash
mkdir -p results/original_gears/scripted_diagnostic
scripts/project_python.sh -u scripts/diagnose_gears_scripted.py > results/original_gears/scripted_diagnostic/run.log 2>&1
scripts/project_python.sh -m unittest discover -s tests -v > results/original_gears/scripted_diagnostic/tests.log 2>&1
```

The probe runs one nominal episode for each of three controllers, seed 10000,
with one environment and an assertion of exactly one hand/gear grasp joint.
Released reward, context bounds, episode configuration, fixed orientation and
5 cm residual scaling remain unchanged. The existing evaluator's explicit
history clearing avoids stale history between independent one-env resets.
Initial gear poses differ by at most 1.17e-9 in position/quaternion components.
The released timeout sets both terminated and truncated at step 47.

Zero action is a zero **residual**, retaining the default guidance. Constant
`z=-1` adds another 5 cm downward IK target displacement. The staged probe uses
10 cm downward target increments until height error is below 1 cm, then ten
times the remaining positive height error, capped at 10 cm. It compensates the
default vertical guidance with its residual, clips to [-1,1], and retains zero
lateral residuals and the released fixed orientation. Positive residual z in
the fine stage brakes the default descent. Default lateral guidance remains.
The existing `get_scripted_actions` helper cycles all six signed translation
directions; it is not an insertion controller.

Measured results (2026-09-15; all 47 steps, ending at the released time limit):

| Controller | Return | Final distance | Minimum distance | Within 1 mm |
|---|---:|---:|---:|---|
| Zero residual | 33.3901 | 2.088 mm | 2.088 mm | Never |
| Constant down | 30.1668 | 43.121 mm | 0.448 mm | One step; not at end |
| Staged | 223.2169 | 0.301 mm | 0.289 mm | Final 18 steps |

Distances above are Euclidean gear-root distances to the released goal. The
NPZ also records the exact SE(3)-based reward distance: it need not equal the
Euclidean metric when rotation drifts. Verified recorded rewards against the
released clipped reciprocal formula. All five existing unit tests pass.

Trajectories are `results/original_gears/scripted_diagnostic/{zero,down,staged}.npz`;
`summary.json` records results, attachment targets/offsets and measurement
semantics. Every step contains raw pre-action observation, commanded and clipped
residual actions, default action, resulting IK translation/target, pre/post gear
and end-effector poses (local environment frame; quaternion wxyz), reward,
both distances, goal error, proximity indicator, stage and both done flags.
Post-step snapshots are captured inside `_get_rewards` before automatic reset,
including the terminal step. No contact sensor is configured; the existing
`get_ee_force` joint-force projection is not treated as a contact detector.
There is no native insertion flag: the 1 mm diagnostic threshold measures goal
proximity, **not contact-confirmed seating**. Return >=50 likewise does not
certify seating.

The initial slower staged probe (2 cm coarse / 4 mm fine increments) reached
only 30.692 mm distance within the time limit. Its traces and summary remain in
`first_pass/`; the final controller was adjusted once, with no reward or physics
changes. Both passes' zero/down results match. Final staged hand-to-gear length
stays within 0.077 mm of the nominal 115 mm; constant-down ranges from 112.828
to 126.446 mm and finishes with 42.333 mm lateral x error. Aggressive continued
commands produce deflection/constraint strain; these traces alone do not isolate
which contacts cause it.

Conclusion: nominal geometric goal reachability and sustained proximity are
shown without a learned policy. There is no API blocker or evidence of an
unavoidable nominal attachment/action failure. Constant-down failure is a
controller/scaling issue in this probe; these runs cannot identify why a learned
policy failed across randomized contexts. The user accepted this diagnostic and directed resumption of method reproduction;
no further seating or randomized scripted-controller audit is planned. This diagnostic does not reproduce the
original GoFlow policy and does not establish randomized-task solvability.

## Historical 64-env single-seed continuation to 5M

The exact resume, milestone-evaluation, calibration and plotting commands, measured
results and stop decision are in [the continuation record](../../docs/gears_single_seed_continuation.md).
That lineage stopped at 5M. The later 1024-env recovery is recorded separately
in [the working baseline](../../docs/gears_baseline.md).
