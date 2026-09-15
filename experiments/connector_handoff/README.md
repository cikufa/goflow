# Active clean clone

All reproduction work and commits now use `/home/shekoufeh/goflow/reproduction` on `goflow-handoff-experiment`. The pre-existing parent checkout and its deletions are preserved. Official upstream is recorded in `docs/upstream.json`. The compatible environment at `/home/shekoufeh/goflow/.conda/envs/goflow-repro` and its existing Isaac Lab dependency are reused. `scripts/project_python.sh` writes only inside this clone and the shared environment; it keeps the rest of the host read-only.

From the clean clone, install GoFlow without reinstalling dependencies:

```bash
scripts/project_python.sh -m pip install --no-deps --no-build-isolation -e .
```

The sections below retain the initial environment installation record. Their environment-creation commands were run in the parent workspace before this clean clone was selected. Simulator/check scripts must now be run from the clean clone.

# Connector handoff reproduction

**Status:** environment preparation is complete. Isaac Lab GPU physics and original GoFlow flow-class compatibility checks pass. Original Gears episodes, baseline training, connector implementation, and scientific evaluation have not run.

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

## Next execution gate

The clean clone and development branch are established; GoFlow is installed in the existing environment. Current GitHub releases/tree contain no trained checkpoints; complete the artifact search, then retrain the original policy if necessary as authorized. Validate and commit the original reproduction before connector work.

Commands for Gears evaluation, connector training, 20-trial evaluation, video generation and reporting will be added when those stages are implemented and validated. No placeholder results or scientific conclusion are claimed.
