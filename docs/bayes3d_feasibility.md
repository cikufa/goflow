# Bayes3D feasibility audit

Inspected official https://github.com/probcomp/bayes3d at
`4e2919dd82c4596b7baca570a15bb7f3a89566a4`, cloned into ignored `.deps/bayes3d`.

The README recommends Python 3.9, Torch 2.2.0/cu118 and JAX 0.4.20 with local CUDA 11 support. Its metadata supports Python >=3.9, but pins `genjax==0.1.1`. On 2026-09-15, a real dependency retrieval attempt failed:

```bash
scripts/project_python.sh -m pip download --no-deps genjax==0.1.1 \
  --dest .cache/bayes3d-wheels --index-url https://pypi.org/simple
```

PyPI has no 0.1.1 artifacts; currently listed stable GenJAX versions require Python >=3.11. Raw output is `.cache/bayes3d-dependency-probe.log`. The Python 3.10 simulator environment was not changed. EGL/GLU headers, GCC and Ninja already exist, so missing system graphics headers are not the demonstrated blocker. The host CUDA compiler is 11.5 whereas Torch uses runtime 11.8; compiled-extension compatibility has not yet been tested.

This demonstrates that the declared installation cannot currently resolve unchanged. It does **not** prove the renderer/likelihood subset cannot run with a source dependency or separate process. GenJAX is imported by Bayes3D's generative-model modules, not directly by its renderer. Such a subset would require an explicit integration audit. No Bayes3D runtime or fallback perception success is claimed at this stage.

No private credentials, sudo, system-driver change or asset-pack download was attempted.

## Isolated renderer integration now passes

Created a second Conda prefix inside the clean clone: `.conda/envs/bayes3d-repro`, Python 3.10.21. The initial Conda payload was 40,323,158 bytes. The simulator environment was not altered. Installed Torch 2.2.0+cu118, torchvision 0.17.0+cu118, JAX 0.4.20, jaxlib 0.4.20+cuda11.cudnn86, NumPy 1.26.4, SciPy 1.11.4 and TensorFlow Probability 0.23.0. Built the unmodified Bayes3D OpenGL/CUDA extension with the existing system compiler and graphics headers.

The first GPU render failed because the system CUDA 11.5 `ptxas` cannot assemble compute capability 8.9 code. JAX's driver fallback returned CUDA_ERROR_INVALID_IMAGE. Installed **only in the perception prefix** `nvidia-cuda-nvcc-cu11==11.8.89` (19.5 MB) and set process-local PATH/XLA_FLAGS to its assembler/libdevice. System CUDA and NVIDIA driver are unchanged.

`scripts/bayes3d_python.sh -u scripts/smoke_bayes3d.py` now passes actual GPU rendering of two box poses: image shape (2,64,64,4), foreground counts 49 and 42, projected horizontal centers 32.0 and 38.5 pixels. Probe runtime 2.13 seconds. Evidence: `.cache/bayes3d-renderer-probe/summary.json` and render.npz.

This is the **Bayes3D rendering subset**, not a claim that every optional module is installed. `pip check` in the separate perception environment reports omitted declared dependencies GenJAX, Open3D and timm. The renderer/import path does not use them. GenJAX 0.1.1 remains unavailable. Do not call this a full unchanged Bayes3D installation; do not substitute a fallback solely because of the packaging issue.

## Paper-level observation adapter (class C)

`experiments/common/bayes3d_pose.py` uses the real Bayes3D renderer with the appendix's pixelwise RGB/depth Laplace mixture, coarse-to-fine tempered SMC MAP search and a grid posterior around the MAP. The renderer returns depth/segmentation rather than shaded RGB, so a known uniform object color supplies predicted RGB for uniformly colored meshes. This is an explicit rendering assumption. SMC counts, proposal scales, tempering, noise scales and grid resolution are unspecified in the paper; the adapter exposes and logs them. The yaw grid retains the entire prior range. The translational grid is centered on the MAP and clipped to the prior bounds, so it may omit posterior tails. Inference accepts images and known calibration/mesh/prior parameters, never the true latent pose.

The synthetic-image test passed for depth scales 0.0025, 0.005 and 0.01 m: finite normalized posteriors, position error below 1.3 mm, and lower position covariance for the closer view at every scale. These tests use 128 SMC particles and a 13×25×25 posterior grid. This is a component check using a synthetic uniformly colored box, not Isaac RGB-D calibration, online inspection, or an original robot-planning reproduction. Evidence is preserved in `results/infrastructure/bayes3d-pose-probe/`. The first test exposed float32 posterior-normalization roundoff against a 1e-8 assertion; weight normalization now uses float64 and the entire six-case check passed.

### Executed install commands

The prefix was created through a subprocess inside `scripts/project_python.sh` with the following effective command (after an identical `--dry-run` and payload-size check):

```bash
CONDA_REGISTER_ENVS=false CONDA_NO_PLUGINS=true PYTHONDONTWRITEBYTECODE=1 \
CONDA_PKGS_DIRS="$PWD/.cache/conda/pkgs" CONDA_ENVS_PATH="$PWD/.conda/envs" \
  /home/shekoufeh/miniconda3/bin/conda create --json --solver classic \
  --override-channels -c conda-forge -p "$PWD/.conda/envs/bayes3d-repro" \
  python=3.10.21 pip=26.2.1 -y
```

Package installation:

```bash
GOFLOW_ENV_PREFIX="$PWD/.conda/envs/bayes3d-repro" scripts/project_python.sh -m pip install \
  torch==2.2.0 torchvision==0.17.0 numpy==1.26.4 \
  --index-url https://download.pytorch.org/whl/cu118
GOFLOW_ENV_PREFIX="$PWD/.conda/envs/bayes3d-repro" scripts/project_python.sh -m pip install \
  'jax[cuda11_local]==0.4.20' scipy==1.11.4 ml-dtypes==0.2.0 numpy==1.26.4 \
  tensorflow-probability==0.23.0 setuptools==69.5.1 setuptools-scm==8.1.0 \
  distinctipy graphviz imageio matplotlib==3.8.4 meshcat natsort \
  opencv-python==4.10.0.84 plyfile pyliblzfse pyransac3d trimesh scikit-learn \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
GOFLOW_ENV_PREFIX="$PWD/.conda/envs/bayes3d-repro" MAX_JOBS=4 \
  scripts/project_python.sh -m pip install --no-deps --no-build-isolation -e .deps/bayes3d
scripts/bayes3d_python.sh -m pip install nvidia-cuda-nvcc-cu11==11.8.89
```

Full installed exports: `environment_bayes3d.yml` and `requirements_bayes3d_frozen.txt`. Interactive activation is `conda activate /home/shekoufeh/goflow/reproduction/.conda/envs/bayes3d-repro`; GPU execution should use `scripts/bayes3d_python.sh` to set the isolated runtime/assembler paths.
