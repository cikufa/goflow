# GoFlow setup audit

Status: isolated simulator environment installed and infrastructure validated. Original GoFlow reproduction has not run; resolution of pre-existing source-file deletions is pending.

## Provenance

- Official upstream: https://github.com/aidan-curtis/goflow
- Branch: main; commit: `a8c6af5de7f427418783fd9faa20d50f38b734a9`.
- GitHub API reports the same commit, no releases, and approximately 254 MB of tracked file contents.
- Existing local Git history already matches upstream. On arrival nearly all tracked files were deleted from the working tree; `.gitignore` remained. Restoration is pending the user's answer. Do not reset or discard these deletions silently.

## Machine observed 2026-09-14

Ubuntu 22.04.5 LTS, x86_64, glibc 2.35; RTX 4090 (24564 MiB), NVIDIA driver 580.105.08; 62 GiB RAM; 229 GiB available disk. `nvidia-smi` advertises CUDA 13.0 driver capability. `/usr/bin/nvcc` is CUDA 11.5. They are different version indicators. Existing Conda executable: `/home/shekoufeh/miniconda3/bin/conda`. GCC, CMake, Ninja and pdftotext are present.

## Candidate compatibility stack

| Component | Selection | Evidence/status |
|---|---|---|
| Isaac Lab | v1.4.1 | Historical `omni.isaac.lab` namespace matches GoFlow; exact original version not pinned upstream |
| Isaac Sim | 4.2.0.2 | Explicitly prescribed in Lab v1.4.1 pip instructions |
| Python | 3.10 | Explicit requirement of that simulator stack |
| PyTorch | 2.4.0, cu118 | Explicit supported build in those instructions |
| CUDA runtime | 11.8 wheel dependencies | Keep system toolkit and driver unchanged; compiled Bayes3D extensions need separate investigation |
| RL Games, Zuko, NumPy | 1.6.1, 1.3.1, 1.26.4 | Lab pins RL Games; Zuko and NumPy are documented compatibility choices |

Infrastructure checks now pass with this stack. This is not a claim to recover the authors' exact versions or to validate Gears behavior.

Source: https://github.com/isaac-sim/IsaacLab/blob/v1.4.1/docs/source/setup/installation/pip_installation.rst

## Installation boundaries and sequence

1. Resolve existing deletions, retain official remote, create `goflow-handoff-experiment`.
2. Complete source audit and commit baseline preparation.
3. Create a new prefix environment at `.conda/envs/goflow-repro`. Redirect Conda package cache, pip cache, temporary files and simulator cache/config/log locations into this project. Check Conda environment registration behavior before creation, since it can write outside the prefix.
4. Resolve pinned package metadata and total download sizes before downloads. Install inside the prefix only. Export actual `environment.yml` and `requirements_frozen.txt` after installation; do not fabricate a frozen environment now.
5. Install only required Lab/RL Games extensions. Test empty simulator, imports, asset resolution, then Gears. Record exact executed commands in the experiment README.
6. Locate any external checkpoint links; current repository and releases contain none. If no usable checkpoint is available, train the original policy as authorized by the brief. Commit successful reproduction before connector work.

## Permission and external requirements

- No sudo or system driver/toolkit changes currently needed.
- Isaac Sim prompts for NVIDIA Omniverse license acceptance. The user explicitly accepted and authorized the installation during this session.
- Public GitHub and pip sources need no credentials. Removed historical `apa_workcells` submodule used private Autodesk Git; do not attempt access without evidence it is needed and available authorization.
- Isaac Sim extensions and assets can consume many GB. Exact resolved download size remains unmeasured. No download over 20 GB is authorized. No full asset-pack download is planned.
- Bayes3D README proposes Python 3.9, Torch 2.2.0 and JAX 0.4.20 with local CUDA 11 support. This differs from the simulator stack; investigate an isolated perception process before considering a fallback. A README mismatch alone is not proof Bayes3D cannot run.

Bayes3D source: https://github.com/probcomp/bayes3d

## Verified setup progress

- Created `.conda/envs/goflow-repro`, Python 3.10.21; Conda packages downloaded about 40 MB. `CONDA_REGISTER_ENVS=false` prevents modification of the external environment registry.
- Installed Torch 2.4.0+cu118 and torchvision 0.19.0+cu118. GPU allocation/reduction returned 64.0 for 64 ones on RTX 4090; runtime reports CUDA 11.8. No driver/toolkit changes.
- Cloned Lab v1.4.1 into `.deps/IsaacLab`, commit `a520a883ce996d855cc9d5255d71fd1c1307633f`.
- HEAD/byte-range metadata audit: pinned Isaac Sim wheels total 6,013,145,120 bytes; Omniverse Kit 106.1.0.140981 is 63,416,811 bytes. Other small Python dependencies are additional. No full asset-pack download initiated.
- `scripts/project_python.sh` uses bubblewrap to make the host filesystem read-only except the project, while retaining device access. ROS PYTHONPATH and user-site packages are excluded. GPU smoke check passed within this wrapper.
- Historical compatibility selections are recorded in `constraints.txt`; unpinned transitive utility packages remain captured by actual exports. These are not claims about the authors' original environment.
- Installed the three Lab editable extensions and RL Games; `pip check` passes. Lab repository version v1.4.1 corresponds to Python distribution omni-isaac-lab 0.30.7 (these version numbers label different things).
- The final AppLauncher probe completed 10 GPU physics steps with camera-support extensions enabled, then exited with code 0. This verifies infrastructure only; no robot-policy episodes were evaluated.
- The exact upstream flow classes passed CUDA sampling, finite density and gradient checks with seed 12345. This produced zero simulator transitions.
- Full installed exports are in `environment.yml` and `requirements_frozen.txt`; observed versions and probe results are in `environment_validation.json`.
