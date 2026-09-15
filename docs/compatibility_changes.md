# Compatibility changes

The project-local Conda environment, pinned CUDA-enabled PyTorch, Isaac Sim 4.2.0.2 and Isaac Lab v1.4.1 are installed. Infrastructure and Gears visual smoke checks pass. Three baseline training pilots have completed; skill competence remains unestablished. Minimal source patches and their effects are recorded below.

The selected stack and unresolved versions are in `environment_setup_plan.md`. Exact author versions are not specified by upstream. Each future patch must record its triggering failure, minimal diff, validation and potential effect on scientific conclusions.

Existing working-tree deletions predate this work and are not compatibility patches.

## Isaac Sim startup: AWS dependency mismatch

First startup failed to load Replicator: `cannot import name DEFAULT_CHECKSUM_ALGORITHM from botocore.httpchecksum`. Isaac Sim's pip_archive bundles botocore 1.34.68, while its unpinned dependency allowed a 2026 boto3/s3transfer release. Pin boto3=1.34.68, botocore=1.34.68 and s3transfer=0.10.1. This aligns dependency APIs; it changes no GoFlow source or objective. Retest pending.

Retest passed: Replicator imports and the final GPU physics probe completed with exit code 0. The additional screenshot-directory path was redirected with app/captureFrame/path and persistent/app/captureFrame/path settings.

An intermediate probe mixed raw SimulationApp with Lab SimulationContext and disabled fast shutdown; it crashed while unloading plugins from a Lab timeline callback. The final probe uses the published Lab AppLauncher with default shutdown behavior and records physics completion before app.close, which can terminate the process directly. It completed 10 steps and exited with code 0. No simulator library patch was needed.

RL Games source builds required poetry-core 1.9.1 in the environment when using --no-build-isolation. Installing that backend resolved metadata generation. The upstream Lab sources are unchanged.

The filesystem sandbox also prevented writing a stage-template directory under the real Documents folder. Redirect the `app_documents` Kit token into `.cache/kit/app_documents`. OptiX's default /tmp cache was blocked and fell back into the project; the small generated cache files were moved under `.cache/optix`, and OPTIX_CACHE_PATH is now set explicitly.

## Process isolation

The host shell exposes ROS Python paths and Python user-site packages. `scripts/project_python.sh` removes that inherited import path, disables user-site packages, redirects caches, and runs under bubblewrap with host writes disabled outside this project. A CUDA tensor allocation/reduction passed. These changes affect process/package isolation, not GoFlow's algorithm.

## Candidate dependency pins

`constraints.txt` holds Lab-prescribed Torch/RL Games pins and conservative historical choices for formerly unpinned libraries. Zuko 1.3.1 is a compatibility assumption, not a recovered author pin. The original upstream flow classes passed CUDA sampling, log-density and gradient checks. Gears visual smoke passed.

## Clean-clone Gears startup

The first released CLI attempt failed because goflow/logs did not exist; scripts/run_goflow.py creates the intended ignored directory and supplies project-local Kit settings. The second failed Isaac Lab 1.4 validation: action_space and observation_space missing. Added explicit space aliases equal to the released num_actions/num_observations/num_states. Replaced runtime class uses of global NUM_ENVS=1024 with self.num_envs so the documented --num_envs option applies consistently to IK buffers, attachments and rewards. The default 1024-environment behavior is preserved.

A one-iteration Gears run reached PPO but crashed at checkpoint naming: mean_rewards was unbound before the first reporting batch. Initialize that display/checkpoint-name variable to NaN; no optimizer data or reward changes. The 4-env visual smoke subsequently completed 1,024 actual transitions in 12.91 seconds of agent runtime, saved a checkpoint and a 202-frame video, and exited 0.

Opt-in instrumentation is selected by GOFLOW_RUN_DIR; ordinary upstream CLI behavior is otherwise unchanged. It delegates training, records transitions/rollouts/losses, persists the flow, and applies GOFLOW_TRANSITION_BUDGET at a horizon boundary. Logging density checks use existing sampled xi and consume no extra random samples.

## Parallel scene size

The first 1,024-environment headless attempt remained at simulation startup for approximately 6.5 minutes with no transitions. It was terminated (SIGTERM did not stop it; SIGKILL did). This is an observed initialization failure, not proof of an out-of-memory error or an Isaac version incompatibility. The 64-environment attempt initialized in seconds and trained normally. It uses the documented CLI batch-size option. Since the released CLI sets PPO minibatch size to twice the environment count, this also changes minibatch size from 2,048 to 128; the network, optimizer, objective and 4,096-episode flow schedule remain unchanged.

## Conflicting inherited grasp constraints

The live USD probe found seven grasp joints in a four-environment scene: env0 has one, and env1–3 have two each with conflicting local offsets. `create_rigid_attachments` used a unique name even when the cloned env0 joint already existed. Remove the unique-name lookup and author the intended joint at the same path, overriding the inherited constraint's attributes. This is one removed executable line; no reward/PPO/flow modification. It changes actual simulation behavior, so the first pilot is preserved at commit `b0a6ae1` and must not be pooled with corrected training. The fix restores one rigid grasp with each environment's sampled transform; it is not a handoff-specific mechanism.

Retest passed: exactly four grasp joints in four environments, each targeting its own hand/gear pair and each with its own sampled local transform. Evidence: `results/original_gears/attachments_after_fix.json`. Command: `GOFLOW_ATTACHMENT_REPORT=attachments_after_fix.json scripts/project_python.sh -u scripts/check_gears_attachments.py`.

## Independent evaluation episode history

The released `_reset_idx` filters deque entries by environment IDs. Synchronous resets of 64 environments clear all ten history entries, but a one-environment reset removes only entry zero and retains history from the preceding episode. This causes evaluation initial states to differ from training and yielded negative critic predictions after the first episode. Final evaluation clears history before each manual reset, matching the all-env training reset state. The upstream environment implementation is preserved; `--released_history_reset` reproduces its one-env behavior. Earlier results are retained, and independent evaluations use separate output directories. The first independent episode is unaffected by this change.
