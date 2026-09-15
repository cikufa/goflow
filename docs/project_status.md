# Project status: baseline gate unmet

**The complete connector experiment has not been implemented or run. The scientific outcome is inconclusive.**

All work is in `/home/shekoufeh/goflow/reproduction`, branch `goflow-handoff-experiment`, with official remote `upstream` at https://github.com/aidan-curtis/goflow.git. Clean upstream SHA was recorded before edits: `a8c6af5de7f427418783fd9faa20d50f38b734a9`. The parent's 207 pre-existing tracked deletions remain preserved. Nothing was pushed.

## Environment

The existing project environment was reused: Python 3.10.21, Isaac Sim 4.2.0.2, Isaac Lab v1.4.1 (`a520a883ce996d855cc9d5255d71fd1c1307633f`), Torch 2.4.0+cu118, RL Games 1.6.1 and Zuko 1.3.1. The RTX 4090 and driver 580.105.08 remain unchanged. System nvcc remains 11.5. Main `pip check` passes.

Bayes3D rendering/inference uses a separate project-local Conda prefix, Torch 2.2.0+cu118, JAX 0.4.20 and a local 11.8 assembler. Its documented rendering subset omits declared GenJAX/Open3D/timm dependencies; full unchanged-package dependency satisfaction is not claimed. Both environments have YAML and frozen-package exports. No sudo, system driver/toolkit changes, external credentials, substantial deletions or download over 20 GB were used. Isaac Sim license acceptance was explicitly authorized by the user.

## Current 1024-env recovery

The released parallelism now runs. A fresh seed-0 run with the released
32768-transition rollout and 2048 minibatch completed 5,013,504 total transitions.
Held-out success is 5/100 uniform and 48/100 learned flow. The same policy is
continuing toward 10M; see [recovery commands and evidence](gears_recovery.md).
The older unsuccessful 64-env results below are preserved as history. No reward,
duration, context bounds or actor architecture was changed for this recovery.
The central critic uses the explicit released-commented configuration. Startup
has intermittently stalled/crashed; a process-local 16-thread Carb cap allowed
the current continuation to initialize, without establishing a general fix.

## Earlier original-task pilots (before the continuation)

| Pilot | Source | Control transitions | PPO transitions | Uniform validation | Agent runtime |
|---|---|---:|---:|---:|---:|
| Initial released-code pilot | `90a9922` | 1,001,472 | 577,536 | 423,936 | 425.823 s |
| Corrected cloned grasp constraints | `19390ef` | 1,001,472 | 577,536 | 423,936 | 424.128 s |
| Explicit privileged-critic profile | `ddc238c` | 2,000,896 | 1,038,336 | 962,560 | 859.647 s |

Total main-pilot transitions: **4,003,840**; measured agent runtime: **1,709.597 seconds**. These totals exclude separate smoke tests, held-out evaluation, startup and internal reset/substep physics. A 1,024-environment startup attempt produced zero transitions and was terminated after about 6.5 minutes; working pilots used 64 environments. Some training overlapped brief GPU infrastructure probes, so these are elapsed execution times rather than isolated throughput measurements.

No public checkpoint was found. The first pilot had conflicting duplicate grasp constraints in cloned environments; it is retained as a rollback record. The minimal constraint-path fix was verified in the live USD stage. The final profile enables the exact central-value configuration in upstream comments and preserves actor architecture/PPO/flow objectives. It completed five GoFlow updates (100 optimizer iterations each) and 507 central-value updates. Recorded critic inputs and value sensitivity to ξ pass checks.

Final **independent** held-out checks match training pose-history initialization:

| Sampling | Episodes | Successes | Mean return | Mean final goal distance |
|---|---:|---:|---:|---:|
| Uniform | 10 | 0 | 9.03764 | 0.05197 m |
| Learned flow | 10 | 0 | 9.39811 | 0.04997 m |

These are Gears checks, **not** the requested 20 balanced connector trials. Released success means episode return >=50. The policy largely cancels default downward motion, moves slightly laterally and hovers near its starting height. It does not insert successfully in these tests. Earlier single-environment evaluations retained the source history-reset bug and remain saved separately.

The plotted privileged value at one fixed recorded initial observation over the yaw=0 x/y slice ranges approximately 7.286–7.540, entirely below 50. This is a slice diagnostic, not a full calibrated belief precondition or proof about all states.

## Released versus added

- Released: native Gears/assets, PPO code, flow, training settings, optional central-value implementation.
- Compatibility/instrumentation additions: explicit Lab spaces, runtime environment-count handling, checkpoint-name initialization, one grasp constraint per clone, independent evaluation history, true transition logs, persisted flow/critic weights and diagnostics.
- Reimplemented: generic paper BFS and joint value/density belief expectation; five meaningful component tests pass. No task-specific online planner demonstration is claimed.
- Partial Bayes3D integration: actual GPU renderer and a documented RGB-D mixture/SMC/grid adapter. Six synthetic-image tests across three noise scales pass; no Isaac RGB-D perception or online inspection integration is claimed.
- Not done: original three-gear online planning demonstration, custom connector scene/macros, connector INSERT training, skill/precondition validation, 20 balanced diagnostic trials, counterfactual replays and connector metrics/videos.

## Evidence and commands

- Generated original report: `results/original_gears/report.md` and `report.json`.
- Actual start/end times and checkpoint SHA-256 values: `results/original_gears/training_runtime.csv`.
- Final videos/contact sheets: `results/original_gears/privileged_independent_flow/{videos,contact_sheets}/`.
- Representative raw frames: `results/original_gears/privileged_independent_flow/raw_frames/episode_000/`.
- Training/density/value plots: `results/original_gears/privileged_training_64/plots/`.
- Synthetic posterior evidence: `results/infrastructure/bayes3d-pose-probe/`.
- Exact commands: `experiments/connector_handoff/README.md`.

All connector-specific success rates, sensing rates, action-sequence distributions and handoff regret are **not available**, since zero connector trials ran. Reporting them as zero would be misleading. The available evidence does not falsify, partially establish or confirm the hypothesized handoff gap. Further original-skill diagnosis or a usable author checkpoint is needed to pass the requested baseline gate.

## Latest single-seed continuation

Seed 0 now has 5,001,216 cumulative transitions: 2,578,432 PPO and 2,422,784 validation.
The continuation added 3,000,320 transitions in 1,469.205 seconds; lineage runtime
is 2,328.847 seconds. Final held-out success is 0/100 uniform and 0/100 learned-flow
episodes, with no material improvement at 3M/4M/5M. Training stopped at 5M; no
10M extension, seed sweep or custom connector experiment ran.

The critic predicts low returns and the joint value/density precondition rejects
the sampled states. A successful skill region has not been reproduced. See
[full continuation result](gears_single_seed_continuation.md) and
[configuration diagnosis](gears_reproduction_diagnosis.md). No further physical
seating audit is planned. The totals above describe the earlier pilots only.
