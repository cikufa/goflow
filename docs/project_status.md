# Project status: original Gears sanity baseline established

**The complete connector experiment has not been implemented or run. The scientific outcome is inconclusive.**

All work is in `/home/shekoufeh/goflow/reproduction`, branch `goflow-handoff-experiment`, with official remote `upstream` at https://github.com/aidan-curtis/goflow.git. Clean upstream SHA was recorded before edits: `a8c6af5de7f427418783fd9faa20d50f38b734a9`. The parent's 207 pre-existing tracked deletions remain preserved. At the user's request, commits through `10f704b` were pushed to their fork, `origin` at https://github.com/cikufa/goflow. Subsequent connector training commits are local.

## Active connector continuation

The native closed-gripper task and visible-pose grasp macros have passed their
local gates. The custom connector required an exposed insertion tip to clear
closed fingers; the fixture still reverses the feasible grasp choice, including
within the actual two-second episode. Each grasp macro succeeds 50/50 times in
the specified visible-jitter/fixed-constraint validation.

The first seed-0 INSERT run completed 5,013,504 transitions: 0/100 uniform and
18/100 learned-flow successes. Four accepted flow episodes all succeed under
the joint precondition, but the paired nominal fixture cases remain weak.
The continuation completed 10,027,008 transitions, with 2/100 uniform and 33/100
flow successes. The joint precondition accepts 28 flow episodes, 23 successful.
Each nominal feasible grasp/fixture pairing succeeds only 2/10; their initial
values remain below JT=50. Training stopped at the bounded 10M budget. The next
gate is actual macro-to-INSERT state alignment and offset-learning diagnosis,
before more PPO or planner scoring. No algorithm/task changes occurred during
the training continuation. Grasp validation from a shared visible-object
placement again passes 50/50 per choice.
See [configuration, commands and results](../experiments/connector_handoff/training.md).
No executed handoff, online perception or planner outcome is established.

The remaining sections retain the original-task history and pre-training audits.

## Environment

The existing project environment was reused: Python 3.10.21, Isaac Sim 4.2.0.2, Isaac Lab v1.4.1 (`a520a883ce996d855cc9d5255d71fd1c1307633f`), Torch 2.4.0+cu118, RL Games 1.6.1 and Zuko 1.3.1. The RTX 4090 and driver 580.105.08 remain unchanged. System nvcc remains 11.5. Main `pip check` passes.

Bayes3D rendering/inference uses a separate project-local Conda prefix, Torch 2.2.0+cu118, JAX 0.4.20 and a local 11.8 assembler. Its documented rendering subset omits declared GenJAX/Open3D/timm dependencies; full unchanged-package dependency satisfaction is not claimed. Both environments have YAML and frozen-package exports. No sudo, system driver/toolkit changes, external credentials, substantial deletions or download over 20 GB were used. Isaac Sim license acceptance was explicitly authorized by the user.

## Current 1024-env recovery

The released parallelism now runs. The seed-0 lineage with the released
32768-transition rollout and 2048 minibatch completed 10,027,008 total transitions.
Held-out success is 6/100 uniform and 59/100 learned flow. The joint precondition
accepts 54 flow episodes, of which 49 succeed. This passes the minimal functional
baseline gate; see [baseline report](gears_baseline.md) and
[recovery commands and evidence](gears_recovery.md).
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
- Custom scene: native keyed connector/socket/fixture geometry and an oracle clearance probe. Both side cases show the intended reversal with the grasp transform; moving the wall away removes failures. A debug video is saved. See [custom design and measured gate](../experiments/connector_handoff/design.md).
- Not done: original three-gear online planning demonstration, executed connector pickup macros, connector INSERT training, learned skill/precondition validation, 20 balanced planning trials, counterfactual replays and connector planning metrics/videos.

## Evidence and commands

- Generated original report: `results/original_gears/report.md` and `report.json`.
- Actual start/end times and checkpoint SHA-256 values: `results/original_gears/training_runtime.csv`.
- Final videos/contact sheets: `results/original_gears/privileged_independent_flow/{videos,contact_sheets}/`.
- Representative raw frames: `results/original_gears/privileged_independent_flow/raw_frames/episode_000/`.
- Training/density/value plots: `results/original_gears/privileged_training_64/plots/`.
- Synthetic posterior evidence: `results/infrastructure/bayes3d-pose-probe/`.
- Exact commands: `experiments/connector_handoff/README.md`.

All connector-specific success rates, sensing rates, action-sequence distributions and handoff regret are **not available**, since zero connector trials ran. Reporting them as zero would be misleading. The available evidence does not falsify, partially establish or confirm the hypothesized handoff gap. The corrected 1024-env baseline now supports beginning the custom diagnostic.

## Historical 64-env single-seed continuation

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
