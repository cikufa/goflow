# One-seed Gears continuation

User-directed scope: continue one original Gears GoFlow policy to about 5M actual
control transitions, extending toward 10M only if competence is still improving.
Use released return >=50 for success. No further scripted/physical-seat audit.
Proceed to connector only after a competent baseline is committed.

The seed-0 privileged pilot at 2,000,896 transitions is the parent checkpoint:
`results/original_gears/privileged_training_64/checkpoints/final.pth`.
Continue its actor, central critic, PPO optimizer, flow and flow optimizer.
This is one policy lineage, not a new seed or method sweep. Counters now restore
from checkpoint instrumentation; the budget is cumulative and includes validation.
PPO-only and validation transitions are separately recorded. Simulator reset
steps are not counted as control transitions.

Configuration is `experiments/original_gears/privileged_goflow.yaml`: the released
Gears YAML with its exact commented central-value configuration enabled, as
required for the paper's privileged V(s,xi). Preserve released actor, reward,
bounds, duration, action scaling and flow updates. Retain 64 environments from
the pilot (the original 1024-environment launch failed). The released CLI sets
PPO minibatch size to twice the environment count, so this remains 128 rather
than 2048. These departures and the known paper/code discrepancies are explicit
in `method_fidelity.md`; this is not an assertion of identical paper settings.

Resume limitations: upstream does not save the central critic optimizer, RNG,
partial episode/validation batches or a fully restorable simulator state.
Weights and available optimizer states continue, with a seed-0 fresh episode
batch and central optimizer reset. This is a warm continuation, not bitwise
continuation of an uninterrupted run. The wrapper preserves prior transition
counts, completed flow-update count and prior training wall time.

Commands from `/home/shekoufeh/goflow/reproduction`:

```bash
mkdir -p results/original_gears/continued_seed0_5m
GOFLOW_RUN_DIR=results/original_gears/continued_seed0_5m \
GOFLOW_TRANSITION_BUDGET=5000000 GOFLOW_SAVE_EVERY=1000000 \
  scripts/project_python.sh -u scripts/run_goflow.py --headless \
  --task Gears-GOFLOW-v0 \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --num_envs 64 --seed 0 --exp_name continued_seed0_5m \
  --checkpoint results/original_gears/privileged_training_64/checkpoints/final.pth \
  > results/original_gears/continued_seed0_5m/run.log 2>&1
```

During that training, an independent monitor evaluates completed checkpoints at
3M, 4M and 5M. It waits for the next training metrics row or final runtime marker
so it never loads a partially written checkpoint. Evaluation uses the released
stochastic policy on 32 uniform and 32 flow contexts at intermediate checkpoints,
and 100 each at the final checkpoint. Held-out seeds start at 30000/40000;
the same seed sets across checkpoints reduce comparison noise. Flow samples
change as the learned distribution changes. Evaluation transitions are additional
to the training budget and are reported separately.

```bash
scripts/project_python.sh -u scripts/evaluate_gears_milestones.py \
  results/original_gears/continued_seed0_5m \
  > results/original_gears/continued_seed0_5m/evaluation_progress.log 2>&1
scripts/project_python.sh scripts/summarize_training_rollouts.py \
  results/original_gears/continued_seed0_5m
scripts/project_python.sh scripts/analyze_gears_reproduction.py \
  results/original_gears/continued_seed0_5m
scripts/project_python.sh scripts/check_privileged_critic.py \
  results/original_gears/continued_seed0_5m
scripts/project_python.sh -m unittest discover -s tests -v
```

The analysis measures held-out success, density/return association, privileged
value versus observed discounted return, and joint value/density preconditions.
It uses the existing Equation 7 implementation. Density cutoffs are explicit
construction assumptions: lower 1%, 5%, 10% quantiles of 10000 learned-flow
samples, with no evaluation-return tuning. The released density units are
retained. Report precision/recall or undefined when no successes/accepts exist;
never call nonuniform density alone a meaningful skill precondition.

## Executed outcome: stop at 5M, no extension

Final checkpoint: `results/original_gears/continued_seed0_5m/checkpoints/final.pth`
(and matching milestone `transitions_005001216.pth`). Seed 0 only.

- Cumulative transitions: **5,001,216**, comprising **2,578,432 PPO** and
  **2,422,784 uniform validation** transitions.
- This continuation: **3,000,320** additional transitions, including **1,540,096
  PPO** and **1,460,224 validation** transitions.
- Agent wall time: **1,469.205 seconds (24.49 minutes)** for this continuation;
  **2,328.847 seconds (38.81 minutes)** including the parent pilot. Excludes
  simulator startup and held-out evaluation; intermediate evaluations shared the
  GPU, so this is elapsed runtime rather than isolated throughput.
- Twelve cumulative flow updates; seven in this continuation. The actual central
  critic completed 752 additional updates (1,259 across the lineage).
- This continuation recorded 63,808 complete episodes: 32,768 training and
  31,040 validation. **Zero successes** in either phase. Mean returns 9.3200
  (training) and 8.8867 (validation).
- Held-out milestone evaluations used **15,416 additional transitions** across
  328 episodes. These are not included in the 5M training/validation budget.

| Checkpoint | Uniform success | Uniform mean return | Flow success | Flow mean return |
|---|---:|---:|---:|---:|
| 3,000,320 | 0/32 | 8.9906 | 0/32 | 9.3121 |
| 4,001,792 | 0/32 | 8.9898 | 0/32 | 9.3294 |
| 5,001,216 | 0/100 | 8.9395 | 0/100 | 9.3156 |

The first 32 seeds at 5M have means 8.9649/9.3217, so the plateau also holds
on the matched evaluation seed subset. **No 10M extension was started.**

### Four requested checks

1. **Policy competence: failed.** It remains essentially unsuccessful under
   randomized training, uniform evaluation and learned-density evaluation.
   Final raw mean vertical residual is about 2.133 with sigma 0.267. Executed
   vertical actions saturate at +1 on 99.17%/99.30% of uniform/flow steps,
   largely cancelling the released default descent. This is a learned-policy
   saturation/stagnation result, not a new physical benchmark audit.
2. **Meaningful learned density: not established.** Weights changed (L2 distance
   1.785 from the same lineage's zero-flow-update checkpoint), with finite
   nonuniform densities. Uniform density/return correlation is 0.645 learned
   versus 0.670 initial; flow-sampled correlation is 0.560 versus 0.597 initial.
   Association with small low-return differences is not evidence of learned
   successful skill coverage.
3. **Privileged value: functioning, imperfectly calibrated for the weak policy.**
   Input separation and context dependence pass. Initial value means are
   7.824/7.801 versus observed discounted returns 7.161/7.462; RMSE 0.697/0.343
   on uniform/flow contexts. Uniform correlation is weak (0.127), flow correlation
   0.491. It estimates low returns with some overestimation rather than predicting
   successful insertion.
4. **Precondition construction: consistent rejection, no positive skill region.**
   Equation 7 uses the joint value/density indicators. Belief scores are zero
   for both evaluation sets at all three density cutoffs; the fixed-state yaw=0
   slice is also empty. There are no false accepts in these samples, but no
   positive successes with which to establish useful applicability/recall.

The original policy is **not successfully reproduced**. The successful-baseline
condition for starting `GraspLeft/GraspRight -> InsertConnector` is unmet; no
connector training or experiment was started. Configuration diagnosis and the
next bounded reproduction step are in [gears_reproduction_diagnosis.md](gears_reproduction_diagnosis.md).

### Result artifacts

- [Final measured report](../results/original_gears/continued_seed0_5m/analysis/5000000/report.md)
- [Final metrics](../results/original_gears/continued_seed0_5m/analysis/5000000/summary.json)
- [Density/value/precondition plot](../results/original_gears/continued_seed0_5m/analysis/5000000/precondition_slice.png)
- [Training and actual central-critic losses](../results/original_gears/continued_seed0_5m/plots/training.png)
- Per-step NPZ and episode CSVs: `results/original_gears/continued_seed0_5m/evaluations/`.
- Source flow reference: `results/original_gears/privileged_training_64/checkpoints/transitions_000200704.pth`, verified to have zero completed flow updates.

Additional commands executed:

```bash
scripts/project_python.sh scripts/analyze_gears_reproduction.py results/original_gears/continued_seed0_5m --milestone 3000000
scripts/project_python.sh scripts/analyze_gears_reproduction.py results/original_gears/continued_seed0_5m --milestone 4000000
scripts/project_python.sh scripts/export_central_value_losses.py results/original_gears/continued_seed0_5m \
  --tensorboard 'logs/2026-09-15_13-46-23_continued_seed0_5m_Gears-GOFLOW-v0_GOFLOW_0_[]/summaries'
scripts/project_python.sh scripts/plot_original_goflow.py results/original_gears/continued_seed0_5m \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --observation results/original_gears/continued_seed0_5m/evaluations/5000000_uniform/episode_000.npz
```

Central-loss export adds the restored PPO-transition offset because RL Games
restarts its central-critic TensorBoard frame count on resume. Final export spans
1,040,384 through 2,578,432 cumulative PPO transitions. All five existing unit
tests pass; actual checkpoint analysis, critic access and rollout accounting
checks also pass. These checks do not change the failed competence outcome.

Final artifact verification command:

```bash
scripts/project_python.sh scripts/verify_gears_continuation.py results/original_gears/continued_seed0_5m
```

It verifies all 328 evaluation traces, 15,416 held-out transitions, cumulative
budget accounting, finite rewards/values, central-loss indexing and identical
actor/critic/flow tensors in the final and last milestone checkpoints. Separate
serialized archives have distinct byte hashes; both are recorded in
`verification.json`. Final checkpoint SHA-256:
`53dd5db759a9709bd5f979889201b9275d2b264bae401f0560c8d682c71fc365`.
