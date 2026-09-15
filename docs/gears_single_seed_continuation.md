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
