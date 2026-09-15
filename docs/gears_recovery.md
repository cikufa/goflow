# Gears recovery in the agreed order

The user authorized continuing the prioritized fixes until one original GoFlow
policy works. Use seed 0, the released success threshold 50, and the existing
project-local environment. No additional physical-seating audit is in scope.

## Step 1: restore released parallelism and PPO batch sizes

The corrected 1024-env probe initialized in about nine seconds and completed.
Intermittent native startup failures recurred later (recorded below), but the
released batch fits and runs. No fewer-env batch approximation is needed.
The released rollout is 1024 x 32 = 32768 transitions, minibatch 2048, eight PPO
mini-epochs. The explicit privileged-critic profile and released two-second
episode, bounds, reward, action scaling and flow update rule are retained.

Bounded probe command (fresh initialization; not the old saturated checkpoint):

```bash
mkdir -p results/original_gears/recovery_1024_seed0
GOFLOW_RUN_DIR=results/original_gears/recovery_1024_seed0 \
GOFLOW_TRANSITION_BUDGET=1000000 GOFLOW_SAVE_EVERY=200000 \
timeout --signal=TERM --kill-after=10s 900s scripts/project_python.sh -u -c \
  'import faulthandler, runpy; faulthandler.dump_traceback_later(120, repeat=True); runpy.run_path("scripts/run_goflow.py", run_name="__main__")' \
  --headless --task Gears-GOFLOW-v0 \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --num_envs 1024 --seed 0 --exp_name recovery_1024_seed0 \
  > results/original_gears/recovery_1024_seed0/run.log 2>&1
```

The probe completed 1,015,808 transitions in 100.63 seconds. Complete-episode
training success was 3.53%, mean return 25.27; uniform validation success 0.72%,
mean return 17.41. This is an early learning signal, not successful reproduction.
It used 589,824 PPO and 425,984 validation transitions, with two flow updates.
Its separate 32-episode learned-density evaluation had mean return 28.72 and
zero successes. The short gate establishes learning progress only.

Seed clarification: the released CLI defaults to seed 0 and overrides the YAML's
42. Seed 0 is therefore not itself a departure from the released CLI default.

## Main run

Restart the same seeded configuration from initialization for an uninterrupted
5M run. Count the probe separately, rather than adding its transitions to the
main policy's budget. The instrumentation now saves the effective configuration,
raw-action statistics, phase return/success and central-critic optimizer/counters.
These additions do not change PPO or flow updates. Full simulator/RNG/phase
restoration is still not claimed for a future resume.

The first main launch stalled before collecting transitions. A native backtrace
showed a wait in PhysX/Carb task scheduling. After terminating that process, a
sequential retry with the same faulthandler entry point as the probe initialized
normally. `startup_stall_run.log` and backtraces preserve the failed attempt.
This is an intermittent startup issue; no cause or general fix is established.
The successful retry used the main command below with `scripts/run_goflow.py`
replaced by the probe's `-c` faulthandler/runpy entry point. No other simulator
process was active at retry startup. No settings/packages/drivers were changed.

At 1,015,808 transitions the main run's actor, privileged critic and flow tensors
are exactly equal to the completed probe checkpoint. This verifies the new
logging/optimizer-persistence additions did not change that initial training.

```bash
mkdir -p results/original_gears/goflow1024_seed0_2s
GOFLOW_RUN_DIR=results/original_gears/goflow1024_seed0_2s \
GOFLOW_TRANSITION_BUDGET=5000000 GOFLOW_SAVE_EVERY=1000000 \
  scripts/project_python.sh -u scripts/run_goflow.py --headless \
  --task Gears-GOFLOW-v0 \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --num_envs 1024 --seed 0 --exp_name goflow1024_seed0_2s \
  > results/original_gears/goflow1024_seed0_2s/run.log 2>&1
scripts/project_python.sh -u scripts/evaluate_gears_milestones.py \
  results/original_gears/goflow1024_seed0_2s \
  > results/original_gears/goflow1024_seed0_2s/evaluation_progress.log 2>&1
```

Continue toward 10M only if the restored configuration is still improving.
If it stalls, the next planned comparison changes only episode duration to the
paper's four seconds, with the difference explicit in both training and evaluation.
Do not change randomization bounds, reward or architecture simultaneously.

## Five-million-transition result and continuation

The fresh main run completed 5,013,504 transitions: 2,555,904 PPO and 2,457,600
uniform validation, with 12 flow updates, in 592.83 seconds of agent wall time.
The probe is separate. Training and evaluation retain stochastic policy actions.

| Milestone (actual transitions) | Uniform successes | Learned-flow successes |
|---|---:|---:|
| 3M (3,014,656) | 1/32 | 14/32 |
| 4M (4,030,464) | 1/32 | 12/32 |
| 5M (5,013,504) | 5/100 | 48/100 |

At 5M the flow mean return is 60.02 versus uniform 20.53. Flow initial-value
versus discounted-return correlation is 0.664, with RMSE 21.80. The explicit
Equation-7 construction accepts 35/100 flow episodes; 30 of those succeed
(85.7% precision, 62.5% recall). Uniform acceptance is zero. The density has
changed, but its return correlation (0.399) is close to the initial density's
(0.395); nonuniformity alone does not establish beneficial learned adaptation.
Detailed outputs: `results/original_gears/goflow1024_seed0_2s/analysis/5000000/`.
This is substantial recovery from the failed 64-env policy, not yet a claim of
paper-level performance or a completed baseline gate.

Continue the same seed to approximately 10M because training is improving.
The first continuation attempt aborted before training with
`malloc(): invalid size (unsorted)`; its log is preserved as
`startup_malloc_crash.log`. A retry with the installed Carbonite scheduler's
process-local thread count capped at 16 initialized and resumed successfully.
This is one successful workaround trial, not proof of the crash's cause or a
general fix. Physics settings and learning parameters are unchanged. The
installed `libcarb.tasking.plugin.so` contains the setting name; see also
[NVIDIA's scheduler setting documentation](https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/tasking/TaskingSettings.html).

```bash
GOFLOW_CPU_THREADS=16 \
GOFLOW_RUN_DIR=results/original_gears/goflow1024_seed0_2s_to10m \
GOFLOW_TRANSITION_BUDGET=10000000 GOFLOW_SAVE_EVERY=2000000 \
scripts/project_python.sh -u -c \
  'import faulthandler, runpy; faulthandler.dump_traceback_later(120, repeat=True); runpy.run_path("scripts/run_goflow.py", run_name="__main__")' \
  --headless --task Gears-GOFLOW-v0 \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --num_envs 1024 --seed 0 --exp_name goflow1024_seed0_2s_to10m \
  --checkpoint results/original_gears/goflow1024_seed0_2s/checkpoints/final.pth \
  > results/original_gears/goflow1024_seed0_2s_to10m/run.log 2>&1
scripts/project_python.sh -u scripts/evaluate_gears_milestones.py \
  results/original_gears/goflow1024_seed0_2s_to10m \
  --milestones 6000000 8000000 10000000 \
  > results/original_gears/goflow1024_seed0_2s_to10m/evaluation_progress.log 2>&1
```

`resume.json` confirms restoration of the prior counts and central optimizer/
counters. Actor, critic, flow and both relevant optimizer states continue;
simulator, RNG and partially collected phase state start fresh. This is a warm
continuation, not bitwise uninterrupted training. Intermediate evaluation uses
32 episodes per distribution, final evaluation 100, fixed seeds 30000/40000.

The live rollout summarizer now reads only archives with a completed metrics
row, avoiding a race against a partially written NPZ. Completed-run accounting
and the five existing unit tests pass.

## Final evaluation extensions

The continuation completed 10,027,008 cumulative transitions (5,111,808 PPO,
4,915,200 validation), 24 flow updates and 156 central-critic update calls.
Cumulative agent wall time was 1,200.64 seconds; the continuation was 607.82
seconds. All checkpoint tensors and recorded PPO losses are finite. The final
complete training phase had 61.1% success over 4096 episodes. These training
episodes do not replace the independent final evaluation.

One offline 8M analysis process failed during SciPy import with an unexpected
`inspect.Signature` comparison TypeError. Its unchanged retry and five fresh
SciPy import probes passed. Both logs are retained (`analysis_8m.log`,
`analysis_8m_retry.log`, `scipy_import_probe.log`); the cause is unestablished.
This occurred before checkpoint analysis and did not interrupt training.

The evaluator can now use a reference sampling flow while holding the actor
fixed; its logged `log_p_phi` still comes from the trained policy's flow.
The analysis checks actor tensor equality, sampling-checkpoint identity and
matching seeds before reporting this comparison. This is a sampling-region
comparison, not an ablation of how the training curriculum affected learning.

```bash
scripts/project_python.sh -u scripts/eval_original_goflow.py \
  --checkpoint results/original_gears/goflow1024_seed0_2s_to10m/checkpoints/final.pth \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --episodes 100 --seed-base 40000 --sampling flow \
  --sampling-flow-checkpoint results/original_gears/recovery_1024_seed0/checkpoints/transitions_000229376.pth \
  --output results/original_gears/goflow1024_seed0_2s_to10m/evaluations/initial_flow_reference
scripts/project_python.sh scripts/analyze_gears_reproduction.py \
  results/original_gears/goflow1024_seed0_2s_to10m --milestone 10000000 \
  --initial-flow-checkpoint results/original_gears/recovery_1024_seed0/checkpoints/transitions_000229376.pth \
  --reference-evaluation results/original_gears/goflow1024_seed0_2s_to10m/evaluations/initial_flow_reference
```

Training plots support `--prior-run` to include both sessions of this lineage;
the independent probe is excluded. Central-value loss axes use restored critic
counters, without adding the prior transition offset twice.

Final outcome: 6/100 uniform and 59/100 learned-flow successes at 10M. The
joint precondition accepts 54 flow episodes, with 49 successes. Full results,
limitations, checkpoint hashes and videos are in [the baseline record](gears_baseline.md).
The baseline rollback commit is `a4e09aa`; the custom geometry diagnostic began
after that commit. No further original-task training or four-second variant ran.
