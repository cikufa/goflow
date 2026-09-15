# Gears recovery in the agreed order

The user authorized continuing the prioritized fixes until one original GoFlow
policy works. Use seed 0, the released success threshold 50, and the existing
project-local environment. No additional physical-seating audit is in scope.

## Step 1: restore released parallelism and PPO batch sizes

The corrected 1024-env environment initialized in about nine seconds. The old
startup stall did not recur. Therefore no fewer-env batch approximation is needed.
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

## Main run

Restart the same seeded configuration from initialization for an uninterrupted
5M run. Count the probe separately, rather than adding its transitions to the
main policy's budget. The instrumentation now saves the effective configuration,
raw-action statistics, phase return/success and central-critic optimizer/counters.
These additions do not change PPO or flow updates. Full simulator/RNG/phase
restoration is still not claimed for a future resume.

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
