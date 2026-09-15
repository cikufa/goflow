# One-seed Gears sanity baseline

The zero-success learned-policy failure is resolved by restoring the released
1024-environment PPO batch. This is a working original-task sanity baseline,
with the explicit privileged-critic reconstruction described below. It is
sufficient to begin the requested connector diagnostic. It is not a claim of
numerically reproducing the paper's full benchmark or real-robot results.

## Final independent results

Success retains the released episode-return threshold of 50. Actions remain
stochastic, as in the released player configuration. Each distribution uses
100 independently reset episodes with recorded contexts, observations, actions,
rewards and initial privileged values.

| Evaluation contexts | Successes | Mean return |
|---|---:|---:|
| Uniform released domain, seeds 30000–30099 | 6/100 | 23.14 |
| Learned flow, seeds 40000–40099 | 59/100 | 86.88 |
| Initial flow, identical final actor and seeds 40000–40099 | 53/100 | 83.63 |

The learned-flow checkpoints improved from 48/100 at 5M to 59/100 at 10M.
The old 64-env lineage had 0/100 for both distributions at 5M. This same-seed
recovery identifies the batch reduction as a consequential configuration
difference, without claiming a multi-seed causal study.

### Density

All 24 flow updates used the released objective and settings. Flow parameter
L2 change from the zero-update reference is 2.481. On final flow-sampled
episodes, learned density/return correlation is 0.592 versus 0.562 for the
initial density; learned-density success AUC is 0.852. The initial-flow sampling
comparison shows a modest measured improvement of 6 percentage points and
3.25 return. With 100 episodes this does not establish a statistically reliable
advantage, and is not a curriculum-training ablation. The initial central bias
accounts for much of the useful sampling region.

### Privileged value and skill precondition

Actual actor inputs have 105 elements, critic inputs 108; the latter contain
the actor prefix followed by the three context parameters. Checkpoint inference
and sensitivity to context at a fixed actor observation pass.

For flow episodes, initial V versus observed discounted return has correlation
0.822, RMSE 23.28, and success-ranking AUC 0.953. Mean predicted value is 55.37
versus observed discounted return 61.42. Uniform values are more conservative:
mean 7.37 versus 17.54 observed. This is useful ranking with imperfect numerical
calibration, particularly outside the training region.

Equation 7 is evaluated using JT=50 and a density cutoff equal to the lower
5% quantile of 10,000 learned-flow samples, chosen without evaluating returns.
The release supplies no calibrated epsilon, so this remains an explicit
construction assumption. Sensitivity checks at 1% and 10% are saved too.

- Flow: accepts 54/100 episodes; 49 succeed, 5 fail. Precision **90.7%**, recall
  **83.1%**. This is better than unconditional 59% success and demonstrates a
  useful, nonempty skill region.
- Uniform: accepts 1/100; that episode succeeds. The sample is too small to
  infer reliable uniform-region precision; five successful episodes are rejected.
- The fixed-observation yaw=0 slice has 5.10% joint acceptance. This is a slice,
  not the overall domain coverage or an online planning result.

## What fixed training

The release uses 1024 environments, horizon 32, batch 32768, minibatch 2048 and
eight PPO mini-epochs. The earlier 64-env workaround implicitly made the
minibatch 128 and rollout 2048, producing very different optimization. Returning
to the released sizes recovered learning without modifying the reward, context
bounds, action scaling, episode duration, actor architecture or flow objective.
The corrected one-grasp-per-environment constraints remain enabled.

The released Gears YAML comments out its privileged critic. We use its exact
commented [512,256] central-value settings via
`experiments/original_gears/privileged_goflow.yaml`; this is an explicit
paper-feature reconstruction. Actor [256,128,64] and flow settings are retained.
The release's two-second duration, xy bounds +/-0.02 m and inactive sampled yaw
remain unchanged. The paper lists four seconds and different bounds. Therefore
its published coverage numbers are not directly comparable to this run.
The four-second test is deferred because the first prioritized correction
already produced a usable baseline; no further seed/method sweep is needed
before the custom diagnostic.

## Budget, lineage and runtime

- Upstream: `https://github.com/aidan-curtis/goflow.git`, main commit
  `a8c6af5de7f427418783fd9faa20d50f38b734a9`.
- Branch: `goflow-handoff-experiment`; seed 0 (also the released CLI default).
- Main source: `d5b33071dd403268aae3cae6c3200cbea7a507f7`. The continuation's
  optional startup thread cap is committed in `ab0ca68`; its command and working
  diff at launch are recorded in run provenance.
- Session 1: `results/original_gears/goflow1024_seed0_2s`, 5,013,504 transitions.
- Session 2: `results/original_gears/goflow1024_seed0_2s_to10m`, another 5,013,504.
- Total: **10,027,008** control transitions = **5,111,808 PPO** + **4,915,200
  validation**. There are 24 flow updates and 156 central-value update calls.
- Agent wall time: **1,200.64 s** total (about 20 minutes), excluding startup and
  independent evaluation. Some evaluation shared the GPU during training.
- The separate 1,015,808-transition initialization probe is not added to this
  policy's budget. Its actor/critic/flow tensors exactly match the main run at
  that checkpoint.
- Continuation restores actor, flow, critic and optimizer/counter state, but
  resets simulator/RNG/partial phase state; it is not bitwise uninterrupted.
- Python 3.10.21, Isaac Sim 4.2.0.2, Isaac Lab 1.4.1, Torch 2.4.0+cu118,
  RL Games 1.6.1, Zuko 1.3.1; existing RTX 4090 and driver unchanged.

Intermittent native startup failures and one non-repeating SciPy import error
are preserved in [the recovery record](gears_recovery.md). A process-local
16-thread Carb cap allowed the final continuation to run; no general cause or
system fix is claimed. No driver/package/system changes were made.

## Reproducibility and outputs

Exact training, continuation and evaluation commands:
[gears_recovery.md](gears_recovery.md).

Final checkpoint (ignored, retained locally):
`results/original_gears/goflow1024_seed0_2s_to10m/checkpoints/final.pth`.
SHA256: `edc19113e2a98d04f2078d6d7f5ddd73e36b4edad4e9adbdc61d88dd37721ddb`.
Profile SHA256: `3b44834dd054da3b77a4b3447743dd2eba41496b5dad43d8273a95e6a362dd0b`.

Under the final session directory:

- `evaluations/`: milestone tables and all per-step episode NPZ files.
- `analysis/10000000/report.md`, `summary.json`, `episodes.csv`: density, value,
  initial-flow comparison and joint-precondition measurements.
- `analysis/10000000/precondition_slice.png`: density/value/precondition slices.
- `plots/training.png`: combined training history of both sessions.
- `runtime.json`, `resume.json`, `effective_config.json`, `provenance.json`:
  configuration and actual budget records.
- `critic_access_check.json`, `checkpoint_integrity.log`: executed checks.

The requested `scripts/project_python.sh -m unittest discover -s tests -v`
passes all five tests. Final rollout accounting, saved-weight/loss finiteness,
critic input separation and the executed reference-density analysis pass.

The first video launch stalled before its first recorded control step. Its
native backtrace waits in Carb/GPU-foundation/RTX rendering; it was stopped and
preserved as `visualization/render_stall.log`. A sequential retry applies the
same optional 16-thread process cap through the shared standalone runtime:

```bash
GOFLOW_CPU_THREADS=16 timeout --signal=TERM --kill-after=10s 240s \
scripts/project_python.sh -u scripts/eval_original_goflow.py \
  --checkpoint results/original_gears/goflow1024_seed0_2s_to10m/checkpoints/final.pth \
  --agent_config experiments/original_gears/privileged_goflow.yaml \
  --episodes 3 --seed-base 40000 --sampling flow --video \
  --output results/original_gears/goflow1024_seed0_2s_to10m/visualization
```

This visualization retry is independent of the completed 300-episode final
evaluation and the numerical baseline gate.

The original-task baseline is committed before any custom experiment changes.
No online belief-space planning or connector result is implied by this gate.
