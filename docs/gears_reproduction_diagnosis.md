# Gears reproduction/configuration diagnosis

This investigation concerns method reproduction under the released return >=50
criterion. It does not audit physical seating or alter that criterion.

## Observed failure mechanism

The 3M and 4M checkpoints each achieved 0/32 successes on uniform held-out
contexts and 0/32 on learned-flow contexts. Uniform mean returns were 8.991 and
8.990; flow means were 9.312 and 9.329. The executed vertical residual was +1
on every evaluated control step. At 4M, its unclipped mean was about 2.686 and
standard deviation 0.337: exploration is overwhelmingly beyond the +1 action
limit. This cancels most default downward guidance and leaves a hovering policy.

This is measured learned-policy saturation/stagnation, not evidence that a
privileged value function or flow failed to load. Both were restored and updated.
The critic estimates low returns, and Equation 7 rejects unsuccessful states.
A nonuniform flow does not imply a usable skill: its return correlation at 4M
was nearly the same as the zero-flow-update reference (uniform episodes:
0.812 learned versus 0.810 initial). No successful skill region was demonstrated.

These observations identify the behavioral failure, not a uniquely proven root
cause. Do not label any configuration discrepancy below as causally established
without a controlled check.

## Concrete reproduction differences, in priority order

1. **PPO batch/parallelism differs substantially from the release.** Gears defaults
   to 1024 environments. The previous launch stalled at initialization for about
   6.5 minutes and was terminated; this was not a diagnosed OOM. The working
   run uses 64. `train_rl.py` sets minibatch size to twice this count, so it is
   128 instead of 2048. With horizon 32 the rollout is 2048 rather than 32768
   transitions. At a fixed transition budget this produces 16 times as many
   PPO update calls, each using smaller minibatches. All eight mini-epochs and
   the 4096-episode flow schedule are retained. This is the clearest concrete
   optimization departure to resolve before another long run. The initialization
   failure preceded the corrected one-grasp-per-environment patch; it does not
   establish that the corrected code cannot initialize at the released count.

2. **Paper and released environment settings differ.** Appendix A.3 specifies
   4 seconds, xy offsets +/-0.05 m and yaw +/-0.393 rad. The release specifies
   2 seconds, xy +/-0.02 m and yaw +/-pi; its yaw-application loop is commented.
   This continuation preserves the release, as requested. A paper-faithful
   configuration needs an explicit reconciliation, especially for episode
   duration and exploration opportunity. Do not silently change these settings.

3. **The privileged critic is a documented reconstruction.** The selected
   released YAML comments it out. We enabled its exact commented [512,256]
   ReLU central-value configuration; actor [256,128,64] ELU remains unchanged.
   Input separation is verified (actor 105, critic 108 with xi suffix).
   Checkpoint inference includes the normalization state. Held-out calibration
   compares values to gamma=.99 discounted returns; JT=50 remains the separate
   precondition threshold. There is no evidence that simply enabling the critic
   fixes the actor's saturated action distribution.

4. **Continuation is not bitwise uninterrupted training.** Upstream checkpoints
   omit the central optimizer, RNG and partial episode/validation batches;
   `env_state` is None. Actor/PPO optimizer, critic weights/normalizers and the
   explicitly persisted flow/flow optimizer continue. The simulator starts a
   fresh seed-0 batch and the central optimizer restarts. Counter restoration
   now preserves the cumulative budget. Future checkpoints should also preserve
   central optimizer/RNG and phase state before claiming exact resumption.

5. **Flow and software fidelity limits remain documented.** GoFlow's released
   objective includes standardized return times log density times density and
   retains its own normalized-coordinate log-probability convention. These are
   preserved rather than replaced with a different objective or density
   normalization. Author package versions/checkpoints are not supplied; our
   Isaac Lab 1.4.1 / Sim 4.2.0.2 / Torch 2.4.0 / RL Games 1.6.1 compatibility
   selection is explicit, not verified as the author's exact runtime.

## Recommended next reproduction step

Resolve the released 1024-environment initialization and resulting PPO batch
configuration first, using a bounded startup/configuration check with the
already corrected attachment code. Reconcile the released 2-second setting
with the paper's 4-second setting before any subsequent long training decision.
Do not respond to the present plateau by merely extending the same saturated
policy, sweeping seeds/methods, changing the reward, or proceeding to connector
claims with an incompetent baseline.

The current continuation and final held-out results are recorded in
`gears_single_seed_continuation.md` and
`results/original_gears/continued_seed0_5m/`. No new training configuration is
introduced by this diagnosis.
