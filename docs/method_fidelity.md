# Method fidelity: initial audit

Status: the one-seed original Gears sanity gate passes after restoring the released 1024-env PPO batch. At 10M, held-out success is 6/100 uniform and 59/100 learned flow; the joint precondition selects 54 flow episodes with 49 successes (90.7% precision). See [baseline and limitations](gears_baseline.md) and [execution record](gears_recovery.md). This is a functional baseline with an explicit privileged-critic configuration, not numerical replication of the paper's full benchmark. The failed 64-env lineage is historical. Native connector geometry and oracle clearance tests now run; grasp macros, INSERT learning and online planning remain pending.

Paper: https://proceedings.mlr.press/v267/curtis25a.html (including appendix and Algorithm 2).
Official code: https://github.com/aidan-curtis/goflow at `a8c6af5de7f427418783fd9faa20d50f38b734a9`.

Artifact classes: A = directly released; B = fully specified reimplementation; C = implementation assumptions required; D = currently unable to reproduce. Classes describe availability, not successful execution.

| Component | Paper | Official code | What we use | Modified? | Reason |
|---|---|---|---|---|---|
| Simulator/robot (A) | IsaacLab, Franka gear insertion | MyPandaEnv in med_gear/direct_panda_position.py; local assets | Native simulator, local assets | Explicit compatibility fixes | Space aliases, runtime batch size, and one constraint per cloned grasp; see compatibility_changes.md |
| PPO (A) | Actor-critic PPO | my_a2c_common.py, my_a2c_continuous.py and RL Games | Released implementation | No | Preserve training behavior |
| Actor observations (A) | Ten poses and finite-difference velocities in Section 5.1 | Policy observation excludes context; history implementation present | Released observations | No | Do not feed hidden geometry to actor |
| Privileged critic (A code, configuration assumption) | V(s, xi) | Environment exposes context; selected YAML disables central value | Initial pilots preserve default; final profile enables exact commented central-value settings | Yes, explicit profile | Input separation/value dependence verified; trained competence not established |
| Flow (A) | Neural spline flow, three transforms, 64 features, eight bins | NormFlowDist: MAF with MonotonicRQSTransform, inverse transform, (64,64), three transforms, eight bins | Exact released construction | No | Preserve direction and normalization conventions |
| Flow hyperparameters (A) | K=100; alpha/beta searched | Gears YAML: alpha=0.5, beta=1.0, num_training_iters=100 | Gears values | No | No manual retuning |
| Domain (A) | x,y in +/-0.05 m; yaw in +/-0.393 rad | x,y +/-0.02 m; yaw +/-pi | Released bounds for baseline | No | Confirmed paper/code discrepancy; runtime behavior pending |
| Checkpoint (D released artifact) | Trained Gears policies evaluated | No checkpoints in tree or GitHub releases; models/ contains geometry | Three local training pilots with recorded provenance | Logging wrapper | No claim of released-checkpoint reproduction; flow and critic weights persisted |
| Belief precondition (B/C) | Equation 7 combines value and density tests in belief expectation | No planner implementation found | Weighted joint-indicator expectation in common/belief_space.py; unit-tested only | Added | Numerical thresholds/representation still need task-specific assumptions |
| BFS (B) | Algorithm 2 uses FIFO frontier, sampled effects, visited states | Not found in released tree | Literal generic algorithm in common/belief_space.py; unit-tested only | Added | No alternative search or handoff objective; no online planning claim |
| Belief/effects (C) | Factored object-pose beliefs and abstract skill effects | Not found | Not implemented | Pending | Need documented representation, equality, effects and sampling assumptions |
| INSPECT (C) | Object-parameterized observation action | Not found | Not implemented | Pending | No special look-before-grasp trigger |
| Bayes3D integration (C, partial) | RGB-D likelihood, coarse-to-fine SMC MAP, grid posterior | No GoFlow integration found; separate library public | Actual Bayes3D renderer plus explicit SMC/grid adapter in isolated process | Added | Synthetic-image checks pass; uniform-color/SMC/grid assumptions documented; no Isaac RGB-D/online inspection claim |
| Taskboard assets (A) | Gear board | taskboard/ and models/ USD/STL files | Released geometry | No | Loaded in visual Gears smoke without private submodule |
| Connector adaptation | Requested diagnostic | Not upstream | Native scene and oracle clearance probe | Added | Original baseline committed first; macro validation/training/planning pending; see experiments/connector_handoff/design.md |

## Potential fidelity issues to retain visibly

Confirmed discrepancy: released Gears uses yaw +/-pi and x/y +/-0.02m, whereas appendix A.3 reports yaw +/-0.393 and x/y +/-0.05m. Start with released configuration; do not present it as numerically identical to the paper.

Confirmed discrepancy: the released GOFLOW.update minimizes a reward term proportional to standardized_return * log_p * exp(log_p), alongside entropy and similarity losses. This is not the literal negative expected-return expression in Algorithm 1. Preserve code for baseline reproduction, report the discrepancy, and do not silently replace it with the paper equation.

`NormFlowDist.log_prob` evaluates normalized coordinates without a physical-coordinate Jacobian correction; sampling rejects values outside bounds. Preserve upstream behavior initially and document density units when selecting epsilon. Do not silently fix the objective or density semantics.

Algorithm 2 samples an effect during planning; it does not authorize reading the real hidden state to predict inspection outcomes. A sampled hypothetical observation and the later real measurement must remain separate.

The README checkpoint example omits `--play`, but train_rl.py uses `train = not args_cli.play`; evaluation requires checking that flag explicitly.

`get_full_state_weights` saves model, optimizer, counters and environment state, but has no explicit GoFlow distribution/optimizer entry. Future checkpoint instrumentation must preserve the learned density as well as the actor and critic; do not assume ordinary PPO checkpoints reconstruct it. Verify the environment-state path before deciding the exact patch.

No success or failure of the hypothesized research gap can be inferred from this setup audit.

## Confirmed active release behavior (clean-clone execution audit)

The Gears YAML comments out central_value_config. A2CBase.has_central_value is therefore false; get_values and get_action_values pass only obs['obs'] to the shared actor/value network. The appended privileged obs['states'] is not used by that active model. The release contains an optional central-value implementation, but the selected Gears configuration does not enable it. A claim of a trained privileged V(s,xi) from this default run would be false. The released-code baseline remains unchanged; any subsequent paper-level reconstruction must explicitly enable and validate the missing privileged critic and record architecture assumptions.

The environment's loop applying roll/pitch/yaw offsets is commented out. yaw_offset is sampled and logged but does not rotate the object. Default episode length is 2 seconds, whereas appendix A.3 says 4 seconds. These differences remain preserved in the released baseline.

Instrumentation logs online rollouts and actual counts, retains flow/optimizer weights in checkpoint metadata, and stops at a PPO rollout/update boundary under an explicit total-transition budget. It delegates PPO and distribution updates unchanged. The upstream frame counter increments only at reporting-batch boundaries, so it is not used as the actual simulator transition count.

## Historical 64-env single-seed 5M continuation outcome

The privileged seed-0 policy was continued to 5,001,216 cumulative control
transitions with its released settings and documented 64-env/central-critic
configuration. Final uniform and flow evaluations both have 0/100 successes.
The actor saturates its vertical residual; the critic predicts low returns and
Equation 7 rejects all evaluated states. No 10M extension or custom task ran.
See [continuation evidence](gears_single_seed_continuation.md) and
[reproduction/configuration diagnosis](gears_reproduction_diagnosis.md). The
user accepted the scripted benchmark diagnostic; further seating audits are
out of scope.
