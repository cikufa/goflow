# Method fidelity: initial audit

Status: source inspection only. No reproduction, trained policy, custom task, or experimental results exist yet.

Paper: https://proceedings.mlr.press/v267/curtis25a.html (including appendix and Algorithm 2).
Official code: https://github.com/aidan-curtis/goflow at `a8c6af5de7f427418783fd9faa20d50f38b734a9`.

Artifact classes: A = directly released; B = fully specified reimplementation; C = implementation assumptions required; D = currently unable to reproduce. Classes describe availability, not successful execution.

| Component | Paper | Official code | What we use | Modified? | Reason |
|---|---|---|---|---|---|
| Simulator/robot (A) | IsaacLab, Franka gear insertion | MyPandaEnv in med_gear/direct_panda_position.py; local assets | Original environment first | No | Runtime validation pending |
| PPO (A) | Actor-critic PPO | my_a2c_common.py, my_a2c_continuous.py and RL Games | Released implementation | No | Preserve training behavior |
| Actor observations (A) | Ten poses and finite-difference velocities in Section 5.1 | Policy observation excludes context; history implementation present | Released observations | No | Do not feed hidden geometry to actor |
| Privileged critic (A) | V(s, xi) | Critic observation concatenates policy observation and context | Released critic | No | Inspect full network path before execution |
| Flow (A) | Neural spline flow, three transforms, 64 features, eight bins | NormFlowDist: MAF with MonotonicRQSTransform, inverse transform, (64,64), three transforms, eight bins | Exact released construction | No | Preserve direction and normalization conventions |
| Flow hyperparameters (A) | K=100; alpha/beta searched | Gears YAML: alpha=0.5, beta=1.0, num_training_iters=100 | Gears values | No | No manual retuning |
| Domain (A) | x,y in +/-0.05 m; yaw in +/-0.393 rad | x,y +/-0.02 m; yaw +/-pi | Released bounds for baseline | No | Confirmed paper/code discrepancy; runtime behavior pending |
| Checkpoint (D) | Trained Gears policies evaluated | No checkpoints in tree or GitHub releases; models/ contains geometry | Search remaining links, then retrain original if unavailable | Pending | No claim of released-checkpoint reproduction |
| Belief precondition (B/C) | Equation 7 combines value and density tests in belief expectation | No planner implementation found | Not implemented yet | Pending | Numerical thresholds/representation require explicit audit |
| BFS (B) | Algorithm 2 uses FIFO frontier, sampled effects, visited states | Not found in released tree | Literal algorithm only after baseline | Pending | No alternative search or handoff objective |
| Belief/effects (C) | Factored object-pose beliefs and abstract skill effects | Not found | Not implemented | Pending | Need documented representation, equality, effects and sampling assumptions |
| INSPECT (C) | Object-parameterized observation action | Not found | Not implemented | Pending | No special look-before-grasp trigger |
| Bayes3D integration (C/D) | RGB-D likelihood, coarse-to-fine SMC MAP, grid posterior | No integration found; Bayes3D separately public | Feasibility investigation pending | No | Cannot label an observation model fallback Bayes3D |
| Taskboard assets (A) | Gear board | taskboard/ and models/ USD/STL files | Released geometry | No | USD dependency audit pending |
| Connector adaptation | Requested diagnostic | Not upstream | Not started | Pending | Must first validate and commit original reproduction |

## Potential fidelity issues to retain visibly

Confirmed discrepancy: released Gears uses yaw +/-pi and x/y +/-0.02m, whereas appendix A.3 reports yaw +/-0.393 and x/y +/-0.05m. Start with released configuration; do not present it as numerically identical to the paper.

Confirmed discrepancy: the released GOFLOW.update minimizes a reward term proportional to standardized_return * log_p * exp(log_p), alongside entropy and similarity losses. This is not the literal negative expected-return expression in Algorithm 1. Preserve code for baseline reproduction, report the discrepancy, and do not silently replace it with the paper equation.

`NormFlowDist.log_prob` evaluates normalized coordinates without a physical-coordinate Jacobian correction; sampling rejects values outside bounds. Preserve upstream behavior initially and document density units when selecting epsilon. Do not silently fix the objective or density semantics.

Algorithm 2 samples an effect during planning; it does not authorize reading the real hidden state to predict inspection outcomes. A sampled hypothetical observation and the later real measurement must remain separate.

The README checkpoint example omits `--play`, but train_rl.py uses `train = not args_cli.play`; evaluation requires checking that flag explicitly.

`get_full_state_weights` saves model, optimizer, counters and environment state, but has no explicit GoFlow distribution/optimizer entry. Future checkpoint instrumentation must preserve the learned density as well as the actor and critic; do not assume ordinary PPO checkpoints reconstruct it. Verify the environment-state path before deciding the exact patch.

No success or failure of the hypothesized research gap can be inferred from this setup audit.
