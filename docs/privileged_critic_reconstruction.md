# Explicit privileged-critic reconstruction

The paper's belief precondition requires V(s, ξ). The released Gears default trains a shared actor/value model on actor observations only, despite exposing a separate critic observation from the environment.

`experiments/original_gears/privileged_goflow.yaml` copies the released Gears settings and enables its **exact commented-out central_value_config**: MLP [512,256], ReLU, 4 mini-epochs, learning rate 5e-4, minibatch size 2048, input/value normalization and value clipping. The actor remains [256,128,64] ELU with the same PPO configuration. All flow updates/objectives/hyperparameters remain released code.

This activates an available implementation to satisfy a paper-defined feature. It is an explicit configuration change, not the released default. The comments provide the architecture, but the authors' actual planning checkpoint and configuration remain unavailable. Results must identify which profile was used. The environment's remaining yaw/episode-duration discrepancies are not silently corrected by this profile.

The `--agent_config` argument selects an explicit YAML before the existing configuration overrides and before saving the effective configuration to logs. Its default retains registry behavior. Training instrumentation records privileged observation tensors when enabled. Runtime validation and trained-value dependence on ξ are required before using this model in planning.

Runtime validation passed: a 4,096-transition smoke and a 2,000,896-transition pilot completed. Recorded actor width is 105; critic width is 108. The actor observation equals the critic prefix and the three-coordinate suffix equals sampled ξ. Perturbing only ξ at fixed actor observations changes the predicted value. The check is `scripts/check_privileged_critic.py`. This proves access and functional dependence, not successful skill learning or calibration.

The generic `critic_loss` in upstream PPO outputs belongs to the shared model's auxiliary value head when central value is enabled. Actual central-critic losses are separately recorded by RL Games under TensorBoard `losses/cval_loss`; `scripts/export_central_value_losses.py` exports them with their training-transition counts. The final pilot has 507 central-value updates. Plots distinguish these two losses.
