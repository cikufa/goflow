# Explicit privileged-critic reconstruction

The paper's belief precondition requires V(s, ξ). The released Gears default trains a shared actor/value model on actor observations only, despite exposing a separate critic observation from the environment.

`experiments/original_gears/privileged_goflow.yaml` copies the released Gears settings and enables its **exact commented-out central_value_config**: MLP [512,256], ReLU, 4 mini-epochs, learning rate 5e-4, minibatch size 2048, input/value normalization and value clipping. The actor remains [256,128,64] ELU with the same PPO configuration. All flow updates/objectives/hyperparameters remain released code.

This activates an available implementation to satisfy a paper-defined feature. It is an explicit configuration change, not the released default. The comments provide the architecture, but the authors' actual planning checkpoint and configuration remain unavailable. Results must identify which profile was used. The environment's remaining yaw/episode-duration discrepancies are not silently corrected by this profile.

The `--agent_config` argument selects an explicit YAML before the existing configuration overrides and before saving the effective configuration to logs. Its default retains registry behavior. Training instrumentation records privileged observation tensors when enabled. Runtime validation and trained-value dependence on ξ are required before using this model in planning.
