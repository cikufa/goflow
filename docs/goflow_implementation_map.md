# Released implementation map

Initial inspection used Git objects at `a8c6af5`; all execution uses the fresh `reproduction` clone. The parent checkout's 207 deletions remain preserved. Three training pilots completed; original skill competence has not been established.

| Item | Source and behavior |
|---|---|
| 1. Environment classes | goflow/environments/{ant,anymal,cartpole,humanoid,quadcopter,med_gear}; registration in environments/__init__.py |
| 2. Gears | med_gear/direct_panda_position.py: MyPandaEnvCfg and MyPandaEnv, subclassing IsaacLab DirectRLEnv |
| 3. PPO | rl_components/my_a2c_continuous.py A2CAgent; my_a2c_common.py; RL Games Runner in train_rl.py |
| 4. Actor inputs | _get_observations: current end-effector pose, finite-difference velocity, pose history; optional force sensing by task config |
| 5. Critic inputs | Environment exposes observation concatenated with context, but Gears YAML disables central_value_config. Active shared actor/value network receives only actor observation. See method_fidelity.md |
| 6. xi generation | Environment context randomization/reset; set_sampling_dist receives train/test distribution from agent |
| 7. Flow | my_a2c_common.py NormFlowDist and GOFLOW |
| 8. Spline architecture | Inverse of Zuko MAF transform with MonotonicRQSTransform; 3 transforms; hidden_features=(64,64); 8 bins |
| 9. Entropy | GOFLOW.update uses 10000 uniform samples; volume * mean(exp(log_p) * log_p), minimized with alpha |
| 10. Similarity | Previous distribution cloned before update; 10000 previous-distribution samples estimate old-to-new KL |
| 11. Uniform samples | UniformDist, get_test_dist, test rollout collection; reward weighting in GOFLOW.update. Returns are standardized before weighting |
| 12. Update schedule | Training loop switches sampling distributions and accumulates rollout results before dr_method.update; YAML train_per_update=4096, val_per_update=4096; 100 optimizer iterations per distribution update |
| 13. Bounds | MyPandaEnvCfg.dr_ranges; task configuration in utils.py. Compare against appendix x/y +/-0.05m and yaw +/-0.393 before execution |
| 14. Reward | insert_relative_pose computes weighted pose distance; reward clip(0.01/distance, -10, 10) |
| 15. Termination | _get_dones returns timeout for both terminated and truncated; no separate physical success boolean here. Report reward threshold and physical seating separately |
| 16. Loading | --checkpoint passed to Runner; --play selects evaluation. my_players.py restores model; inspect flow persistence before using checkpoints for preconditions |
| 17. Trained models | None found in complete Git tree or empty GitHub releases response; models/ is geometry |
| 18. Taskboard | taskboard/gear_medium.usd, gear_small.usd, gear_large.usd, taskboard.usd and additional USD/STL meshes; utils.py has gear-specific task factories |
| 19. Multi-step planner | Not found in released source; paper Algorithm 2 provides BFS pseudocode |
| 20. Bayes3D | No integration found; separate probcomp/bayes3d repository |
| 21. Inspection | No released high-level INSPECT implementation found |
| 22. Belief updates | No released probabilistic belief-update implementation found |

## Important inherited configuration

Gears GOFLOW.yaml selects alpha=0.5, beta=1.0, threshold=50; PPO MLP [256,128,64] with ELU, horizon=32, mini_epochs=8, learning_rate=3e-4, gamma=0.99, tau=0.95. max_frames=1000000. train_rl.py overrides minibatch_size to num_envs*2. Record actual transition counts, including uniform validation rollouts, rather than inferring them from epoch names.

Local Gears USD assets load, actor dimensions pass, and the wrapper saves flow weights. Actual simulator transitions are logged separately because upstream's frame counter increments only at reporting boundaries. Successful smoke execution is not evidence of learned competence.

Follow-up inspection confirmed released bounds: context order is yaw_offset, x_offset, y_offset; yaw +/-pi, x/y +/-0.02m. This differs from the appendix. The released reward objective also differs from the literal paper equation; see method_fidelity.md. The distribution-update budgets count collected episodes, not individual transitions.
