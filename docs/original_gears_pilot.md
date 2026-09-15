# Released Gears pilot: rollback record

Source used for training: `90a9922`, upstream `a8c6af5de7f427418783fd9faa20d50f38b734a9`.
No public checkpoint was found; this checkpoint was trained locally.

| Measurement | Result |
|---|---|
| Parallel environments | 64 (1,024-env startup did not finish in ~6.5 minutes) |
| Seed | 0 |
| Actual control transitions | 1,001,472 |
| PPO training transitions | 577,536 |
| Uniform validation transitions | 423,936 |
| Completed GoFlow updates | 2, each 100 optimizer iterations |
| Agent runtime | 425.823 seconds, excluding scene startup |
| Held-out uniform evaluation | 0/10 successes, seeds 10000–10009 |
| Uniform mean return | 8.93731 (released success threshold: 50) |
| Nominal-condition evaluation | 0/3 successes |
| Nominal mean return | 9.39575 |
| Privileged critic | Disabled by released YAML |

Checkpoint: `results/original_gears/released_training_64/checkpoints/final.pth`.
Machine-readable raw rollouts, complete episode returns, PPO losses, rounded upstream flow-loss logs, counts, checkpoint/flow weights and evaluations are retained under `results/original_gears/` (ignored by Git). Control transitions exclude internal physics substeps and reset stabilization steps.

## What the policy did

At nominal geometry the green gear begins 5 cm above its shaft, rigidly attached to the open gripper using the released grasp model. In the recorded trajectory the policy usually commanded a positive vertical residual. The gripper and gear moved slightly sideways and largely hovered. It did not insert the gear. A separately labeled constant-downward-action physics diagnostic moved the gear and obtained return 32.2775, still below 50; this is not a learned-policy success.

Video: `results/original_gears/nominal_visual/videos/episode_000.mp4`.
Contact sheet: `results/original_gears/nominal_visual/contact_sheet.png`.
The first frame in that initial video is black while rendering initializes; subsequent evaluation script revisions warm render buffers without physics steps.

## Confirmed cloned-constraint defect

`scripts/check_gears_attachments.py` inspected the live USD stage after reset. Environment 0 has one grasp joint. Environments 1–3 each have TWO joints between the same hand and gear. One inherits environment 0's local attachment offset; the other has its own sampled offset. The offsets differ, so these constraints conflict.

Evidence: `results/original_gears/attachments_before_fix.json`. For example, env1's inherited local position is `[0.01888965, -0.01922202, 0.115]`, while its new joint's local position is `[0.01163877, 0.01093389, 0.115]`.

Cause: `clone_environments(copy_from_source=False)` inherits env0 edits; `create_rigid_attachments` then chooses a unique name and creates an additional joint instead of overriding the inherited joint. This observation undermines the baseline's simulator validity. It does not prove this defect is the sole cause of training failure.

Required next compatibility correction: define/override the intended joint at the same path in each environment. Verify one joint per environment with the probe and retrain from scratch. Preserve this initial run and commit as the rollback point.

## Scientific status

**Experiment inconclusive.** Original skill competence has not been established. No connector training, online belief-space execution, 20-trial handoff evaluation or counterfactual results exist yet. The generic Algorithm 2/Equation 7 implementation passes unit tests, which are not a robot planning demonstration. The published planner cannot be assessed using this failed pilot.
