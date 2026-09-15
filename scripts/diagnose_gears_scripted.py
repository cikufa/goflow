"""One nominal episode per scripted residual controller; no policy or training."""
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common.runtime import kit_arguments

OUTPUT = ROOT / 'results/original_gears/scripted_diagnostic'
OUTPUT.mkdir(parents=True, exist_ok=True)
kit = kit_arguments()
from omni.isaac.lab.app import AppLauncher
app = AppLauncher(headless=True, kit_args=kit).app
try:
    import numpy as np
    import torch
    import omni.usd
    from pxr import UsdPhysics
    from goflow.environments.med_gear.direct_panda_position import (
        MyPandaEnv, MyPandaEnvCfg, INITIAL_CFG, IPose, math_utils,
    )

    def array(tensor):
        return tensor.detach().cpu().numpy()[0].copy()

    goal = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).multiply(
        IPose.from_pose(INITIAL_CFG.HOLE_T_PEG_GOAL))

    class Probe(MyPandaEnv):
        def snapshot(self):
            pos = self.scene['peg'].data.root_pos_w - self.scene.env_origins
            quat = self.scene['peg'].data.root_quat_w
            ee_pos, ee_quat = self._compute_frame_pose('robot1', self.robots['robot1'])
            relative = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).invert().multiply(IPose(pos, quat))
            error = IPose.from_pose(INITIAL_CFG.HOLE_T_PEG_GOAL).multiply(relative.invert())
            weights = torch.tensor(INITIAL_CFG.peg_goal_weights, device=self.device)
            reward_distance = torch.linalg.vector_norm(torch.cat(
                (error.pos, math_utils.axis_angle_from_quat(error.quat)), dim=1) * weights, dim=1)
            return dict(gear_pose=array(torch.cat((pos, quat), dim=1)),
                        ee_pose=array(torch.cat((ee_pos, ee_quat), dim=1)),
                        goal_error=array(pos - goal.pos),
                        goal_distance=float(torch.linalg.vector_norm(pos - goal.pos)),
                        reward_distance=float(reward_distance[0]))

        def _get_rewards(self):
            reward = super()._get_rewards()
            # DirectRLEnv auto-resets after rewards: preserve the terminal physical state.
            self.post_physics = self.snapshot()
            return reward

    class Nominal:
        def rsample(self, shape):
            return torch.zeros(*shape, 3)

    cfg = MyPandaEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 10000
    env = Probe(cfg)
    env.set_sampling_dist(Nominal())
    summaries = []
    initial_states = []
    for control in ('zero', 'down', 'staged'):
        env.seed(10000)
        env.pose_history.clear()  # independent episode; same reset pattern as evaluator
        obs, _ = env.reset()
        initial = env.snapshot()
        initial_states.append(initial['gear_pose'])
        assert torch.count_nonzero(env.context) == 0
        attachments = []
        for prim in omni.usd.get_context().get_stage().Traverse():
            if prim.IsA(UsdPhysics.FixedJoint) and 'AssemblerFixedJoint' in str(prim.GetPath()):
                joint = UsdPhysics.FixedJoint(prim)
                attachments.append(dict(path=str(prim.GetPath()),
                    body0=[str(p) for p in joint.GetBody0Rel().GetTargets()],
                    body1=[str(p) for p in joint.GetBody1Rel().GetTargets()],
                    local_pos0=list(joint.GetLocalPos0Attr().Get()),
                    local_pos1=list(joint.GetLocalPos1Attr().Get())))
        assert len(attachments) == 1, attachments
        assert attachments[0]['body0'] == ['/World/envs/env_0/robot1/franka/panda_hand']
        assert len(attachments[0]['body1']) == 1 and attachments[0]['body1'][0].startswith('/World/envs/env_0/peg/')
        records = []
        for step in range(env.max_episode_length):
            before = env.snapshot()
            default = env._compute_default_action('robot1', env.robots['robot1'])
            command = torch.zeros((1, 3), device=env.device)
            stage = control
            if control == 'down':
                command[:, 2] = -1
            elif control == 'staged':
                # Coarse/fine downward IK increments, then cancel vertical default
                # guidance. Positive residual here means braking, not an upward target.
                height = float(before['goal_error'][2])
                descent = min(.10, max(0., 10 * height))
                stage = 'coarse' if height > .01 else ('fine' if height > 0 else 'hold')
                command[:, 2] = (-descent - default[:, 2] * env.default_action_scale[2]) / env.trans_action_scale[2]
            executed = command.clamp(-1, 1)
            delta = default[:, :3] * env.default_action_scale + executed * env.trans_action_scale
            record = dict(step=step, time_s=(step + 1) * env.step_dt,
                          observation=array(obs['policy']), commanded_action=array(command),
                          executed_action=array(executed), default_action=array(default),
                          ik_delta=array(delta), ik_target_position=before['ee_pose'][:3] + array(delta),
                          pre_gear_pose=before['gear_pose'], pre_ee_pose=before['ee_pose'], stage=stage)
            obs, reward, terminated, truncated, _ = env.step(executed)
            record.update(env.post_physics)
            record.update(reward=float(reward[0]), terminated=bool(terminated[0]),
                          truncated=bool(truncated[0]),
                          within_1mm=env.post_physics['goal_distance'] <= .001)
            records.append(record)
            if record['terminated'] or record['truncated']:
                break
        else:
            raise RuntimeError('Released episode did not terminate within max_episode_length')
        path = OUTPUT / f'{control}.npz'
        np.savez_compressed(path, **{key: np.asarray([r[key] for r in records]) for key in records[0]},
                            initial_gear_pose=initial['gear_pose'], initial_ee_pose=initial['ee_pose'],
                            context=array(env.context), goal_pose=array(goal.to_vec()))
        summary = dict(controller=control, seed=10000, steps=len(records),
                       episode_return=sum(r['reward'] for r in records),
                       final_goal_distance_m=records[-1]['goal_distance'],
                       minimum_goal_distance_m=min(r['goal_distance'] for r in records),
                       reaches_goal_within_1mm=any(r['within_1mm'] for r in records),
                       final_within_1mm=records[-1]['within_1mm'],
                       termination_reason='released time limit (sets both terminated and truncated)',
                       trajectory=str(path.relative_to(ROOT)), attachments=attachments,
                       initial_gear_pose=initial['gear_pose'].tolist())
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    report = dict(controllers=summaries, goal_position=array(goal.pos).tolist(),
                  initial_pose_max_difference=float(np.max(np.abs(np.stack(initial_states) - initial_states[0]))),
                  contact_indicator='Unavailable: no configured contact sensor; get_ee_force is an unvalidated joint-force projection, not a contact flag.',
                  insertion_indicator='No native insertion flag. within_1mm measures Euclidean gear-root goal proximity only; it does not certify physical seating.',
                  observation_timing='Raw pre-action policy observation; gear/EE/reward/distances are post-physics BEFORE auto-reset.',
                  action_semantics='executed_action is clipped residual sent to env; ik_delta includes released default guidance. Rotation fixed, residual x/y zero.',
                  staged_controller='100mm downward target increments above 10mm height error, then proportional increments (10 times height error); zero vertical increment below goal. Default z compensated; default x/y retained.',
                  configuration='Released reward, context bounds, 2s episode configuration and action scaling unchanged. One env, nominal context zero, seed10000. History cleared at explicit reset.',
                  physical_seating_confirmed=False)
    (OUTPUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    env.close()
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
app.close()
