"""Logging/budget/checkpoint wrapper around the released PPO agent.

All policy and flow updates are delegated unchanged to upstream GoFlow.
"""
import json
import os
from pathlib import Path
import time
import contextlib
import sys

import numpy as np
import torch

from goflow.rl_components.my_a2c_continuous import A2CAgent


class BudgetComplete(Exception):
    pass


class InstrumentedA2CAgent(A2CAgent):
    def __init__(self, *args, **kwargs):
        self.run_dir = Path(os.environ['GOFLOW_RUN_DIR']).resolve()
        (self.run_dir / 'rollouts').mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'checkpoints').mkdir(exist_ok=True)
        self.transitions = self.training_transitions = self.validation_transitions = 0
        self.budget = int(os.environ.get('GOFLOW_TRANSITION_BUDGET', '1000000'))
        self.save_every = int(os.environ.get('GOFLOW_SAVE_EVERY', '25000'))
        self.next_save = self.save_every
        self.started = time.time()
        self.rollout_number = 0
        self.rows = []
        super().__init__(*args, **kwargs)
        self.flow_updates = 0
        original_update = self.dr_method.update
        def logged_update(*a, **kw):
            with (self.run_dir / f'flow_update_{self.flow_updates:04d}.log').open('w') as output:
                class Tee:
                    def write(_, text):
                        sys.__stdout__.write(text)
                        return output.write(text)
                    def flush(_):
                        sys.__stdout__.flush()
                        output.flush()
                with contextlib.redirect_stdout(Tee()):
                    result = original_update(*a, **kw)
            self.flow_updates += 1
            return result
        self.dr_method.update = logged_update

    def env_step(self, actions):
        env = self.vec_env.env.unwrapped
        observation = self.obs['obs'] if isinstance(self.obs, dict) else self.obs
        row = {'xi': env.context.detach().cpu().numpy().copy(),
               'observations': observation.detach().cpu().numpy().copy(),
               'actions': actions.detach().cpu().numpy().copy()}
        row['executed_actions'] = np.clip(row['actions'], -1., 1.)
        result = super().env_step(actions)
        _, rewards, dones, _ = result
        row.update(rewards=rewards.detach().cpu().numpy().copy(),
                   dones=dones.detach().cpu().numpy().copy())
        self.rows.append(row)
        self.transitions += env.num_envs
        if self.validating:
            self.validation_transitions += env.num_envs
        else:
            self.training_transitions += env.num_envs
        return result

    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        state['goflow_distribution'] = self.dr_method.current_dist.flow.state_dict()
        state['goflow_distribution_optimizer'] = self.dr_method.dist_optimizer.state_dict()
        state['instrumentation'] = self.counts()
        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        super().set_full_state_weights(weights, set_epoch=set_epoch)
        if 'goflow_distribution' in weights:
            self.dr_method.current_dist.flow.load_state_dict(weights['goflow_distribution'])
            self.dr_method.dist_optimizer.load_state_dict(weights['goflow_distribution_optimizer'])

    def counts(self):
        return {'transitions': self.transitions, 'training_transitions': self.training_transitions,
                'validation_transitions': self.validation_transitions,
                'wall_seconds': time.time() - self.started, 'started_unix': self.started,
                'parallel_envs': self.num_actors, 'gpu': torch.cuda.get_device_name(),
                'privileged_critic_enabled': self.has_central_value,
                'completed_flow_updates': self.flow_updates}

    def train_epoch(self, validation=False):
        result = super().train_epoch(validation=validation)
        if self.rows:
            arrays = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
            np.savez_compressed(self.run_dir / 'rollouts' / f'{self.rollout_number:06d}.npz', **arrays)
            self.rows.clear()
            self.rollout_number += 1
        metrics = self.counts() | {'validation': validation, 'epoch': self.epoch_num}
        with torch.no_grad():
            lp = self.dr_method.current_dist.log_prob(torch.as_tensor(arrays['xi'][-1], device=self.device))
            metrics['log_p_phi'] = {'min': float(lp.min()), 'max': float(lp.max()), 'mean': float(lp.mean())}
        if not validation:
            for key, values in zip(('actor_loss', 'critic_loss', 'bounds_loss', 'policy_entropy', 'kl'), result[4:9]):
                metrics[key] = float(torch.as_tensor(values).float().mean())
        with (self.run_dir / 'training.jsonl').open('a') as f:
            f.write(json.dumps(metrics) + '\n')
        if self.transitions >= self.next_save:
            self.save(str(self.run_dir / 'checkpoints' / f'transitions_{self.transitions:09d}'))
            self.next_save = (self.transitions // self.save_every + 1) * self.save_every
        if self.transitions >= self.budget:
            raise BudgetComplete
        return result

    def train(self):
        try:
            return super().train()
        except BudgetComplete:
            self.save(str(self.run_dir / 'checkpoints' / 'final'))
            (self.run_dir / 'runtime.json').write_text(json.dumps(self.counts(), indent=2) + '\n')
            return self.last_mean_rewards, self.epoch_num
