"""Algorithm tests only: these are not robot/perception experiment results."""
import unittest
import numpy as np
from experiments.common.belief_space import Skill, bfs, precondition_score


class PlannerTests(unittest.TestCase):
    def test_joint_support_not_product_of_marginals(self):
        r = precondition_score([60, 0], np.log([.01, 1]), [1, 1],
                               return_threshold=50, density_threshold=.1)
        self.assertEqual(r['score'], 0)
        self.assertEqual(r['value_mass'], .5)
        self.assertEqual(r['density_mass'], .5)

    def test_strict_thresholds(self):
        r = precondition_score([50, 51, 51], np.log([1., .1, 1.]), [1, 1, 2],
                               return_threshold=50, density_threshold=.1)
        self.assertEqual(r['score'], .5)
        skill = Skill('equal', lambda b: .5, lambda b, rng: 'goal')
        self.assertEqual(bfs('start', lambda b: b == 'goal', [skill], .5,
                             np.random.default_rng(0)).status, 'failure')

    def test_fifo_shortest_sequence(self):
        edges = [('long1', 'start', 'middle1'), ('short', 'start', 'middle2'),
                 ('long2', 'middle1', 'middle3'), ('finish', 'middle2', 'goal'),
                 ('long3', 'middle3', 'goal')]
        skills = [Skill(name, lambda b, src=src: float(b == src),
                        lambda b, rng, dst=dst: dst) for name, src, dst in edges]
        result = bfs('start', lambda b: b == 'goal', skills, .9, np.random.default_rng(0))
        self.assertEqual(result.plan, ['short', 'finish'])

    def test_visited_stops_cycles_and_initial_goal(self):
        skill = Skill('stay', lambda b: 1., lambda b, rng: b)
        result = bfs('start', lambda b: False, [skill], .9, np.random.default_rng(0))
        self.assertEqual(result.status, 'failure')
        self.assertEqual(result.expanded, 2)  # paper starts visited empty
        self.assertEqual(bfs('goal', lambda b: True, [], .9, np.random.default_rng(0)).plan, [])

    def test_no_precondition_means_all_beliefs(self):
        def forbidden_check(b):
            raise AssertionError('An unconditional skill has no applicability test')
        inspect = Skill('Inspect', forbidden_check, lambda b, rng: 'observed', unconditional=True)
        result = bfs('prior', lambda b: b == 'observed', [inspect], 1., np.random.default_rng(0))
        self.assertEqual(result.plan, ['Inspect'])


if __name__ == '__main__':
    unittest.main()
