"""Paper Algorithm 2 and Equation 7, independent of task/perception assumptions.

States must be hashable beliefs. Callbacks receive beliefs, never simulator state.
The task adapter owns the underspecified effect model, belief equality and thresholds.
"""
from collections import deque
from dataclasses import dataclass
from typing import Callable, Hashable, Any

import numpy as np


def precondition_score(values, log_density, weights, *, return_threshold, density_threshold):
    """Weighted belief expectation of the JOINT value/density indicators (Eq. 7).

    Density units must match the trained flow's log_prob convention. Applicability
    is a separate strict comparison score > eta, as specified by the paper.
    """
    values, log_density, weights = (np.asarray(x, dtype=float) for x in (values, log_density, weights))
    if values.ndim != 1 or values.shape != log_density.shape or values.shape != weights.shape:
        raise ValueError('Expected matching one-dimensional belief-particle arrays')
    if not np.all(np.isfinite(values)) or np.any(np.isnan(log_density)) or np.any(np.isposinf(log_density)):
        raise ValueError('Nonfinite critic/density output')
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Belief weights must be finite, nonnegative and have positive mass')
    if not np.isfinite(density_threshold) or density_threshold <= 0:
        raise ValueError('Density threshold must be positive and finite')
    value_ok = values > return_threshold
    density_ok = log_density > np.log(density_threshold)
    w = weights / weights.sum()
    return {'score': float(w @ (value_ok & density_ok)),
            'value_mass': float(w @ value_ok), 'density_mass': float(w @ density_ok),
            'value_ok': value_ok.tolist(), 'density_ok': density_ok.tolist()}


@dataclass(frozen=True)
class Skill:
    name: str
    precondition: Callable[[Hashable], float]
    sample_effect: Callable[[Hashable, np.random.Generator], Hashable]
    unconditional: bool = False  # e.g. published INSPECT has Pre = all beliefs


@dataclass
class SearchResult:
    status: str
    plan: list[str] | None
    expanded: int
    trace: list[dict[str, Any]]


def bfs(initial, goal, skills, eta, rng, *, max_expansions=10000):
    """Literal FIFO search with one sampled effect per applicable skill expansion.

    A resource limit is an execution safeguard, reported distinctly from Failure.
    Ordered skills determine ties. Initial visited set is empty as in Algorithm 2.
    """
    if not 0 <= eta <= 1:
        raise ValueError('eta must be in [0, 1]')
    frontier, visited, plans = deque([initial]), set(), {initial: []}
    trace, expanded = [], 0
    while frontier:
        b = frontier.popleft()
        if goal(b):
            return SearchResult('success', plans[b], expanded, trace)
        if expanded >= max_expansions:
            return SearchResult('resource_limit', None, expanded, trace)
        expanded += 1
        for skill in skills:
            score = 1.0 if skill.unconditional else float(skill.precondition(b))
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'Invalid applicability probability for {skill.name}')
            applicable = skill.unconditional or score > eta
            event = {'expansion': expanded, 'belief': repr(b), 'skill': skill.name,
                     'score': score, 'applicable': applicable, 'prefix': plans[b].copy()}
            if applicable:
                successor = skill.sample_effect(b, rng)
                event['effect'] = repr(successor)
                event['already_visited'] = successor in visited
                if successor not in visited:
                    frontier.append(successor)
                    visited.add(successor)
                    plans[successor] = plans[b] + [skill.name]
            trace.append(event)
    return SearchResult('failure', None, expanded, trace)
