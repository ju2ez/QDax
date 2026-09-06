"""Generality metrics: how many DISTINCT controllers cover the task set.

The specialist<->generalist axis is measured by counting distinct controllers
in a MONET repertoire (one genotype per task):

  - specialists: every task has its own controller  -> k_distinct = N
  - generalists: many tasks share one controller    -> k_distinct << N

With the "copy" social-learning operator duplicates are EXACT, so an exact
count is meaningful. With blending operators (sbx, iso_dd) offspring are never
bit-identical, so an exact count always returns N (the singleton trap) — use
an epsilon or behavioural variant instead.

Budget interpretation (count-based): with a fixed per-task network of P
parameters, the initial budget is B0 = N * P and the realised budget after
sharing is B = k_distinct * P; compression = k_distinct / N. NOTE this measures
STORAGE compression at fixed capacity — it does not reallocate the freed
parameters, so it shows the cost of sharing, not generalists outperforming
specialists (that needs resizing each net to B0 / k).
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def _flatten_genotypes(genotypes) -> jnp.ndarray:
    """Pytree of [N, ...] leaves -> a dense [N, D] matrix."""
    leaves = jax.tree.leaves(genotypes)
    n = leaves[0].shape[0]
    return jnp.concatenate([leaf.reshape(n, -1) for leaf in leaves], axis=1)


def count_distinct_exact(genotypes, valid_mask: Optional[jnp.ndarray] = None) -> int:
    """Number of bit-identical controllers. Correct for the "copy" operator."""
    flat = _flatten_genotypes(genotypes)
    if valid_mask is not None:
        flat = flat[valid_mask]
    return int(jnp.unique(flat, axis=0).shape[0])


def count_distinct_eps(
    genotypes, eps: float, valid_mask: Optional[jnp.ndarray] = None
) -> int:
    """Number of controllers that differ by more than `eps` in parameter space.

    Greedy single-pass clustering: a genotype joins the first representative
    within `eps` (normalised euclidean distance), else starts a new one. Use
    for blending operators, where exact duplicates never occur. `eps` is
    scale-dependent — prefer the behavioural variant when fitness is available.
    """
    flat = _flatten_genotypes(genotypes)
    if valid_mask is not None:
        flat = flat[valid_mask]
    reps: list[jnp.ndarray] = []
    for g in flat:
        if not reps:
            reps.append(g)
            continue
        d = jnp.linalg.norm(jnp.stack(reps) - g, axis=1) / jnp.sqrt(g.shape[0])
        if float(d.min()) > eps:
            reps.append(g)
    return len(reps)


def count_distinct_behavioural(
    cross_fitness: jnp.ndarray, delta: float
) -> Tuple[int, jnp.ndarray]:
    """Distinct controllers by BEHAVIOUR, the defensible variant.

    Args:
        cross_fitness: [N, N] matrix; cross_fitness[i, j] = fitness of
            controller j evaluated on task i (diagonal = each task's own
            controller, i.e. the specialist scores).
        delta: relative tolerance. Controllers i and j are "the same" if each
            performs within `delta` of the other's task specialist score.

    Returns:
        (k_distinct, labels) — labels[i] is the representative index for task i.

    This asks "do these controllers do the same job?" rather than "are these
    parameter vectors close?", so it is invariant to parameter-space scale and
    to weight-space symmetries.
    """
    n = cross_fitness.shape[0]
    own = jnp.diag(cross_fitness)
    denom = jnp.maximum(jnp.abs(own), 1e-8)
    # interchangeable[i, j]: controller j is within delta of task i's specialist
    interchangeable = (own[:, None] - cross_fitness) / denom[:, None] <= delta

    labels = -jnp.ones(n, dtype=jnp.int32)
    reps: list[int] = []
    labels_np = [-1] * n
    for i in range(n):
        placed = False
        for r in reps:
            # mutually interchangeable: r works on i's task and i works on r's
            if bool(interchangeable[i, r]) and bool(interchangeable[r, i]):
                labels_np[i] = r
                placed = True
                break
        if not placed:
            reps.append(i)
            labels_np[i] = i
    labels = jnp.asarray(labels_np, dtype=jnp.int32)
    return len(reps), labels


def budget_compression(
    k_distinct: int, num_tasks: int, params_per_controller: int
) -> dict:
    """Count-based budget accounting: initial vs realised controller budget."""
    b0 = num_tasks * params_per_controller
    b = k_distinct * params_per_controller
    return {
        "k_distinct": k_distinct,
        "num_tasks": num_tasks,
        "params_per_controller": params_per_controller,
        "budget_initial": b0,
        "budget_realised": b,
        "compression": b / b0,
        "tasks_per_controller": num_tasks / max(k_distinct, 1),
    }


def mlp_param_count(
    obs_size: int, action_size: int, hidden_sizes: Tuple[int, ...]
) -> int:
    """Parameter count of an MLP policy — the per-task budget P(H)."""
    total = 0
    prev = obs_size
    for h in hidden_sizes:
        total += prev * h + h
        prev = h
    total += prev * action_size + action_size
    return total


def hidden_for_budget(
    budget: int, obs_size: int, action_size: int, n_hidden_layers: int = 2
) -> int:
    """Largest uniform hidden width H whose MLP fits in `budget` parameters.

    Inverse of mlp_param_count — use to size each net at B / k for the
    fixed-total-budget (reallocation) design, where sharing buys capacity.
    """
    h = 1
    while (
        mlp_param_count(obs_size, action_size, (h + 1,) * n_hidden_layers) <= budget
    ):
        h += 1
    return h
