"""Core components of the MONET algorithm."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.containers.monet_repertoire import MONETRepertoire
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class MONETState(flax.struct.PyTreeNode):
    """State of the MONET algorithm.

    Args:
        neighbor_indices: for each task, the indices of its neighbors in the
            task similarity graph, sorted by decreasing similarity, of shape
            (num_tasks, num_neighbors). This is static: the graph is built
            once from the task descriptors at initialization.
    """

    neighbor_indices: jax.Array


def _tree_where(mask: jax.Array, on_true: Genotype, on_false: Genotype) -> Genotype:
    """Select between two pytrees element-wise with a (batch_size,) mask."""
    return jax.tree.map(
        lambda a, b: jnp.where(
            mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim)), a, b
        ),
        on_true,
        on_false,
    )


def _tree_uniform(
    like: Genotype, key: RNGKey, minval: float, maxval: float
) -> Genotype:
    """Sample a pytree of uniform random values with the same shapes as like."""
    leaves, treedef = jax.tree.flatten(like)
    keys = jax.random.split(key, len(leaves))
    new_leaves = [
        jax.random.uniform(k, leaf.shape, minval=minval, maxval=maxval)
        for k, leaf in zip(keys, leaves)
    ]
    return jax.tree.unflatten(treedef, new_leaves)


def _gaussian_mutation(
    x: Genotype, key: RNGKey, mutation_std: float, minval: float, maxval: float
) -> Genotype:
    """Gaussian mutation, with a standard deviation scaled by the range of
    the solution space, as in the reference implementation of MONET."""
    leaves, treedef = jax.tree.flatten(x)
    keys = jax.random.split(key, len(leaves))
    new_leaves = [
        jnp.clip(
            leaf + jax.random.normal(k, leaf.shape) * mutation_std * (maxval - minval),
            minval,
            maxval,
        )
        for k, leaf in zip(keys, leaves)
    ]
    return jax.tree.unflatten(treedef, new_leaves)


def _polynomial_mutation(
    x: Genotype, key: RNGKey, eta: float, minval: float, maxval: float
) -> Genotype:
    """Polynomial mutation (cf Deb 2001, p 124), as implemented in the
    reference implementation of MONET."""
    leaves, treedef = jax.tree.flatten(x)
    keys = jax.random.split(key, len(leaves))

    def _mutate(leaf: jax.Array, k: RNGKey) -> jax.Array:
        r = jax.random.uniform(k, leaf.shape)
        delta = jnp.where(
            r < 0.5,
            jnp.power(2.0 * r, 1.0 / (eta + 1.0)) - 1.0,
            1.0 - jnp.power(2.0 * (1.0 - r), 1.0 / (eta + 1.0)),
        )
        return jnp.clip(leaf + delta, minval, maxval)

    return jax.tree.unflatten(treedef, [_mutate(le, k) for le, k in zip(leaves, keys)])


def _sbx_crossover(
    x: Genotype, y: Genotype, key: RNGKey, eta: float, minval: float, maxval: float
) -> Genotype:
    """Simulated Binary Crossover (cf Deb 2001, p 113), as implemented in
    the reference implementation of MONET."""
    x_leaves, treedef = jax.tree.flatten(x)
    y_leaves = jax.tree.leaves(y)
    keys = jax.random.split(key, len(x_leaves))

    def _cross(x_leaf: jax.Array, y_leaf: jax.Array, k: RNGKey) -> jax.Array:
        k_1, k_2 = jax.random.split(k)
        r_1 = jax.random.uniform(k_1, x_leaf.shape)
        r_2 = jax.random.uniform(k_2, x_leaf.shape)

        x_low = jnp.minimum(x_leaf, y_leaf)
        x_high = jnp.maximum(x_leaf, y_leaf)
        delta = x_high - x_low
        valid = delta > 1e-15
        safe_delta = jnp.where(valid, delta, 1.0)

        def _beta_q(beta: jax.Array) -> jax.Array:
            alpha = 2.0 - jnp.power(beta, -(eta + 1.0))
            return jnp.where(
                r_1 <= 1.0 / alpha,
                jnp.power(r_1 * alpha, 1.0 / (eta + 1.0)),
                jnp.power(1.0 / (2.0 - r_1 * alpha), 1.0 / (eta + 1.0)),
            )

        beta_q_low = _beta_q(1.0 + 2.0 * (x_low - minval) / safe_delta)
        c_1 = 0.5 * (x_low + x_high - beta_q_low * delta)
        beta_q_high = _beta_q(1.0 + 2.0 * (maxval - x_high) / safe_delta)
        c_2 = 0.5 * (x_low + x_high + beta_q_high * delta)

        c_1 = jnp.clip(c_1, minval, maxval)
        c_2 = jnp.clip(c_2, minval, maxval)
        z = jnp.where(r_2 <= 0.5, c_2, c_1)
        return jnp.where(valid, z, x_leaf)

    return jax.tree.unflatten(
        treedef, [_cross(xl, yl, k) for xl, yl, k in zip(x_leaves, y_leaves, keys)]
    )


def _iso_dd_crossover(
    x: Genotype,
    y: Genotype,
    key: RNGKey,
    iso_sigma: float,
    line_sigma: float,
    minval: float,
    maxval: float,
) -> Genotype:
    """Iso+Line variation (Vassiliades and Mouret, GECCO 2018), as
    implemented in the reference implementation of MONET."""
    x_leaves, treedef = jax.tree.flatten(x)
    y_leaves = jax.tree.leaves(y)
    batch_size = x_leaves[0].shape[0]

    key, subkey = jax.random.split(key)
    # one line coefficient per individual, shared across the genotype
    line = jax.random.normal(subkey, (batch_size,)) * line_sigma

    keys = jax.random.split(key, len(x_leaves))
    new_leaves = []
    for x_leaf, y_leaf, k in zip(x_leaves, y_leaves, keys):
        iso = jax.random.normal(k, x_leaf.shape) * iso_sigma
        line_broadcast = line.reshape((batch_size,) + (1,) * (x_leaf.ndim - 1))
        z = x_leaf + iso + line_broadcast * (x_leaf - y_leaf)
        new_leaves.append(jnp.clip(z, minval, maxval))
    return jax.tree.unflatten(treedef, new_leaves)


class MONET:
    """Core elements of the MONET algorithm.

    MONET (Multi-Task Optimization over Networks of Tasks, Hatzky et al.,
    https://arxiv.org/abs/2604.21991) is a graph-based multi-task
    optimization algorithm: the tasks are the nodes of a similarity graph
    and each node stores the best solution found so far for its task (as in
    Multi-task MAP-Elites, the task descriptors play the role of the
    centroids of the repertoire). At every step, a random focal task is
    selected and, with probability p_ind, its elite is mutated (individual
    learning); otherwise, its elite is crossed-over with the elite of one of
    its nearest neighbors in the graph (social learning). The offspring is
    evaluated on the focal task and replaces the elite if its fitness is
    greater than or equal to the fitness of the elite.

    This implementation processes offsprings in batches, mirroring the
    parallel mode of the reference implementation
    (https://github.com/ju2ez/MONET): a batch of candidates is prepared from
    a snapshot of the archive, evaluated in parallel and committed with
    elitism. As in the reference implementation, when the selected neighbor
    has no elite yet, social learning falls back to a Gaussian mutation of
    the focal elite; and when the focal task has no elite yet, the offspring
    is a copy of the neighbor's elite (init_strategy="copy") or a random
    solution.

    Social learning with the "conformity" operator does not cross the focal
    elite with ONE neighbour: it reads the focal task's whole neighbourhood
    and proposes either the locally most common elite (copy conformity,
    conformity_mode="majority") or the focal elite pulled towards the
    proximity-weighted neighbourhood mean (medoid pressure,
    conformity_mode="medoid"); conformity_mode="frequency" is the legacy
    frequency-dependent variant. See ``_conformity_candidate``.

    Note: the "regression" and "plane_dd" operators and the random /
    distance-proportional neighborhood modes of the reference implementation
    are not supported.

    Args:
        scoring_function: a function that takes a batch of genotypes and the
            descriptors of the tasks on which they must be evaluated, and
            computes their fitnesses.
        metrics_function: a function that takes a repertoire and computes
            any useful metric to track its evolution.
        batch_size: number of offsprings generated and evaluated at each
            update.
        num_neighbors: number of nearest neighbors of each task in the task
            similarity graph (the reference implementation uses 1% of the
            number of tasks).
        p_ind: probability of individual learning (mutation); social
            learning (crossover with a neighbor) is applied otherwise.
        neighbor_strategy: how the neighbor used for social learning is
            selected among the neighbors of the focal task, one of
            "best_fitness", "random", "most_similar", "least_similar" or
            "fitness_proportional".
        individual_learning: the mutation operators used for individual
            learning, a tuple with elements among "gaussian_mutation" and
            "polynomial_mutation". If several operators are given, one is
            selected uniformly at random for each offspring.
        social_learning: the crossover operators used for social learning, a
            tuple with elements among "sbx", "iso_dd" and "copy". If several
            operators are given, one is selected uniformly at random for
            each offspring. "copy" adopts the neighbour's elite verbatim
            (no blending) — the hard end of the specialist<->generalist
            dial: accepted copies make a single controller cover several
            tasks, and because the duplicates are exact, the number of
            distinct controllers is a direct measure of generality.
            "conformity" consumes the WHOLE neighbourhood of the focal task
            (its num_neighbors nearest tasks holding an elite) instead of
            the single neighbour picked by ``neighbor_strategy``, and
            produces one candidate according to ``conformity_mode`` (see
            ``_conformity_candidate``). The candidate is evaluated on the
            focal task and accepted with the usual >= rule of the
            repertoire; nothing is enforced unconditionally.
        init_strategy: how the offspring of a focal task without elite is
            initialized: "copy" copies the elite of the selected neighbor
            (when social learning is applied and the neighbor has an elite),
            "random" samples a random solution.
        mutation_std: standard deviation of the Gaussian mutation, relative
            to the range of the solution space.
        sbx_eta: eta parameter of the SBX crossover.
        polynomial_eta: eta parameter of the polynomial mutation.
        iso_sigma: iso parameter of the Iso+Line variation.
        line_sigma: line parameter of the Iso+Line variation.
        top_k_similar: number of candidate neighbors considered by the
            "most_similar" and "least_similar" strategies.
        conformity_mode: what the "conformity" operator does with the
            neighbourhood, one of
              "majority"  — copy conformity: the candidate is the most
                            frequent elite genotype among the occupied
                            neighbours (variants grouped by EXACT
                            equality; ties broken uniformly at random
                            among the tied variants, so an all-distinct
                            neighbourhood copies a uniformly random
                            neighbour);
              "medoid"    — medoid pressure: the candidate is the focal
                            elite pulled linearly towards the
                            proximity-weighted mean of the neighbours'
                            elites (weights 1 / (d_i + 1e-8) over the
                            task-descriptor distances, normalised),
                            candidate = clip(x + lambda * (target - x));
              "frequency" — LEGACY (the previous behaviour, kept so the
                            conf_a{alpha}_m{m} results stay reproducible;
                            NOT the paper's conformity any more):
                            conformity_m neighbours are drawn uniformly
                            with replacement and variant j is adopted
                            verbatim with probability
                            c_j**alpha / sum_k c_k**alpha.
            Whatever the mode, conformity IGNORES ``neighbor_strategy``:
            that strategy selects the ONE neighbour handed to sbx / copy /
            iso_dd, while conformity reads the whole neighbourhood.
        conformity_lambda: pull strength of the "medoid" mode, in (0, 1];
            1 sets the candidate to the weighted mean itself. Unused by the
            other modes.
        conformity_m: number of neighbours sampled by the legacy
            "frequency" mode (>= 1). Unused by the other modes.
        conformity_alpha: exponent of the legacy "frequency" mode (>= 0).
            Unused by the other modes.
        minval: minimum value of the solution space.
        maxval: maximum value of the solution space.
        repertoire_init: a function to initialize the repertoire.
    """

    individual_learning_operators = ("gaussian_mutation", "polynomial_mutation")
    social_learning_operators = ("sbx", "iso_dd", "copy", "conformity")
    conformity_modes = ("majority", "medoid", "frequency")
    neighbor_strategies = (
        "best_fitness",
        "random",
        "most_similar",
        "least_similar",
        "fitness_proportional",
    )

    def __init__(
        self,
        scoring_function: Optional[
            Callable[[Genotype, Descriptor, RNGKey], Tuple[Fitness, ExtraScores]]
        ],
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        batch_size: int,
        num_neighbors: int,
        p_ind: float = 0.5,
        neighbor_strategy: str = "best_fitness",
        individual_learning: Tuple[str, ...] = ("gaussian_mutation",),
        social_learning: Tuple[str, ...] = ("sbx",),
        init_strategy: str = "copy",
        mutation_std: float = 0.05,
        sbx_eta: float = 10.0,
        polynomial_eta: float = 5.0,
        iso_sigma: float = 1.0 / 300.0,
        line_sigma: float = 20.0 / 300.0,
        top_k_similar: int = 3,
        conformity_m: int = 3,
        conformity_alpha: float = 1.0,
        conformity_mode: str = "majority",
        conformity_lambda: float = 0.5,
        minval: float = 0.0,
        maxval: float = 1.0,
        repertoire_init: Callable[
            [Genotype, Fitness, Descriptor, Centroid, Optional[ExtraScores]],
            MONETRepertoire,
        ] = MONETRepertoire.init,
    ) -> None:
        for operator in individual_learning:
            if operator not in self.individual_learning_operators:
                raise ValueError(
                    f"Unknown individual learning operator: {operator}. "
                    f"Supported operators: {self.individual_learning_operators}."
                )
        for operator in social_learning:
            if operator not in self.social_learning_operators:
                raise ValueError(
                    f"Unknown social learning operator: {operator}. "
                    f"Supported operators: {self.social_learning_operators}."
                )
        if neighbor_strategy not in self.neighbor_strategies:
            raise ValueError(
                f"Unknown neighbor selection strategy: {neighbor_strategy}. "
                f"Supported strategies: {self.neighbor_strategies}."
            )
        if init_strategy not in ("copy", "random"):
            raise ValueError(
                f"Unknown init strategy: {init_strategy}. "
                'Supported strategies: ("copy", "random").'
            )
        if conformity_m < 1:
            raise ValueError(f"conformity_m must be >= 1, got {conformity_m}.")
        if conformity_alpha < 0.0:
            raise ValueError(
                f"conformity_alpha must be >= 0, got {conformity_alpha}."
            )
        if conformity_mode not in self.conformity_modes:
            raise ValueError(
                f"Unknown conformity mode: {conformity_mode}. "
                f"Supported modes: {self.conformity_modes}."
            )
        if not (0.0 < float(conformity_lambda) <= 1.0):
            raise ValueError(
                f"conformity_lambda must be in (0, 1], got {conformity_lambda}."
            )

        self._scoring_function = scoring_function
        self._metrics_function = metrics_function
        self._batch_size = batch_size
        self._num_neighbors = num_neighbors
        self._p_ind = p_ind
        self._neighbor_strategy = neighbor_strategy
        self._individual_learning_ops = tuple(individual_learning)
        self._social_learning_ops = tuple(social_learning)
        self._init_strategy = init_strategy
        self._mutation_std = mutation_std
        self._sbx_eta = sbx_eta
        self._polynomial_eta = polynomial_eta
        self._iso_sigma = iso_sigma
        self._line_sigma = line_sigma
        self._top_k_similar = top_k_similar
        self._conformity_m = int(conformity_m)
        self._conformity_alpha = float(conformity_alpha)
        self._conformity_mode = str(conformity_mode)
        self._conformity_lambda = float(conformity_lambda)
        self._minval = minval
        self._maxval = maxval
        self._repertoire_init = repertoire_init

    def init(
        self,
        genotypes: Genotype,
        task_descriptors: Descriptor,
        key: RNGKey,
    ) -> Tuple[MONETRepertoire, MONETState, Metrics]:
        """
        Initialize a MONET repertoire with an initial population of
        genotypes, each evaluated on a randomly selected task, and build the
        task similarity graph.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            task_descriptors: descriptors of the tasks to solve, of shape
                (num_tasks, num_task_descriptors). They are used as the
                centroids of the repertoire.
            key: a random key used for stochastic operations.

        Returns:
            An initialized repertoire, the MONET state (holding the task
            similarity graph) and initial metrics.
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        num_tasks = task_descriptors.shape[0]
        batch_size = jax.tree.leaves(genotypes)[0].shape[0]

        # evaluate each initial genotype on a random task
        key, subkey = jax.random.split(key)
        task_indices = jax.random.randint(subkey, (batch_size,), 0, num_tasks)
        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = self._scoring_function(
            genotypes, task_descriptors[task_indices], subkey
        )

        return self.init_ask_tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            task_indices=task_indices,
            task_descriptors=task_descriptors,
            key=key,
            extra_scores=extra_scores,
        )

    def init_ask_tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        task_indices: jax.Array,
        task_descriptors: Descriptor,
        key: RNGKey,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[MONETRepertoire, MONETState, Metrics]:
        """
        Initialize a MONET repertoire with an initial population of
        genotypes and their evaluations, and build the task similarity
        graph.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitnesses of the initial genotypes, evaluated on the
                tasks given by task_indices
            task_indices: indices of the tasks on which the initial
                genotypes have been evaluated, of shape (batch_size,)
            task_descriptors: descriptors of the tasks to solve, of shape
                (num_tasks, num_task_descriptors)
            key: a random key used for stochastic operations.
            extra_scores: extra scores of the initial genotypes (optional)

        Returns:
            An initialized repertoire, the MONET state (holding the task
            similarity graph) and initial metrics.
        """
        if extra_scores is None:
            extra_scores = {}

        # init the repertoire: the task descriptors are the centroids
        repertoire = self._repertoire_init(
            genotypes,
            fitnesses,
            task_descriptors[task_indices],
            task_descriptors,
            extra_scores,
        )

        # build the task similarity graph: the similarity between two tasks
        # is 1 / (1 + distance), so the most similar neighbors are the
        # nearest neighbors in task descriptor space
        num_tasks = task_descriptors.shape[0]
        # SINGLE-TASK GUARD (added 2026-09-22). With num_tasks == 1 this expression gave 0,
        # so neighbor_candidates had shape (batch, 0) and _select_neighbors crashed in
        # jnp.argmax with "attempt to get argmax of an empty sequence". A one-task MONET is
        # degenerate but legitimate -- it is the k=1 point of a controller-count curve -- and it
        # should reduce to individual learning, not raise.
        # Clamping to 1 makes the sole candidate the task itself (the inf diagonal below means
        # top_k has nothing else to return). That is inert rather than wrong: the social operator
        # then crosses the focal elite with itself, and SBX on identical parents leaves every
        # gene untouched (delta <= 1e-15 fails the `valid` test at _sbx_crossover), so the
        # offspring is the focal elite re-evaluated and accepted on >=, which changes nothing.
        num_neighbors = max(1, min(self._num_neighbors, num_tasks - 1))
        distances = jnp.sum(
            jnp.square(task_descriptors[:, None, :] - task_descriptors[None, :, :]),
            axis=-1,
        )
        # exclude self-loops
        distances = distances + jnp.diag(jnp.full(num_tasks, jnp.inf))
        _, neighbor_indices = jax.lax.top_k(-distances, num_neighbors)
        monet_state = MONETState(neighbor_indices=neighbor_indices)

        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, monet_state, metrics

    def _select_neighbors(
        self,
        repertoire: MONETRepertoire,
        neighbor_candidates: jax.Array,
        key: RNGKey,
    ) -> jax.Array:
        """Select, for each focal task, the neighbor used for social
        learning.

        Args:
            repertoire: the MONET repertoire
            neighbor_candidates: neighbors of each focal task, sorted by
                decreasing similarity, of shape (batch_size, num_neighbors)
            key: a jax PRNG random key

        Returns:
            the indices of the selected neighbor tasks, of shape
            (batch_size,)
        """
        batch_size, num_neighbors = neighbor_candidates.shape
        neighbor_fitnesses = repertoire.fitnesses[neighbor_candidates, 0]

        if self._neighbor_strategy == "best_fitness":
            columns = jnp.argmax(neighbor_fitnesses, axis=1)
        elif self._neighbor_strategy == "random":
            columns = jax.random.randint(key, (batch_size,), 0, num_neighbors)
        elif self._neighbor_strategy == "most_similar":
            top_k = min(self._top_k_similar, num_neighbors)
            columns = jax.random.randint(key, (batch_size,), 0, top_k)
        elif self._neighbor_strategy == "least_similar":
            top_k = min(self._top_k_similar, num_neighbors)
            columns = (
                num_neighbors - 1 - jax.random.randint(key, (batch_size,), 0, top_k)
            )
        else:  # fitness_proportional
            occupied = neighbor_fitnesses > -jnp.inf
            min_fitness = jnp.min(
                jnp.where(occupied, neighbor_fitnesses, jnp.inf),
                axis=1,
                keepdims=True,
            )
            weights = jnp.where(occupied, neighbor_fitnesses - min_fitness + 1e-6, 0.0)
            # fall back to a uniform choice when no neighbor has an elite
            logits = jnp.where(
                weights > 0.0, jnp.log(jnp.maximum(weights, 1e-30)), -jnp.inf
            )
            logits = jnp.where(
                jnp.sum(weights, axis=1, keepdims=True) > 0.0,
                logits,
                jnp.zeros_like(logits),
            )
            columns = jax.random.categorical(key, logits, axis=1)

        return neighbor_candidates[jnp.arange(batch_size), columns]

    def _conformity_neighbors(
        self,
        repertoire: MONETRepertoire,
        neighbor_candidates: jax.Array,
        key: RNGKey,
    ) -> Tuple[jax.Array, jax.Array]:
        """LEGACY conformity_mode="frequency": frequency-dependent
        (conformist) transmission after Boyd & Richerson. Kept verbatim so
        the existing conf_a{alpha}_m{m} results stay reproducible; it is NOT
        the paper's conformity any more (see ``_conformity_candidate`` for
        the "majority" and "medoid" modes).

        For each focal task, ``conformity_m`` neighbours are sampled UNIFORMLY
        WITH REPLACEMENT from the task's neighbourhood (restricted to
        neighbours that hold an elite), their elite genotypes are grouped into
        variants by exact equality, and variant j is adopted verbatim with

            P(j) = c_j ** alpha / sum_k c_k ** alpha,

        where c_j is the number of the m samples carrying variant j.

        Implementation: sample i is drawn with weight c(i) ** (alpha - 1),
        where c(i) counts the samples equal to sample i. Since variant j
        contributes c_j such samples, the induced variant law is exactly
        c_j ** alpha / sum_k c_k ** alpha.

        Reductions: alpha = 1 gives unbiased copying of one uniformly sampled
        neighbour; m = 1 gives copying of one uniformly sampled neighbour for
        any alpha; alpha = 0 gives a uniform draw over the distinct variants.

        NOTE: this operator does NOT use ``neighbor_strategy``. Frequency-
        dependent transmission is defined over a uniform sample of the
        neighbourhood, so the m samples are always drawn uniformly over the
        occupied neighbours; ``neighbor_strategy`` still governs the single
        neighbour handed to the other social operators. Consequence for
        experiments: "conformity + best_fitness" is NOT a distinct condition -
        it is identical to "conformity + random", so it cannot serve as a
        control that separates the operator from the neighbour rule. Use
        sbx/copy with ``neighbor_strategy`` in {random, best_fitness} for that.

        Memory: the variant counting compares m x m genotype pairs per focal
        task, i.e. O(batch_size * m^2 * genotype_size) during the ask step.

        Args:
            repertoire: the MONET repertoire
            neighbor_candidates: neighbors of each focal task, of shape
                (batch_size, num_neighbors)
            key: a jax PRNG random key

        Returns:
            the indices of the adopted neighbor tasks, of shape (batch_size,),
            and a boolean mask, of shape (batch_size,), that is True where the
            focal task had at least one neighbor holding an elite.
        """
        batch_size, num_neighbors = neighbor_candidates.shape
        m = self._conformity_m
        occupied = repertoire.fitnesses[neighbor_candidates, 0] > -jnp.inf
        any_occupied = jnp.any(occupied, axis=1)

        # uniform over the occupied neighbours; if a focal task has none, fall
        # back to a uniform draw over all of them (the caller then replaces the
        # offspring exactly as it does for an unoccupied single neighbour)
        logits = jnp.where(occupied, 0.0, -jnp.inf)
        logits = jnp.where(any_occupied[:, None], logits, jnp.zeros_like(logits))
        key, subkey = jax.random.split(key)
        columns = jax.random.categorical(subkey, logits, axis=-1, shape=(m, batch_size))
        columns = jnp.transpose(columns)                       # (batch_size, m)
        sampled = jnp.take_along_axis(neighbor_candidates, columns, axis=1)

        # count, for every sample, how many of the m samples carry the same
        # genotype (exact equality over every leaf of the pytree)
        equal = jnp.ones((batch_size, m, m), dtype=bool)
        for leaf in jax.tree.leaves(repertoire.genotypes):
            values = leaf[sampled]                             # (batch_size, m, ...)
            values = values.reshape(batch_size, m, -1)
            equal = equal & jnp.all(
                values[:, :, None, :] == values[:, None, :, :], axis=-1
            )
        counts = jnp.sum(equal, axis=2).astype(jnp.float32)     # (batch_size, m)

        key, subkey = jax.random.split(key)
        log_weights = (self._conformity_alpha - 1.0) * jnp.log(counts)
        picked = jax.random.categorical(subkey, log_weights, axis=-1)
        adopted = sampled[jnp.arange(batch_size), picked]
        return adopted, any_occupied

    def _conformity_candidate(
        self,
        repertoire: MONETRepertoire,
        x: Genotype,
        task_indices: jax.Array,
        neighbor_candidates: jax.Array,
        key: RNGKey,
    ) -> Tuple[Genotype, Genotype, jax.Array]:
        """The candidate produced by the "conformity" social operator.

        Let N be the focal task's neighbourhood (``neighbor_candidates``, its
        num_neighbors nearest tasks in the similarity graph; the focal itself
        is never in it) restricted to the neighbours holding an elite. The
        candidate depends on ``conformity_mode``:

        "majority" (copy conformity): the elites of N are grouped into
            variants by EXACT genotype equality (every leaf, every
            component) and the candidate is the most frequent variant. Ties
            are broken uniformly at random among the tied variants: since
            tied variants have the same count, a uniform draw over the
            neighbours whose count equals the maximum is uniform over the
            tied variants. An all-distinct neighbourhood therefore copies a
            uniformly random neighbour.
        "medoid" (medoid pressure): with d_i the Euclidean distance between
            the focal task descriptor and neighbour i's task descriptor (the
            repertoire centroids), w_i = (1 / (d_i + 1e-8)) / sum_j 1 / (d_j
            + 1e-8) over N, target = sum_i w_i * genotype_i and the candidate
            is clip(x + lambda * (target - x), minval, maxval), lambda in
            (0, 1]; lambda = 1 sets the candidate to the target exactly.
        "frequency" (LEGACY): the previous operator, ``_conformity_neighbors``
            — m neighbours drawn uniformly with replacement from N, variant j
            adopted verbatim with probability c_j**alpha / sum_k c_k**alpha.

        In every mode the candidate is evaluated on the focal task and
        accepted with the repertoire's >= rule; nothing is enforced. If N is
        empty the returned occupancy flag is False and the caller replaces
        the candidate exactly as it does for an unoccupied single neighbour
        (Gaussian mutation of the focal elite).

        NOTE: conformity does NOT use ``neighbor_strategy``. That strategy
        selects the ONE neighbour handed to sbx / copy / iso_dd; conformity
        consumes the whole neighbourhood, so "conformity + best_fitness" and
        "conformity + random" are the same condition.

        Memory ("majority"): the variant counting compares K x K genotype
        pairs per focal task, i.e. O(batch_size * K^2 * genotype_size); when
        that exceeds ~2^26 elements the comparison is scanned one genotype
        component at a time, which bounds it at O(batch_size * K^2).

        Args:
            repertoire: the MONET repertoire (genotypes, fitnesses, and
                centroids = task descriptors are read)
            x: the focal elites, pytree with leaves (batch_size, ...)
            task_indices: the focal tasks, of shape (batch_size,)
            neighbor_candidates: neighbors of each focal task, of shape
                (batch_size, num_neighbors)
            key: a jax PRNG random key

        Returns:
            the candidate genotype (pytree like ``x``), the neighbourhood
            representative it was built from (the majority variant, the
            weighted mean target, or the legacy adopted variant; used to
            initialise an empty focal cell under init_strategy="copy") and a
            boolean mask, of shape (batch_size,), that is True where the
            focal task had at least one neighbour holding an elite.
        """
        batch_size, num_neighbors = neighbor_candidates.shape
        mode = self._conformity_mode

        if mode == "frequency":
            adopted, any_occupied = self._conformity_neighbors(
                repertoire, neighbor_candidates, key
            )
            y = jax.tree.map(lambda g: g[adopted], repertoire.genotypes)
            return y, y, any_occupied

        occupied = repertoire.fitnesses[neighbor_candidates, 0] > -jnp.inf
        any_occupied = jnp.any(occupied, axis=1)
        # elites of the neighbourhood, (batch_size, num_neighbors, ...)
        neighbor_genotypes = jax.tree.map(
            lambda g: g[neighbor_candidates], repertoire.genotypes
        )

        if mode == "majority":
            # flatten every leaf to (batch_size, num_neighbors, features)
            values = jnp.concatenate(
                [
                    leaf.reshape(batch_size, num_neighbors, -1)
                    for leaf in jax.tree.leaves(neighbor_genotypes)
                ],
                axis=-1,
            )
            num_features = values.shape[-1]
            if batch_size * num_neighbors * num_neighbors * num_features <= 2**26:
                equal = jnp.all(values[:, :, None, :] == values[:, None, :, :], axis=-1)
            else:

                def _scan_feature(carry: jax.Array, v: jax.Array) -> Tuple[jax.Array, None]:
                    return carry & (v[:, :, None] == v[:, None, :]), None

                equal, _ = jax.lax.scan(
                    _scan_feature,
                    jnp.ones((batch_size, num_neighbors, num_neighbors), dtype=bool),
                    jnp.moveaxis(values, -1, 0),
                )
            # count of each neighbour's variant over the OCCUPIED neighbours
            counts = jnp.sum(equal & occupied[:, None, :], axis=2)
            counts = jnp.where(occupied, counts, 0)
            is_max = occupied & (counts == jnp.max(counts, axis=1, keepdims=True))
            # uniform over the neighbours carrying a maximal-count variant ==
            # uniform over the tied variants; an empty neighbourhood gets a
            # uniform draw over all columns (replaced by the caller anyway)
            logits = jnp.where(is_max, 0.0, -jnp.inf)
            logits = jnp.where(any_occupied[:, None], logits, jnp.zeros_like(logits))
            columns = jax.random.categorical(key, logits, axis=-1)
            y = jax.tree.map(
                lambda g: g[jnp.arange(batch_size), columns], neighbor_genotypes
            )
            return y, y, any_occupied

        # mode == "medoid": proximity-weighted mean of the occupied neighbours
        focal_descriptors = repertoire.centroids[task_indices]              # (B, D)
        neighbor_descriptors = repertoire.centroids[neighbor_candidates]    # (B, K, D)
        distances = jnp.sqrt(
            jnp.sum(
                jnp.square(neighbor_descriptors - focal_descriptors[:, None, :]),
                axis=-1,
            )
        )
        weights = jnp.where(occupied, 1.0 / (distances + 1e-8), 0.0)
        total = jnp.sum(weights, axis=1, keepdims=True)
        weights = weights / jnp.where(total > 0.0, total, 1.0)              # (B, K)

        def _weighted_mean(leaf: jax.Array) -> jax.Array:
            w = weights.reshape(weights.shape + (1,) * (leaf.ndim - 2))
            return jnp.sum(w * leaf, axis=1)

        target = jax.tree.map(_weighted_mean, neighbor_genotypes)
        lam = self._conformity_lambda
        if lam >= 1.0:
            candidate = jax.tree.map(
                lambda t: jnp.clip(t, self._minval, self._maxval), target
            )
        else:
            candidate = jax.tree.map(
                lambda xl, t: jnp.clip(xl + lam * (t - xl), self._minval, self._maxval),
                x,
                target,
            )
        return candidate, target, any_occupied

    def _apply_operators(
        self,
        operators: Tuple[str, ...],
        offsprings: Tuple[Genotype, ...],
        key: RNGKey,
        batch_size: int,
    ) -> Genotype:
        """Select, for each offspring, the result of one of the operators,
        uniformly at random."""
        if len(operators) == 1:
            return offsprings[0]
        operator_indices = jax.random.randint(key, (batch_size,), 0, len(operators))
        result = offsprings[0]
        for i in range(1, len(operators)):
            result = _tree_where(operator_indices == i, offsprings[i], result)
        return result

    def _individual_learning(
        self, x: Genotype, key: RNGKey, batch_size: int
    ) -> Genotype:
        """Apply individual learning (mutation) to the focal elites."""
        offsprings = []
        for operator in self._individual_learning_ops:
            key, subkey = jax.random.split(key)
            if operator == "gaussian_mutation":
                offsprings.append(
                    _gaussian_mutation(
                        x, subkey, self._mutation_std, self._minval, self._maxval
                    )
                )
            else:  # polynomial_mutation
                offsprings.append(
                    _polynomial_mutation(
                        x, subkey, self._polynomial_eta, self._minval, self._maxval
                    )
                )
        key, subkey = jax.random.split(key)
        return self._apply_operators(
            self._individual_learning_ops, tuple(offsprings), subkey, batch_size
        )

    def _social_learning(
        self,
        x: Genotype,
        y: Genotype,
        key: RNGKey,
        batch_size: int,
        y_conformity: Optional[Genotype] = None,
    ) -> Genotype:
        """Apply social learning (crossover with a neighbor elite) to the
        focal elites. ``y_conformity`` is the candidate produced by the
        conformity operator (see ``_conformity_candidate``); it is only read
        when "conformity" is among the social learning operators."""
        offsprings = []
        for operator in self._social_learning_ops:
            key, subkey = jax.random.split(key)
            if operator == "sbx":
                offsprings.append(
                    _sbx_crossover(
                        x, y, subkey, self._sbx_eta, self._minval, self._maxval
                    )
                )
            elif operator == "conformity":
                # The offspring IS the conformity candidate: the local
                # majority variant ("majority"), the focal elite pulled
                # towards the proximity-weighted neighbourhood mean
                # ("medoid") or the legacy frequency-biased variant
                # ("frequency"). Built in ask() by _conformity_candidate.
                offsprings.append(jax.tree.map(lambda leaf: leaf, y_conformity))
            elif operator == "copy":
                # Verbatim conformity: adopt the neighbour's elite unchanged.
                # The hard end of the specialist<->generalist dial — the
                # offspring IS the neighbour's controller, so accepted copies
                # make one controller cover several tasks (duplicates are
                # exact, which is what makes #distinct-controllers meaningful).
                offsprings.append(jax.tree.map(lambda leaf: leaf, y))
            else:  # iso_dd
                offsprings.append(
                    _iso_dd_crossover(
                        x,
                        y,
                        subkey,
                        self._iso_sigma,
                        self._line_sigma,
                        self._minval,
                        self._maxval,
                    )
                )
        key, subkey = jax.random.split(key)
        return self._apply_operators(
            self._social_learning_ops, tuple(offsprings), subkey, batch_size
        )

    def ask(
        self,
        repertoire: MONETRepertoire,
        monet_state: MONETState,
        key: RNGKey,
    ) -> Tuple[Genotype, jax.Array, ExtraScores]:
        """
        Generate a new batch of genotypes: sample the focal tasks, apply
        individual or social learning to their elites and return the
        offsprings together with the indices of the focal tasks on which
        they must be evaluated.

        Args:
            repertoire: the MONET repertoire
            monet_state: the MONET state, holding the task similarity graph
            key: a jax PRNG random key

        Returns:
            a new batch of genotypes, the indices of the tasks on which they
            must be evaluated and an empty extra information dictionary
        """
        batch_size = self._batch_size
        num_tasks = repertoire.centroids.shape[0]

        # sample the focal tasks and gather their elites
        key, subkey = jax.random.split(key)
        task_indices = jax.random.randint(subkey, (batch_size,), 0, num_tasks)
        x = jax.tree.map(lambda g: g[task_indices], repertoire.genotypes)
        focal_occupied = repertoire.fitnesses[task_indices, 0] > -jnp.inf

        # individual learning vs social learning
        key, subkey = jax.random.split(key)
        individual = jax.random.bernoulli(subkey, self._p_ind, (batch_size,))

        # select the neighbors and gather their elites
        key, subkey = jax.random.split(key)
        neighbor_indices = self._select_neighbors(
            repertoire, monet_state.neighbor_indices[task_indices], subkey
        )
        y = jax.tree.map(lambda g: g[neighbor_indices], repertoire.genotypes)
        neighbor_occupied = repertoire.fitnesses[neighbor_indices, 0] > -jnp.inf

        # conformity reads the whole neighbourhood (not the single neighbour
        # selected above) and builds its own candidate, so it carries its
        # own occupancy mask; when it is the only social operator that mask
        # and its neighbourhood representative also drive the fallback and
        # the init_strategy="copy" path below
        if "conformity" in self._social_learning_ops:
            key, subkey = jax.random.split(key)
            y_conformity, conformity_rep, conformity_occupied = (
                self._conformity_candidate(
                    repertoire,
                    x,
                    task_indices,
                    monet_state.neighbor_indices[task_indices],
                    subkey,
                )
            )
            if self._social_learning_ops == ("conformity",):
                neighbor_occupied = conformity_occupied
                y = conformity_rep
        else:
            y_conformity = y

        # individual learning: mutation of the focal elite
        key, subkey = jax.random.split(key)
        individual_offspring = self._individual_learning(x, subkey, batch_size)

        # social learning: crossover with the neighbor elite, with a
        # fallback to a Gaussian mutation when the neighbor has no elite
        key, subkey = jax.random.split(key)
        social_offspring = self._social_learning(
            x, y, subkey, batch_size, y_conformity=y_conformity
        )
        key, subkey = jax.random.split(key)
        social_offspring = _tree_where(
            neighbor_occupied,
            social_offspring,
            _gaussian_mutation(
                x, subkey, self._mutation_std, self._minval, self._maxval
            ),
        )

        genotypes = _tree_where(individual, individual_offspring, social_offspring)

        # focal tasks without elite: copy the neighbor elite (when social
        # learning is applied and the neighbor has one) or sample a random
        # solution
        key, subkey = jax.random.split(key)
        random_genotypes = _tree_uniform(x, subkey, self._minval, self._maxval)
        if self._init_strategy == "copy":
            init_genotypes = _tree_where(
                (~individual) & neighbor_occupied, y, random_genotypes
            )
        else:
            init_genotypes = random_genotypes
        genotypes = _tree_where(focal_occupied, genotypes, init_genotypes)

        return genotypes, task_indices, {}

    def tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        task_indices: jax.Array,
        repertoire: MONETRepertoire,
        monet_state: MONETState,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[MONETRepertoire, MONETState, Metrics]:
        """
        Add new genotypes to the repertoire.

        Args:
            genotypes: new genotypes to add to the repertoire
            fitnesses: fitnesses of the new genotypes on their tasks
            task_indices: indices of the tasks on which the new genotypes
                have been evaluated
            repertoire: the MONET repertoire
            monet_state: the MONET state
            extra_scores: extra scores of the new genotypes
        """
        if extra_scores is None:
            extra_scores = {}

        fitnesses = jnp.ravel(fitnesses)
        task_descriptors = repertoire.centroids[task_indices]

        # add genotypes to the repertoire: as the descriptor of an offspring
        # is exactly the descriptor of its task, it lands in the task's cell
        repertoire = repertoire.add(
            genotypes, task_descriptors, fitnesses, extra_scores
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, monet_state, metrics

    def update(
        self,
        repertoire: MONETRepertoire,
        monet_state: MONETState,
        key: RNGKey,
    ) -> Tuple[MONETRepertoire, MONETState, Metrics]:
        """
        Performs one iteration of the MONET algorithm.
        1. A batch of focal tasks is sampled uniformly at random.
        2. The elite of each focal task is either mutated (individual
            learning) or crossed-over with the elite of one of the
            neighbors of the task in the similarity graph (social
            learning).
        3. The offsprings are scored on their focal tasks and added to the
            repertoire if their fitness is greater than or equal to the
            fitness of the current elite.

        Args:
            repertoire: the MONET repertoire
            monet_state: the MONET state, holding the task similarity graph
            key: a jax PRNG random key

        Returns:
            the updated MONET repertoire, the MONET state and metrics about
            the repertoire
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        # generate offsprings on their focal tasks
        key, subkey = jax.random.split(key)
        genotypes, task_indices, _ = self.ask(repertoire, monet_state, subkey)

        # score the offsprings on the focal tasks
        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = self._scoring_function(
            genotypes, repertoire.centroids[task_indices], subkey
        )

        return self.tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            task_indices=task_indices,
            repertoire=repertoire,
            monet_state=monet_state,
            extra_scores=extra_scores,
        )

    def scan_update(
        self,
        carry: Tuple[MONETRepertoire, MONETState, RNGKey],
        _: Any,
    ) -> Tuple[Tuple[MONETRepertoire, MONETState, RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible
        with the jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the MONET state and a
                random key.
            _: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and MONET state, with a new random key
            and metrics.
        """
        repertoire, monet_state, key = carry
        key, subkey = jax.random.split(key)
        (
            repertoire,
            monet_state,
            metrics,
        ) = self.update(
            repertoire,
            monet_state,
            subkey,
        )

        return (repertoire, monet_state, key), metrics
