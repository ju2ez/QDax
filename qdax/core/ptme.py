"""Parametric-Task MAP-Elites (PT-ME) in JAX.

Anne & Mouret, "Parametric-Task MAP-Elites" (arXiv:2402.01275). This is a port of the numpy
reference implementation used by MONET (algorithms/pt_me.py in graph-elites), written so that
every stochastic decision has the same distribution as the reference at batch_size = 1:

* The archive is a CVT over the CONTINUOUS task space: one elite per centroid cell. Each elite
  stores its genotype, its fitness and the task vector ("situation") it was evaluated on, which
  is a sampled task, not the centroid.
* Initialisation evaluates one random genotype on each of the initial tasks and inserts it into
  the task's cell with ">=" acceptance.
* Each iteration draws one candidate:
  - with probability ``proba_regression``: sample a task ``s`` uniformly, take the elites in the
    precomputed neighbourhood of ``s``'s cell (Delaunay for task_dim <= 3, k-NN above) and fit a
    local linear regression genotype ~ situation over them (ordinary least squares with an
    intercept, minimum-norm when rank deficient, exactly ``sklearn.linear_model.LinearRegression``);
    the candidate is the prediction at ``s`` plus ``N(0, linreg_sigma)`` (ONE scalar draw) times
    the per-dimension standard deviation of the neighbours' genotypes, clipped to the bounds.
    With fewer than two filled neighbours it falls back to a uniformly chosen filled elite plus
    per-dimension ``N(0, linreg_sigma)`` noise; with an empty archive, to a uniform genotype.
  - otherwise: two elites ``x, y`` drawn uniformly among the filled cells, SBX crossover
    (Deb 2001, eta 10, the same operator as MONET's), evaluated on the task chosen by the
    CLOSEST-TO-PARENT tournament: draw ``k`` tasks uniformly and keep the one whose cell elite's
    situation is nearest to ``x``'s situation (tasks whose cell is empty are ignored; if all are
    empty, the first draw). ``k`` is the arm of a UCB1 bandit over ``tournament_sizes``, updated
    after every SBX evaluation with success = "the candidate was inserted"; until every arm has
    been pulled once the next arm is drawn uniformly among the unpulled ones.
  The candidate is evaluated on its task and inserted into the task's cell with ">=".

The reference is strictly sequential (one evaluation per iteration). Here ``batch_size``
candidates are drawn from the same archive state and then inserted ONE AT A TIME in order,
including the bandit updates, so batch_size = 1 reproduces the reference exactly. With larger
batches the parents are staler and every SBX candidate of a batch is drawn with the SAME
tournament size (the pre-batch arm); each is credited to that arm when inserted, and the arm
for the next batch is whatever the last SBX insertion selected.

Two reference behaviours are deliberately not reproduced because they are unreachable in
practice: (a) fitnesses must be FINITE (-inf is the empty-cell sentinel; the reference would
insert a NaN reward into an empty cell and freeze it there), and (b) the reference skips, but
still charges, an iteration in which 64 uniform cell draws all hit empty cells (probability
0.37^64 at 1% fill); parents here are drawn uniformly over the filled cells directly.

Fitness is used only through comparisons, so any monotone transform of the reference's reward
(the arm's exp(-distance) versus QDax's -distance) leaves the algorithm unchanged.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np

from qdax.core.monet import _sbx_crossover
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, Metrics, RNGKey


class PTMEState(flax.struct.PyTreeNode):
    """Archive plus bandit state.

    Attributes:
        genotypes: elites, (num_cells, ...). Undefined where ``fitnesses`` is -inf.
        fitnesses: (num_cells,), -inf marks an empty cell.
        situations: the task each elite was evaluated on, (num_cells, task_dim).
        bandit_successes: per tournament-size arm, number of inserted SBX candidates.
        bandit_selected: per arm, number of pulls.
        k_arm: index of the tournament size in use for the next SBX candidate.
        n_success / n_fail: counts of inserted / rejected main-loop candidates.
    """

    genotypes: Genotype
    fitnesses: Fitness
    situations: Descriptor
    bandit_successes: jax.Array
    bandit_selected: jax.Array
    k_arm: jax.Array
    n_success: jax.Array
    n_fail: jax.Array


def delaunay_cell_neighbors(centroids: np.ndarray, k_nn: int = 15) -> np.ndarray:
    """Cell neighbourhoods as the reference builds them: Delaunay simplices for task_dim <= 3,
    the k_nn nearest centroids above. Every cell is in its own neighbourhood. Returns an int
    array (num_cells, max_neighbors) padded with -1."""
    from scipy.spatial import Delaunay, cKDTree

    centroids = np.asarray(centroids, dtype=np.float64)
    n, dim = centroids.shape
    if dim > 3:
        k = min(k_nn, n - 1)
        nbrs = [list(cKDTree(centroids).query(centroids[i], k=k + 1)[1]) for i in range(n)]
    else:
        nbrs = [{i} for i in range(n)]
        for simplex in Delaunay(centroids).simplices:
            for i in simplex:
                nbrs[i].update(int(j) for j in simplex)
        nbrs = [sorted(s) for s in nbrs]
    width = max(len(s) for s in nbrs)
    out = -np.ones((n, width), dtype=np.int32)
    for i, s in enumerate(nbrs):
        out[i, : len(s)] = s
    return out


def _nearest_cell(centroids: jax.Array, task: jax.Array) -> jax.Array:
    return jnp.argmin(jnp.sum((centroids - task[None]) ** 2, axis=-1))


class PTME:
    """PT-ME with the reference's operators. See the module docstring."""

    def __init__(
        self,
        scoring_function: Callable[[Genotype, Descriptor, RNGKey], Tuple[Fitness, ExtraScores]],
        centroids: jax.Array,
        cell_neighbors: jax.Array,
        proba_regression: float = 0.5,
        linreg_sigma: float = 1.0,
        tournament_sizes: Tuple[int, ...] = (1, 5, 10, 50, 100, 500),
        sbx_eta: float = 10.0,
        minval: float = 0.0,
        maxval: float = 1.0,
        batch_size: int = 1,
        task_sampler: Optional[Callable[[RNGKey, int], Descriptor]] = None,
        metrics_function: Optional[Callable[[PTMEState], Metrics]] = None,
    ) -> None:
        self._scoring_function = scoring_function
        self._centroids = jnp.asarray(centroids)
        self._cell_neighbors = jnp.asarray(cell_neighbors, dtype=jnp.int32)
        self._proba_regression = float(proba_regression)
        self._linreg_sigma = float(linreg_sigma)
        self._tournament_sizes = jnp.asarray(tournament_sizes, dtype=jnp.int32)
        self._max_tournament = int(max(tournament_sizes))
        self._sbx_eta = float(sbx_eta)
        self._minval, self._maxval = float(minval), float(maxval)
        self._batch_size = int(batch_size)
        self._task_dim = int(self._centroids.shape[1])
        self._task_sampler = task_sampler or (
            lambda key, n: jax.random.uniform(key, (n, self._task_dim))
        )
        self._metrics_function = metrics_function or self.default_metrics

    @property
    def num_cells(self) -> int:
        return int(self._centroids.shape[0])

    # ------------------------------------------------------------------ init
    def init(
        self, init_genotypes: Genotype, init_tasks: Descriptor, key: RNGKey
    ) -> Tuple[PTMEState, Metrics]:
        """Evaluate ``init_genotypes[i]`` on ``init_tasks[i]`` and insert each into the task's
        cell with ">=" acceptance in index order (so among exact ties the last wins)."""
        key, subkey = jax.random.split(key)
        fitnesses, _ = self._scoring_function(init_genotypes, init_tasks, subkey)
        n = fitnesses.shape[0]
        cells = jax.vmap(partial(_nearest_cell, self._centroids))(init_tasks)
        num_cells = self.num_cells
        best = jax.ops.segment_max(fitnesses, cells, num_segments=num_cells)
        is_best = fitnesses == best[cells]
        winner = jax.ops.segment_max(
            jnp.where(is_best, jnp.arange(n), -1), cells, num_segments=num_cells
        )
        filled = winner >= 0
        safe = jnp.maximum(winner, 0)
        genotypes = jax.tree.map(lambda g: g[safe], init_genotypes)
        state = PTMEState(
            genotypes=genotypes,
            fitnesses=jnp.where(filled, best, -jnp.inf),
            situations=init_tasks[safe],
            bandit_successes=jnp.zeros_like(self._tournament_sizes, dtype=jnp.float32),
            bandit_selected=jnp.zeros_like(self._tournament_sizes, dtype=jnp.float32),
            k_arm=jnp.asarray(0, dtype=jnp.int32),   # reference starts at k_closest2parent = 1
            n_success=jnp.asarray(0, dtype=jnp.int32),
            n_fail=jnp.asarray(0, dtype=jnp.int32),
        )
        return state, self._metrics_function(state)

    # ------------------------------------------------------------- operators
    def _regression(self, state: PTMEState, s: jax.Array, key: RNGKey) -> Genotype:
        k_eps, k_pick, k_fb, k_uni = jax.random.split(key, 4)
        genos = state.genotypes                                         # leaves (num_cells, ...)
        filled = jnp.isfinite(state.fitnesses)
        cell = _nearest_cell(self._centroids, s)
        nb = self._cell_neighbors[cell]                                  # (width,), -1 padded
        safe_nb = jnp.maximum(nb, 0)
        valid = (nb >= 0) & filled[safe_nb]
        n_valid = valid.sum()
        w = valid / jnp.maximum(n_valid, 1)
        X = state.situations[safe_nb]                                    # (width, task_dim)
        mx = (w[:, None] * X).sum(0)
        Xc = (X - mx[None]) * valid[:, None]
        # sklearn LinearRegression: centre X and Y, minimum-norm least squares on the centred
        # data, intercept from the means. Zeroed rows contribute nothing to the fit.
        # sklearn solves the centred system with scipy lstsq(cond=1e-6). A neighbourhood with
        # n_valid <= task_dim rows is rank deficient by construction, so its remaining singular
        # direction is centring round-off and must be cut whatever the dtype.
        rtol = jnp.where(n_valid <= self._task_dim, 1e-3, 1e-6)
        pinv_Xc = jnp.linalg.pinv(Xc, rtol=rtol)                         # (task_dim, width)
        eps = jax.random.normal(k_eps, ())                               # ONE scalar, as np.random.normal(0, sigma)

        def _leaf(g: jax.Array) -> jax.Array:
            Y = g[safe_nb].reshape(nb.shape[0], -1)                      # (width, d)
            my = (w[:, None] * Y).sum(0)
            Yc = (Y - my[None]) * valid[:, None]
            W = pinv_Xc @ Yc                                             # (task_dim, d)
            pred = (s - mx) @ W + my
            std = jnp.sqrt((w[:, None] * (Y - my[None]) ** 2).sum(0))    # population std, ddof=0
            return (pred + eps * self._linreg_sigma * std).reshape(g.shape[1:])

        reg = jax.tree.map(_leaf, genos)

        # fallback: fewer than two filled neighbours -> a uniformly chosen filled elite + noise
        p = jnp.where(filled.any(), filled / jnp.maximum(filled.sum(), 1), 1.0 / self.num_cells)
        pick = jax.random.choice(k_pick, self.num_cells, p=p)
        leaves, treedef = jax.tree.flatten(genos)
        fb_keys = jax.random.split(k_fb, len(leaves))
        fb = jax.tree.unflatten(
            treedef,
            [g[pick] + jax.random.normal(k, g.shape[1:]) * self._linreg_sigma
             for g, k in zip(leaves, fb_keys)],
        )
        uni_keys = jax.random.split(k_uni, len(leaves))
        uni = jax.tree.unflatten(
            treedef,
            [jax.random.uniform(k, g.shape[1:], minval=self._minval, maxval=self._maxval)
             for g, k in zip(leaves, uni_keys)],
        )
        out = jax.tree.map(
            lambda r, f, u: jnp.where(n_valid >= 2, r, jnp.where(filled.any(), f, u)), reg, fb, uni
        )
        return jax.tree.map(lambda g: jnp.clip(g, self._minval, self._maxval), out)

    def _closest_to_parent(
        self, state: PTMEState, parent_situation: jax.Array, k: jax.Array, key: RNGKey
    ) -> jax.Array:
        tasks = self._task_sampler(key, self._max_tournament)            # (K_max, task_dim)
        cells = jax.vmap(partial(_nearest_cell, self._centroids))(tasks)
        filled = jnp.isfinite(state.fitnesses)
        valid = (jnp.arange(self._max_tournament) < k) & filled[cells]
        dist = jnp.sum((state.situations[cells] - parent_situation[None]) ** 2, axis=-1)
        dist = jnp.where(valid, dist, jnp.inf)
        j = jnp.argmin(dist)
        return jnp.where(valid.any(), tasks[j], tasks[0])

    def _sbx_candidate(self, state: PTMEState, key: RNGKey) -> Tuple[Genotype, jax.Array]:
        k_x, k_y, k_sbx, k_t = jax.random.split(key, 4)
        filled = jnp.isfinite(state.fitnesses)
        p = jnp.where(filled.any(), filled / jnp.maximum(filled.sum(), 1), 1.0 / self.num_cells)
        ix = jax.random.choice(k_x, self.num_cells, p=p)
        iy = jax.random.choice(k_y, self.num_cells, p=p)
        x = jax.tree.map(lambda g: g[ix], state.genotypes)
        y = jax.tree.map(lambda g: g[iy], state.genotypes)
        child = _sbx_crossover(x, y, k_sbx, self._sbx_eta, self._minval, self._maxval)
        k = self._tournament_sizes[state.k_arm]
        task = self._closest_to_parent(state, state.situations[ix], k, k_t)
        return child, task

    def _one_candidate(
        self, state: PTMEState, key: RNGKey
    ) -> Tuple[Genotype, jax.Array, jax.Array]:
        k_u, k_s, k_reg, k_sbx = jax.random.split(key, 4)
        use_regression = jax.random.uniform(k_u) < self._proba_regression
        s_reg = self._task_sampler(k_s, 1)[0]
        g_reg = self._regression(state, s_reg, k_reg)
        g_sbx, s_sbx = self._sbx_candidate(state, k_sbx)
        genotype = jax.tree.map(
            lambda a, b: jnp.where(use_regression, a, b), g_reg, g_sbx
        )
        task = jnp.where(use_regression, s_reg, s_sbx)
        return genotype, task, use_regression, state.k_arm

    # ------------------------------------------------------------- ask / tell
    def ask(
        self, state: PTMEState, key: RNGKey
    ) -> Tuple[Genotype, Descriptor, jax.Array, jax.Array]:
        """Draw ``batch_size`` candidates from the current archive.

        Returns (genotypes, tasks, is_regression, arm_used), each with a leading batch axis;
        ``arm_used`` is the bandit arm whose tournament size drew the candidate's task."""
        keys = jax.random.split(key, self._batch_size)
        return jax.vmap(self._one_candidate, in_axes=(None, 0))(state, keys)

    def tell(
        self,
        state: PTMEState,
        genotypes: Genotype,
        tasks: Descriptor,
        fitnesses: Fitness,
        is_regression: jax.Array,
        arm_used: jax.Array,
        key: RNGKey,
    ) -> PTMEState:
        """Insert the evaluated candidates ONE AT A TIME in batch order, updating the bandit
        after every SBX candidate, exactly as the reference's sequential loop does."""

        def _insert(st: PTMEState, item):
            g, task, fit, reg, arm, k_item = item
            cell = _nearest_cell(self._centroids, task)
            is_elite = fit >= st.fitnesses[cell]              # empty cell is -inf: always accepted
            new_genotypes = jax.tree.map(
                lambda arr, val: arr.at[cell].set(jnp.where(is_elite, val, arr[cell])),
                st.genotypes, g,
            )
            new_fit = st.fitnesses.at[cell].set(jnp.where(is_elite, fit, st.fitnesses[cell]))
            new_sit = st.situations.at[cell].set(jnp.where(is_elite, task, st.situations[cell]))

            # bandit: only SBX ("closest2parent") candidates are pulls, credited to the arm
            # that drew them (== st.k_arm at batch_size 1; earlier than the carry otherwise)
            succ = st.bandit_successes.at[arm].add(jnp.where(reg, 0.0, is_elite.astype(jnp.float32)))
            sel = st.bandit_selected.at[arm].add(jnp.where(reg, 0.0, 1.0))
            unpulled = sel == 0
            total = sel.sum()
            mean = succ / jnp.maximum(sel, 1.0)
            conf = jnp.sqrt(2.0 * jnp.log(jnp.maximum(total, 1.0)) / jnp.maximum(sel, 1.0))
            ucb_arm = jnp.argmax(mean + conf)
            explore_p = unpulled / jnp.maximum(unpulled.sum(), 1)
            explore_arm = jax.random.choice(k_item, sel.shape[0], p=explore_p)
            next_arm = jnp.where(unpulled.any(), explore_arm, ucb_arm)
            new_arm = jnp.where(reg, st.k_arm, next_arm.astype(jnp.int32))

            st = st.replace(
                genotypes=new_genotypes, fitnesses=new_fit, situations=new_sit,
                bandit_successes=succ, bandit_selected=sel, k_arm=new_arm,
                n_success=st.n_success + is_elite.astype(jnp.int32),
                n_fail=st.n_fail + (~is_elite).astype(jnp.int32),
            )
            return st, None

        item_keys = jax.random.split(key, self._batch_size)
        state, _ = jax.lax.scan(
            _insert, state, (genotypes, tasks, fitnesses, is_regression, arm_used, item_keys)
        )
        return state

    def update(self, state: PTMEState, key: RNGKey) -> Tuple[PTMEState, Metrics]:
        key, k_ask, k_score, k_tell = jax.random.split(key, 4)
        genotypes, tasks, is_reg, arm_used = self.ask(state, k_ask)
        fitnesses, _ = self._scoring_function(genotypes, tasks, k_score)
        state = self.tell(state, genotypes, tasks, fitnesses, is_reg, arm_used, k_tell)
        return state, self._metrics_function(state)

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(self, carry: Tuple[PTMEState, RNGKey], _: None):
        state, key = carry
        key, subkey = jax.random.split(key)
        state, metrics = self.update(state, subkey)
        return (state, key), metrics

    # --------------------------------------------------------------- metrics
    def default_metrics(self, state: PTMEState) -> Metrics:
        filled = jnp.isfinite(state.fitnesses)
        n = jnp.maximum(filled.sum(), 1)
        return {
            "coverage": filled.mean(),
            "mean_fitness": jnp.where(filled, state.fitnesses, 0.0).sum() / n,
            "max_fitness": jnp.where(filled, state.fitnesses, -jnp.inf).max(),
            "k_closest2parent": self._tournament_sizes[state.k_arm],   # the SIZE, as the reference logs
        }
