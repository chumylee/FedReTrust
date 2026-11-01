"""Reproducible label skew with fixed client sizes and preserved image membership.

We draw per-class Dirichlet preferences across clients, as in the pinned
Viceroy reference. We then fit both class totals and original client sizes by
alternating row/column scaling, and round with a capacitated residual flow.
The equal-size constraint is an explicit adaptation, not the authors' original
unconstrained Dirichlet allocation. Only original client-pool indices are used.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LabelSkewConfig:
    partition_seed: int = 20261017
    alpha: float = 0.5
    num_classes: int = 10
    balance_tolerance: float = 1e-8
    max_balance_iterations: int = 10000

    def __post_init__(self):
        if type(self.partition_seed) is not int or self.partition_seed < 0:
            raise ValueError("Partition seed must be a nonnegative integer.")
        for name in ("num_classes", "max_balance_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.alpha, bool) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("Dirichlet alpha must be finite and positive.")
        if (isinstance(self.balance_tolerance, bool) or not math.isfinite(self.balance_tolerance)
                or not 0 < self.balance_tolerance < 1e-3):
            raise ValueError("Use a small, finite positive margin tolerance.")


def _integer_totals(values):
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Margins must be one-dimensional integer arrays.")
    return values.astype(np.int64)


def fit_margins(preferences, row_totals, column_totals, options):
    """Fit a nonnegative table with specified integer row and column sums."""
    table = np.asarray(preferences, dtype=np.float64).copy()
    rows = _integer_totals(row_totals)
    cols = _integer_totals(column_totals)
    if (table.ndim != 2 or table.shape != (len(rows), len(cols))
            or not np.isfinite(table).all() or np.any(table <= 0)
            or np.any(rows <= 0) or np.any(cols < 0) or rows.sum() != cols.sum()):
        raise ValueError("Positive preferences, positive row sizes and matching nonnegative class totals required.")
    table *= cols
    for iteration in range(1, options.max_balance_iterations + 1):
        table *= (rows / table.sum(axis=1))[:, None]
        sums = table.sum(axis=0)
        scales = np.divide(cols, sums, out=np.ones_like(sums), where=sums > 0)
        table *= scales
        error = max(float(np.abs(table.sum(axis=1) - rows).max()),
                    float(np.abs(table.sum(axis=0) - cols).max()))
        if not np.isfinite(table).all():
            raise FloatingPointError("Non-finite margin fitting.")
        if error <= options.balance_tolerance:
            return table, {"iterations": iteration, "maximum_margin_error": error}
    raise RuntimeError("Margin fitting did not converge within its declared budget.")


def round_margins(table, row_totals, column_totals):
    """Round each cell down or up while preserving every integer margin.

    Floors leave a small bipartite flow problem. Row deficits supply units;
    fractional cells have capacity one; class deficits consume units. Reverse
    residual edges permit earlier allocations to be rerouted when necessary.
    No observations are dropped, duplicated, or moved to reserved sets.
    """
    table = np.asarray(table, dtype=np.float64)
    rows = _integer_totals(row_totals)
    cols = _integer_totals(column_totals)
    if (table.ndim != 2 or table.shape != (len(rows), len(cols))
            or not np.isfinite(table).all() or np.any(table < 0)
            or np.any(rows < 0) or np.any(cols < 0) or rows.sum() != cols.sum()
            or not np.allclose(table.sum(axis=1), rows, rtol=0, atol=1e-6)
            or not np.allclose(table.sum(axis=0), cols, rtol=0, atol=1e-6)):
        raise ValueError("Finite nonnegative table must already have the specified margins.")
    integer = np.floor(table).astype(np.int64)
    fractions = table - integer
    row_need = rows - integer.sum(axis=1)
    col_need = cols - integer.sum(axis=0)
    if np.any(row_need < 0) or np.any(col_need < 0):
        raise ValueError("Invalid floor deficits.")
    nr, nc = table.shape
    source, sink = nr + nc, nr + nc + 1
    capacity = np.zeros((sink + 1, sink + 1), dtype=np.int64)
    capacity[source, :nr] = row_need
    capacity[:nr, nr:nr + nc] = fractions > 0
    capacity[nr:nr + nc, sink] = col_need
    residual = capacity.copy()
    sent, needed = 0, int(row_need.sum())
    while sent < needed:
        parent = np.full(sink + 1, -1, dtype=np.int64)
        parent[source] = source
        queue = deque([source])
        while queue and parent[sink] == -1:
            node = queue.popleft()
            neighbors = np.flatnonzero(residual[node] > 0).tolist()
            if node < nr:
                neighbors.sort(key=lambda v: (-fractions[node, v - nr] if nr <= v < nr + nc else 0, v))
            for neighbor in neighbors:
                if parent[neighbor] == -1:
                    parent[neighbor] = node
                    queue.append(neighbor)
        if parent[sink] == -1:
            raise RuntimeError("Integer margin rounding has no feasible residual path.")
        amount, node = needed - sent, sink
        while node != source:
            previous = int(parent[node])
            amount = min(amount, int(residual[previous, node]))
            node = previous
        node = sink
        while node != source:
            previous = int(parent[node])
            residual[previous, node] -= amount
            residual[node, previous] += amount
            node = previous
        sent += amount
    integer += capacity[:nr, nr:nr + nc] - residual[:nr, nr:nr + nc]
    if (not np.array_equal(integer.sum(axis=1), rows)
            or not np.array_equal(integer.sum(axis=0), cols)
            or np.any(np.abs(integer - table) >= 1 + 1e-10)):
        raise AssertionError("Rounded class allocations violate margins or cell bounds.")
    return integer


def prepare_label_skew(labels, original_clients, options=None):
    """Redistribute exactly the original client pool, retaining each client's size.

    Input order is canonicalized. Separate local RNG streams generate class
    preferences, shuffle class members, and shuffle each final client list.
    Neither the NumPy global RNG nor the caller's inputs are modified.
    """
    options = LabelSkewConfig() if options is None else options
    if not isinstance(options, LabelSkewConfig):
        raise TypeError("Use LabelSkewConfig.")
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0) or np.any(labels >= options.num_classes)):
        raise ValueError("Expected integer labels in the configured class range.")
    if not original_clients or set(original_clients) != {str(i) for i in range(len(original_clients))}:
        raise ValueError("Client IDs must be consecutive strings starting at zero.")
    client_ids = [str(i) for i in range(len(original_clients))]
    values = [i for cid in client_ids for i in original_clients[cid]]
    if (not values or any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)) for i in values)
            or len(set(values)) != len(values) or min(values) < 0 or max(values) >= len(labels)):
        raise ValueError("Provide distinct valid original client-pool indices.")
    sizes = np.array([len(original_clients[cid]) for cid in client_ids], dtype=np.int64)
    if np.any(sizes <= 0):
        raise ValueError("Every client must have training images.")
    pool = np.array(sorted(values), dtype=np.int64)
    class_totals = np.bincount(labels[pool], minlength=options.num_classes)
    rng = np.random.default_rng(np.random.SeedSequence([options.partition_seed, 0]))
    preferences = rng.dirichlet(np.full(len(client_ids), options.alpha), size=options.num_classes).T
    continuous, balancing = fit_margins(preferences, sizes, class_totals, options)
    counts = round_margins(continuous, sizes, class_totals)
    clients = {cid: [] for cid in client_ids}
    for digit in range(options.num_classes):
        members = pool[labels[pool] == digit]
        digit_rng = np.random.default_rng(np.random.SeedSequence([options.partition_seed, 1, digit]))
        members = digit_rng.permutation(members)
        offset = 0
        for i, cid in enumerate(client_ids):
            stop = offset + int(counts[i, digit])
            clients[cid].extend(members[offset:stop].tolist())
            offset = stop
        assert offset == len(members)
    for i, cid in enumerate(client_ids):
        client_rng = np.random.default_rng(np.random.SeedSequence([options.partition_seed, 2, i]))
        clients[cid] = client_rng.permutation(clients[cid]).tolist()
    if sorted(i for group in clients.values() for i in group) != pool.tolist():
        raise AssertionError("The original client pool was not preserved.")
    actual_counts = class_counts(labels, clients, options.num_classes)
    if not np.array_equal(actual_counts, counts):
        raise AssertionError("Assigned labels do not match the planned count table.")
    return clients, {"config": asdict(options), "class_totals": class_totals.tolist(),
                     "client_sizes": sizes.tolist(), "preferences": preferences.tolist(),
                     "continuous_counts": continuous.tolist(), "integer_counts": counts.tolist(),
                     "rounding_maximum_cell_error": float(np.abs(counts - continuous).max()),
                     "balancing": balancing}


def class_counts(labels, clients, num_classes=10):
    labels = np.asarray(labels)
    return np.array([np.bincount(labels[clients[str(i)]], minlength=num_classes)
                     for i in range(len(clients))], dtype=np.int64)


def distribution_metrics(counts):
    """Descriptive label skew: entropy, dominant fraction, TV and JS in bits."""
    counts = np.asarray(counts, dtype=np.float64)
    if (counts.ndim != 2 or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts.sum(axis=1) <= 0)):
        raise ValueError("Provide a nonnegative class table with nonempty clients.")
    proportions = counts / counts.sum(axis=1, keepdims=True)
    global_proportions = counts.sum(axis=0) / counts.sum()
    middle = (proportions + global_proportions) / 2
    safe = lambda x: np.where(x > 0, x, 1)
    entropy = -(proportions * np.log2(safe(proportions))).sum(axis=1)
    js = .5 * (proportions * np.log2(safe(proportions) / safe(middle))).sum(axis=1)
    js += .5 * (global_proportions * np.log2(safe(global_proportions) / safe(middle))).sum(axis=1)
    return {"samples": counts.sum(axis=1).astype(int).tolist(),
            "dominant_digit": proportions.argmax(axis=1).tolist(),
            "dominant_fraction": proportions.max(axis=1).tolist(),
            "present_classes": (counts > 0).sum(axis=1).tolist(),
            "entropy_bits": entropy.tolist(), "effective_classes": np.exp2(entropy).tolist(),
            "total_variation_from_global": (.5 * np.abs(proportions - global_proportions).sum(axis=1)).tolist(),
            "js_divergence_from_global_bits": js.tolist()}
