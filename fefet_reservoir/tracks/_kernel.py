"""Scoring kernels: ridge readout, STM/PC targets, and the fixed-split and
k-fold delay loops.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "solve_ridge",
    "predict",
    "stm_target",
    "pc_target",
]


# ---------------------------------------------------------------------------
# ridge regression core
# ---------------------------------------------------------------------------


def solve_ridge(X, y, ridge_lambda: float) -> np.ndarray:
    """Ridge regression weights: ``inv(X'X + ridge_lambda*I) X' y``."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    d = X.shape[1]
    return np.linalg.inv(X.T @ X + ridge_lambda * np.eye(d)) @ X.T @ y


def predict(X, w) -> np.ndarray:
    """``X @ w`` -- apply fitted readout weights."""
    return np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# STM / PC target values and the column-major reshape
# ---------------------------------------------------------------------------


def _reshape_fortran(col: np.ndarray, samples_per_pulse: int, n_inputs: int) -> np.ndarray:
    """Reshape a flat response column into ``(samples_per_pulse, n_inputs)``
    in column-major (Fortran) order, so column *j* holds the
    ``samples_per_pulse`` samples of pulse *j* -- the stacking order of
    ``response.csv``.
    """
    return np.asarray(col, dtype=np.float64).reshape((samples_per_pulse, n_inputs), order="F")


def stm_target(u_in, delay: int) -> np.ndarray:
    """STM target: the input shifted forward by ``delay`` steps, zero-padded
    at the front.
    """
    u_in = np.asarray(u_in, dtype=np.float64).reshape(-1)
    n_inputs = u_in.size
    target = np.zeros(n_inputs, dtype=np.float64)
    if delay < n_inputs:
        target[delay:] = u_in[: n_inputs - delay]
    return target


def pc_target(u_in, delay: int) -> np.ndarray:
    """Parity-check target:
    ``d[n] = (u[n] + u[n-1] + ... + u[n-delay]) mod 2`` for ``n >= delay``,
    and ``0`` for the first ``delay`` entries.
    """
    u_in = np.asarray(u_in, dtype=np.float64).reshape(-1)
    n_inputs = u_in.size
    acc = u_in.copy()
    for rep in range(1, delay + 1):
        acc = acc + np.roll(u_in, rep)
    tail = acc[delay:]
    pc_target_seq = np.zeros(n_inputs, dtype=np.float64)
    pc_target_seq[delay:] = np.mod(tail, 2)
    return pc_target_seq


# ---------------------------------------------------------------------------
# the fixed train/test split loop (STM / PC)
# ---------------------------------------------------------------------------


def _fixed_split(
    inputs: np.ndarray,
    responses: np.ndarray,
    *,
    target_fn,
    train_sets: tuple[int, ...],
    test_sets: tuple[int, ...],
    samples_per_pulse: int,
    washout: int,
    ridge_lambda: float,
    max_delay: int,
) -> dict:
    """STM/PC fixed train/test split.

    ``target_fn`` builds a length-``n_inputs`` target from ``(u_in, delay)``
    (:func:`stm_target` or :func:`pc_target`). Sweeps
    ``delay = 0..max_delay``, fitting one ridge readout per delay on the
    train sets and scoring it on the test sets. No shuffling: this is the
    single-split protocol.

    ``train_sets`` / ``test_sets`` are 1-based set numbers.
    """
    n_inputs = inputs.shape[0]
    cor2 = np.zeros(max_delay + 1)

    for x in range(max_delay + 1):
        delay = x

        Xs, ys = [], []
        for set_no in train_sets:
            u_in = inputs[:, set_no - 1]
            features = _reshape_fortran(responses[:, set_no - 1], samples_per_pulse, n_inputs)
            Xs.append(features[:, washout:n_inputs].T)
            ys.append(target_fn(u_in, delay)[washout:n_inputs])
        x_train = np.vstack(Xs)
        y_train = np.concatenate(ys)
        w = solve_ridge(x_train, y_train, ridge_lambda)

        Xs_t, ys_t = [], []
        for set_no in test_sets:
            u_in = inputs[:, set_no - 1]
            features = _reshape_fortran(responses[:, set_no - 1], samples_per_pulse, n_inputs)
            Xs_t.append(features[:, washout:n_inputs].T)
            ys_t.append(target_fn(u_in, delay)[washout:n_inputs])
        x_test = np.vstack(Xs_t)
        y_test = np.concatenate(ys_t)
        y_pred = predict(x_test, w)

        cor = np.corrcoef(y_pred, y_test)[0, 1]
        cor2[x] = cor**2

    mc = float(np.sum(cor2[1:]))
    return {"mc": mc, "cor2": cor2}


# ---------------------------------------------------------------------------
# the k-fold cross-validation loop (STM / PC)
# ---------------------------------------------------------------------------


def _run_delay_benchmark(
    inputs,
    responses,
    *,
    target_fn,
    ridge_lambda: float,
    rng: np.random.Generator,
    samples_per_pulse: int,
    washout: int,
    max_delay: int,
    train_sets: tuple[int, ...],
    test_sets: tuple[int, ...],
) -> dict:
    """STM/PC k-fold cross-validation loop.

    Shuffles the dataset columns once with ``rng``, then for
    ``fold_num = num_dataset`` folds rolls the shuffled columns by ``fold``
    (a cyclic shift, ``np.roll``) and reruns the fixed train/test split at
    every delay. (The single unshuffled split is :func:`_fixed_split`.)
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)
    num_dataset = inputs.shape[1]
    n_inputs = inputs.shape[0]

    order = np.arange(num_dataset)
    rng.shuffle(order)
    input_mixed = inputs[:, order]
    meas_mixed = responses[:, order]
    fold_num = num_dataset

    mc_array = np.zeros(fold_num)
    cor2_array = np.zeros((max_delay + 1, fold_num))

    for fold in range(fold_num):
        input_cols = np.roll(input_mixed, fold, axis=1)
        meas_cols = np.roll(meas_mixed, fold, axis=1)

        cor2 = np.zeros(max_delay + 1)
        for x in range(max_delay + 1):
            delay = x

            Xs, ys = [], []
            for set_no in train_sets:
                u_in = input_cols[:, set_no - 1]
                features = _reshape_fortran(meas_cols[:, set_no - 1], samples_per_pulse, n_inputs)
                Xs.append(features[:, washout:n_inputs].T)
                ys.append(target_fn(u_in, delay)[washout:n_inputs])
            x_train = np.vstack(Xs)
            y_train = np.concatenate(ys)
            w = solve_ridge(x_train, y_train, ridge_lambda)

            Xs_t, ys_t = [], []
            for set_no in test_sets:
                u_in = input_cols[:, set_no - 1]
                features = _reshape_fortran(meas_cols[:, set_no - 1], samples_per_pulse, n_inputs)
                Xs_t.append(features[:, washout:n_inputs].T)
                ys_t.append(target_fn(u_in, delay)[washout:n_inputs])
            x_test = np.vstack(Xs_t)
            y_test = np.concatenate(ys_t)
            y_pred = predict(x_test, w)

            cor = np.corrcoef(y_pred, y_test)[0, 1]
            cor2[x] = cor**2

        mc_array[fold] = np.sum(cor2[1:])
        cor2_array[:, fold] = cor2

    return {
        "mc_per_fold": mc_array,
        "mc_kfold": float(np.mean(mc_array)),
        "cor2_kfold": np.mean(cor2_array, axis=1),
    }
