"""Orakel-Tests (unabhängiger Rechenweg, klein und schnell): Perplexity-Kalibrierung per scipy-Nullstellensuche, KL(P‖Q) im Log-Raum
(ohne Untergrenze), Gradient per zentrierter finiter Differenz und eine Schleifen-Neuimplementierung der Optimierung (explizite Paarsummen)."""

import numpy as np
import pytest
from scipy.optimize import brentq

from tsne_algorithm import conditional_affinities, fit_tsne, gradient, joint_affinities, kl_divergence, low_dim_affinities, squared_distances


def _row_p(d_row, i, beta):
    s = d_row.copy()
    s[i] = np.inf
    s = s - s.min()
    w = np.exp(-s * beta)
    w[i] = 0.0
    return w / w.sum()


def _entropy(d_row, i, beta):
    p = _row_p(d_row, i, beta)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _true_kl(P, Y, kernel):
    """KL(P‖Q) im Log-Raum, ohne Untergrenze für Q."""
    d2 = ((Y[:, None] - Y[None]) ** 2).sum(-1)
    off = ~np.eye(len(Y), dtype=bool)
    logw = -d2 if kernel == "gauss" else -np.log1p(d2)
    lw = logw[off]
    log_z = lw.max() + np.log(np.exp(lw - lw.max()).sum())
    mask = P > 0
    return float((P[mask] * (np.log(P[mask]) - (logw - log_z)[mask])).sum())


def test_perplexity_calibration_matches_root_finding_and_handles_duplicates():
    rng = np.random.default_rng(1)
    for it in range(12):
        n = int(rng.integers(8, 25))
        Z = rng.standard_normal((n, 3)) * rng.choice([1.0, 3.0], 3)
        if it % 4 == 0:
            Z[1] = Z[0]
        d2 = squared_distances(Z)
        perplexity = float(rng.uniform(2.0, n - 3.0))
        cond, beta = conditional_affinities(d2, perplexity)
        for i in range(n):
            lb = brentq(lambda x: _entropy(d2[i], i, np.exp(x)) - np.log(perplexity), -30, 30, xtol=1e-13)
            assert np.abs(_row_p(d2[i], i, np.exp(lb)) - cond[i]).max() < 1e-6
            assert abs(np.exp(_entropy(d2[i], i, beta[i])) - perplexity) < 1e-5 * perplexity
        P, _, _ = joint_affinities(d2, perplexity)
        assert abs(P.sum() - 1.0) < 1e-12 and np.allclose(P, P.T) and not np.diag(P).any()


@pytest.mark.parametrize("kernel", ["student", "gauss"])
@pytest.mark.parametrize("spread", [0.3, 1.0, 3.0])
def test_gradient_and_reported_kl_match_the_log_space_objective(kernel, spread):
    """Auch bei weit gestreuten Punkten (Gauß-Kern: q < 1e-12) muss die gemeldete KL die echte KL(P‖Q) sein."""
    rng = np.random.default_rng(2)
    n = 22
    A = rng.random((n, n))
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    P = A / A.sum()
    Y = rng.standard_normal((n, 2)) * spread
    g, Q = gradient(P, Y, kernel)
    assert abs(kl_divergence(P, Q) - _true_kl(P, Y, kernel)) < 1e-9
    for i, c in ((0, 0), (5, 1), (13, 0), (21, 1)):
        Yp, Ym = Y.copy(), Y.copy()
        Yp[i, c] += 1e-6
        Ym[i, c] -= 1e-6
        numeric = (_true_kl(P, Yp, kernel) - _true_kl(P, Ym, kernel)) / 2e-6
        assert abs(g[i, c] - numeric) < 1e-6 * max(1.0, abs(numeric))


def test_gauss_kernel_q_is_not_clamped_at_a_large_floor():
    Y = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [8.0, 8.0]])
    Q, _ = low_dim_affinities(Y, "gauss")
    assert Q.min() < 1e-20


def _reference_fit(Z, perplexity, n_iter, lr, exaggeration, kernel):
    n = len(Z)
    d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
    cond = np.zeros((n, n))
    for i in range(n):
        lb = brentq(lambda x: _entropy(d2[i], i, np.exp(x)) - np.log(perplexity), -30, 30, xtol=1e-14)
        cond[i] = _row_p(d2[i], i, np.exp(lb))
    P = (cond + cond.T) / (2 * n)
    Zc = Z - Z.mean(0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    Y = Zc @ vt[:2].T
    Y = Y / Y[:, 0].std() * 1e-4
    update, gains = np.zeros_like(Y), np.ones_like(Y)
    exag_iters = min(250, n_iter // 4)
    kl_history = []
    for it in range(1, n_iter + 1):
        Pe = P * exaggeration if it <= exag_iters else P
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dd = ((Y[i] - Y[j]) ** 2).sum()
                    W[i, j] = 1 / (1 + dd) if kernel == "student" else np.exp(-dd)
        Q = W / W.sum()
        G = np.zeros_like(Y)
        for i in range(n):
            for j in range(n):
                if i != j:
                    factor = (Pe[i, j] - Q[i, j]) * (W[i, j] if kernel == "student" else 1.0)
                    G[i] += 4 * factor * (Y[i] - Y[j])
        kl_history.append(sum(P[i, j] * np.log(P[i, j] / Q[i, j]) for i in range(n) for j in range(n) if i != j and P[i, j] > 0))
        for i in range(n):
            for c in range(2):
                gains[i, c] = gains[i, c] + 0.2 if update[i, c] * G[i, c] < 0 else gains[i, c] * 0.8
                gains[i, c] = max(gains[i, c], 0.01)
        update = (0.5 if it <= exag_iters else 0.8) * update - lr * gains * G
        Y = Y + update
        Y = Y - Y.mean(0)
    return Y, np.array(kl_history)


@pytest.mark.parametrize("kernel,lr", [("student", 25.0), ("gauss", 1.0)])
def test_fit_matches_a_loop_reimplementation(kernel, lr):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((14, 3)) @ rng.standard_normal((3, 5))
    model = fit_tsne(X, 4.0, 30, lr, 12.0, kernel)
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    Y, kl_history = _reference_fit(Z, 4.0, 30, lr, 12.0, kernel)
    assert np.abs(model.embedding - Y).max() < 1e-6
    assert np.abs(model.kl_history - kl_history).max() < 1e-6
