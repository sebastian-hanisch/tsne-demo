import numpy as np
import pytest
from scipy.spatial import procrustes as scipy_procrustes
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from sklearn.manifold._t_sne import _joint_probabilities

from tsne_algorithm import (
    embed_new_naive, fit_tsne, gradient, joint_affinities, kl_divergence, low_dim_affinities, pca_init, procrustes_disparity, row_perplexity, snapshot_iterations,
    squared_distances,
)
from tsne_isomap import fit_isomap, standardize
from tsne_lle import SingularNeighbourhood, fit_lle
from tsne_scenario import generate_dataset


def _data(n=120, d=6, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 3, n)
    return np.column_stack([np.cos(t), np.sin(t), t, 0.1 * rng.standard_normal((n, d - 3))]) if d > 3 else np.column_stack([np.cos(t), np.sin(t), t])


def test_affinities_hit_the_target_perplexity_and_are_symmetric_and_normalised():
    Z = standardize(generate_dataset(200, 2, 1.0, 0.25, 0, 7).X)
    for perplexity in (5, 30, 80):
        P, cond, sigma = joint_affinities(squared_distances(Z), perplexity)
        assert np.allclose(row_perplexity(cond), perplexity, atol=1e-6)
        assert np.allclose(cond.sum(1), 1.0) and np.allclose(np.diag(cond), 0.0)
        assert np.allclose(P, P.T) and abs(P.sum() - 1.0) < 1e-12 and (sigma > 0).all()


def test_joint_affinities_match_sklearn():
    Z = standardize(generate_dataset(150, 2, 1.0, 0.25, 0, 7).X)
    d2 = squared_distances(Z)
    P, _, _ = joint_affinities(d2, 30)
    sk = squareform(_joint_probabilities(d2.astype(np.float32), 30, 0))
    assert np.abs(sk - P).max() < 2e-6


def test_perplexity_must_be_below_n_minus_one():
    Z = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError):
        joint_affinities(squared_distances(Z), 19)
    joint_affinities(squared_distances(Z), 18)


@pytest.mark.parametrize("kernel", ["student", "gauss"])
def test_gradient_matches_finite_differences(kernel):
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((30, 2))
    P, _, _ = joint_affinities(squared_distances(rng.standard_normal((30, 5))), 8)
    g, _ = gradient(P, Y, kernel)
    for i, c in ((0, 0), (3, 1), (17, 0), (29, 1)):
        Yp, Ym = Y.copy(), Y.copy()
        Yp[i, c] += 1e-6
        Ym[i, c] -= 1e-6
        numeric = (kl_divergence(P, low_dim_affinities(Yp, kernel)[0]) - kl_divergence(P, low_dim_affinities(Ym, kernel)[0])) / 2e-6
        assert abs(g[i, c] - numeric) < 1e-7


def test_gauss_kernel_does_not_underflow_for_far_apart_points():
    Y = np.array([[0.0, 0.0], [40.0, 0.0], [0.0, 40.0], [40.0, 40.0]])
    Q, _ = low_dim_affinities(Y, "gauss")
    assert np.isfinite(Q).all() and abs(Q.sum() - 1.0) < 1e-6


def test_fit_matches_sklearn_exact_tsne():
    """Gleiche Standardisierung, PCA-Start, Lernrate, Exaggeration (250 Iterationen wie bei sklearn), 750 Iterationen: dieselbe KL-Divergenz und dieselbe Einbettung bis auf Drehung.
    (Bei kleinem n, z. B. 120 Touren, landen beide Implementierungen in verschiedenen lokalen Minima - die Optimierung ist chaotisch; deshalb 300 Touren, dort stimmen sie auf 5 Stellen überein.)"""
    ds = generate_dataset(300, 2, 1.0, 0.25, 0, 7)
    ours = fit_tsne(ds.X, 30, 1000)
    Z = standardize(ds.X)
    sk = TSNE(2, perplexity=30, init="pca", learning_rate="auto", max_iter=1000, random_state=0, method="exact").fit(Z)
    assert ours.exaggeration_iters == 250 and abs(ours.kl - sk.kl_divergence_) < 0.02
    assert procrustes_disparity(ours.embedding, sk.embedding_) < 0.05


def test_pca_init_is_deterministic_and_random_init_depends_only_on_its_own_seed():
    X = generate_dataset(100, 2, 1.0, 0.25, 0, 7).X
    a, b = fit_tsne(X, 10, 60), fit_tsne(X, 10, 60)
    assert np.array_equal(a.embedding, b.embedding)
    r0, r0b, r1 = fit_tsne(X, 10, 60, init="random", seed=0), fit_tsne(X, 10, 60, init="random", seed=0), fit_tsne(X, 10, 60, init="random", seed=1)
    assert np.array_equal(r0.embedding, r0b.embedding) and not np.array_equal(r0.embedding, r1.embedding)
    assert abs(np.std(pca_init(standardize(X))[:, 0]) - 1e-4) < 1e-9


def test_kl_falls_and_snapshots_are_recorded():
    ds = generate_dataset(150, 2, 1.0, 0.25, 0, 7)
    m = fit_tsne(ds.X, 20, 300)
    assert m.kl_history.shape == (300,) and m.kl_history[-1] < m.kl_history[m.exaggeration_iters] < m.kl_history[0] * 1.001 + 1
    assert m.kl_history[-1] < 0.6 * m.kl_history[0]
    assert set(m.snapshots) == set(snapshot_iterations(300, m.exaggeration_iters)) and 0 in m.snapshots and 300 in m.snapshots
    assert np.array_equal(m.snapshots[300], m.embedding) and abs(m.embedding.mean()) < 1e-9


def test_exaggeration_phase_is_a_quarter_of_the_iterations_capped_at_250():
    X = generate_dataset(80, 2, 1.0, 0.25, 0, 7).X
    assert fit_tsne(X, 10, 40).exaggeration_iters == 10
    assert fit_tsne(X, 10, 1100).exaggeration_iters == 250


def test_learning_rate_auto_and_explicit():
    X = generate_dataset(300, 2, 1.0, 0.25, 0, 7).X
    assert fit_tsne(X, 30, 20).learning_rate == 50.0
    assert fit_tsne(X, 30, 20, learning_rate=123.0).learning_rate == 123.0
    assert fit_tsne(generate_dataset(600, 2, 1.0, 0.25, 0, 7).X, 30, 20).learning_rate == 50.0       # max(600 / 12 / 4, 50): für alle n der Demo 50


def test_invalid_options_are_rejected():
    X = generate_dataset(60, 2, 1.0, 0.25, 0, 7).X
    with pytest.raises(ValueError):
        fit_tsne(X, 10, 20, kernel="cauchy")
    with pytest.raises(ValueError):
        fit_tsne(X, 10, 20, init="spectral")


def test_naive_out_of_sample_lands_near_the_training_coordinates_of_the_same_point():
    ds = generate_dataset(150, 2, 1.0, 0.25, 0, 7)
    m = fit_tsne(ds.X, 20, 300)
    y = embed_new_naive(m, ds.X[:10], 5)
    assert np.abs(y - m.embedding[:10]).max() < 0.25 * np.abs(m.embedding).max()


def test_procrustes_matches_scipy_and_ignores_rotation_scale_and_reflection():
    rng = np.random.default_rng(1)
    A, B = rng.standard_normal((50, 2)), rng.standard_normal((50, 2))
    assert abs(procrustes_disparity(A, B) - scipy_procrustes(A, B)[2]) < 1e-12
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    assert procrustes_disparity(A, 3.0 * A @ R) < 1e-12 and procrustes_disparity(A, A * [1, -1]) < 1e-12


def test_copied_isomap_and_lle_match_their_reference_values():
    ds = generate_dataset(300, 2, 1.0, 0.25, 0, 7)
    r = fit_isomap(ds.X, 8, 2)
    assert abs(float(np.abs(r.embedding).sum()) - 1786.6726227517283) < 1e-6 and abs(float(r.geodesic.sum()) - 621595.593373937) < 1e-3
    lle = fit_lle(ds.X, 14, 2, 1e-2)
    assert np.allclose(lle.embedding.mean(0), 0, atol=1e-9) and abs(lle.eigenvalues[0]) < 1e-10
    with pytest.raises(SingularNeighbourhood):
        fit_lle(ds.X, 20, 2, 0.0)
