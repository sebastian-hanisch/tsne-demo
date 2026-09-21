import numpy as np
import pytest
from sklearn.manifold import trustworthiness as sk_trustworthiness

import tsne_constants as C
from tsne_evaluation import (
    Settings, analyse, convergence_rows, crowding, distance_fidelity, distance_fidelity_split, make_dataset, out_of_sample, pca_project, perplexity_status, perplexity_sweep,
    r2_quadratic, stability, timing_sweep, trustworthiness, verdict,
)
from tsne_scenario import generate_dataset


def test_scenario_is_bit_identical_to_the_pca_demo_generator():
    """Eingefrorene Referenzwerte aus pca-demo (`generate_dataset`, gleiche Argumente): die Kopie darf nicht abweichen."""
    d = generate_dataset(300, 2, 1.0, 0.25, 0, 7)
    assert d.X.shape == (300, 12) and abs(float(d.X.sum()) - 14553337.310875032) < 1e-6
    assert np.allclose(d.X[0, :3], [54469.13732407575, 74.83224937121233, 767.2459701758783]) and abs(float(d.z.sum()) - (-80.2378453142044)) < 1e-9
    assert abs(float(generate_dataset(200, 3, 0.4, 0.3, 5, 42).X.sum()) - 10763498.969287368) < 1e-6


def test_trustworthiness_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6))
    for embedding in (X[:, :2], rng.standard_normal((80, 2)), pca_project(X)):
        assert abs(trustworthiness(X, embedding, 8) - sk_trustworthiness(X, embedding, n_neighbors=8)) < 1e-9


def test_r2_quadratic_recovers_monotone_reparametrisations_and_rejects_noise():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((200, 2))
    coords = np.column_stack([z[:, 0] + 0.3 * z[:, 0] ** 2, z[:, 1] + 0.2 * z[:, 1] ** 2])
    assert r2_quadratic(coords, z) > 0.9
    assert r2_quadratic(rng.standard_normal((200, 2)), z) < 0.1


def test_distance_fidelity_is_one_for_a_scaled_copy_and_split_reports_near_and_far():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((100, 2))
    assert distance_fidelity(3.0 * z, z) > 0.999999
    near, far = distance_fidelity_split(3.0 * z, z)
    assert near > 0.999999 and far > 0.999999
    # Abstände oberhalb der Mediangrenze auf einen Wert gestaucht: nahe Paare bleiben treu, ferne verlieren die Ordnung
    squashed = z * (1.0 / (1.0 + np.linalg.norm(z, axis=1, keepdims=True) ** 2))
    near2, far2 = distance_fidelity_split(squashed, z)
    assert near2 > far2


def test_analysis_on_the_default_surface_reproduces_the_measured_headline():
    ds = make_dataset(300, 2, 1.0, 0.25, 0, 7)
    a = analyse(ds, Settings())
    m = a.metrics
    assert m["tsne"]["r2"] > 0.88 and m["isomap"]["r2"] > 0.94 and m["pca"]["r2"] < 0.6
    assert m["tsne"]["far"] < m["isomap"]["far"] - 0.1                         # t-SNE erhält ferne Abstände schlechter als Isomap
    assert a.alt is None and "alt" not in m and a.tsne.kernel == "student"
    assert len(a.iso_indices) == 300 and sorted(a.snapshot_r2) == [it for it in sorted(a.tsne.snapshots) if it > 0]
    assert convergence_rows(a)[-1][0] == 750 and abs(convergence_rows(a)[-1][1] - m["tsne"]["r2"]) < 1e-12


def test_gauss_analysis_also_fits_the_student_kernel_for_comparison():
    ds = make_dataset(300, 3, 1.0, 0.25, 0, 7)
    a = analyse(ds, Settings(kernel="gauss"))
    assert a.alt is not None and a.alt.kernel == "student" and a.tsne.kernel == "gauss"
    assert a.metrics["alt"]["trust"] > a.metrics["tsne"]["trust"] + 0.04 and a.metrics["alt"]["kl"] < a.tsne.kl


def test_sonderfahrten_break_tsne_but_not_pca_or_isomap():
    ds = make_dataset(300, 2, 1.0, 0.25, 5, 7)
    m = analyse(ds, Settings()).metrics
    assert m["tsne"]["r2"] < 0.3 < 0.6 < m["pca"]["r2"] and m["isomap"]["r2"] > 0.6
    assert m["tsne"]["far"] < 0.3 and m["pca"]["far"] > 0.85


def test_early_exaggeration_off_costs_far_pair_fidelity():
    """Belegt die Aussage in der Hilfe des Reglers 'Early Exaggeration' (0.69 -> 0.33 bei Seed 7, 300 Touren)."""
    ds = make_dataset(300, 2, 1.0, 0.25, 0, 7)
    far_on = analyse(ds, Settings(exaggeration=12)).metrics["tsne"]["far"]
    far_off = analyse(ds, Settings(exaggeration=1)).metrics["tsne"]["far"]
    assert 0.6 < far_on < 0.8 and 0.2 < far_off < 0.45


def test_perplexity_sweep_is_deterministic_and_shows_the_small_end_breaking():
    rows = perplexity_sweep(2, 1.0, 0.25, 0, n_tours=120, n_iter=200, perplexities=(3, 10, 30), seeds=(100_000, 100_001))
    assert rows == perplexity_sweep(2, 1.0, 0.25, 0, n_tours=120, n_iter=200, perplexities=(3, 10, 30), seeds=(100_000, 100_001))
    assert rows[0]["r2"] < rows[1]["r2"] and [r["perplexity"] for r in rows] == [3, 10, 30]


def test_sweep_seeds_are_separate_from_demo_seeds():
    assert min(C.SWEEP_SEEDS) >= 100_000 > C.DEFAULT_SEED


def test_perplexity_status_only_flags_small_perplexities():
    rows = [{"perplexity": p, "r2": r} for p, r in ((3, 0.58), (5, 0.75), (10, 0.9), (20, 0.86), (30, 0.9), (50, 0.93), (80, 0.88))]
    assert perplexity_status(rows, 3) == "small" and perplexity_status(rows, 30) is None and perplexity_status(rows, 100) is None
    assert perplexity_status([{"perplexity": p, "r2": 0.5} for p in (3, 10, 30)], 3) is None       # flache Kurve: nichts zu melden


@pytest.mark.parametrize("q,curv,noise,out,settings,code", [
    (2, 1.0, 0.25, 0, Settings(), "tsne_wins"),
    (2, 1.0, 0.25, 5, Settings(), "global_structure"),
    (2, 1.0, 0.25, 0, Settings(n_iter=50), "not_converged"),
    (2, 1.0, 0.25, 0, Settings(learning_rate=2000.0), "lr_high"),
    (3, 1.0, 0.25, 0, Settings(kernel="gauss"), "crowding"),
    (2, 0.0, 0.25, 0, Settings(), "no_advantage"),
])
def test_verdict_codes(q, curv, noise, out, settings, code):
    ds = make_dataset(300, q, curv, noise, out, 7)
    rows = perplexity_sweep(q, curv, noise, out, n_tours=120, n_iter=150, perplexities=(3, 10, 30), seeds=(100_000,))
    assert verdict(analyse(ds, settings), ds, settings, rows)[1] == code


def test_high_learning_rate_ends_at_a_much_higher_kl_on_every_dataset():
    """Das R² bei großer Lernrate ist chaotisch (auf CI 0.48 statt lokal 0.26) - die KL-Divergenz ist das robuste Zeichen; deshalb stützt sich Verdict und Preset darauf."""
    for seed in (7, 8, 9):
        ds = make_dataset(300, 2, 1.0, 0.25, 0, seed)
        high = analyse(ds, Settings(learning_rate=2000.0))
        assert high.ref is not None and high.tsne.kl > 1.5 * high.ref.kl
    assert analyse(make_dataset(300, 2, 1.0, 0.25, 0, 7), Settings(learning_rate=200.0)).ref is not None
    assert analyse(make_dataset(300, 2, 1.0, 0.25, 0, 7), Settings(learning_rate=50.0)).ref is None


def test_divergence_is_detected_and_the_last_finite_state_is_kept():
    ds = make_dataset(300, 4, 1.0, 1.0, 10, 7)
    s = Settings(kernel="gauss", learning_rate=2000.0)
    a = analyse(ds, s)
    assert a.tsne.diverged_at > 0 and np.isfinite(a.tsne.embedding).all() and np.isfinite(a.tsne.kl_history).all()
    assert set(a.tsne.snapshots) == set(sorted(a.tsne.snapshots)) and a.tsne.n_iter in a.tsne.snapshots
    assert verdict(a, ds, s, None)[1] == "diverged"
    assert analyse(ds, Settings()).tsne.diverged_at == 0


def test_verdict_perplexity_small_uses_the_sweep_rows():
    ds = make_dataset(300, 2, 1.0, 0.25, 0, 7)
    s = Settings(perplexity=3)
    rows = [{"perplexity": p, "r2": r} for p, r in ((3, 0.58), (10, 0.9), (30, 0.9))]
    assert verdict(analyse(ds, s), ds, s, rows)[1] == "perplexity_small"
    assert verdict(analyse(ds, s), ds, s, None)[1] != "perplexity_small"


def test_stability_uses_random_starts_and_is_reproducible():
    ds = make_dataset(120, 2, 1.0, 0.25, 0, 7)
    s = Settings(perplexity=15, n_iter=200)
    a, b = stability(ds, s, seeds=(0, 1)), stability(ds, s, seeds=(0, 1))
    # Procrustes einer Einbettung gegen sich selbst ist 0 nur bis auf Rundung (CI: 2.2e-16)
    assert abs(a["procrustes"][0]) < 1e-9 and np.allclose(a["procrustes"], b["procrustes"], atol=1e-9) and len(a["embeddings"]) == 2
    assert not np.array_equal(a["embeddings"][0], a["embeddings"][1])


def test_out_of_sample_holds_out_the_last_fraction():
    ds = make_dataset(150, 2, 1.0, 0.25, 0, 7)
    oos = out_of_sample(ds, Settings(perplexity=15, n_iter=200))
    assert len(oos["test"]) == 30 and oos["test"][0] == 120 and oos["y_test"].shape == (30, 2)
    # Der genaue R² schwankt zwischen Plattformen (dieser Datensatz: lokal 0.44, CI 0.24; über 12 Datensätze lokal 0.36-0.69).
    # Geprüft wird nur, dass die Einbettung klar über Zufall liegt (Zufallseinbettungen: 99 %-Quantil 0.10).
    assert 0.0 <= oos["shift"] <= 1.0 and oos["r2_train"] > 0.15


def test_crowding_runs_both_kernels():
    ds = make_dataset(150, 3, 1.0, 0.25, 0, 7)
    out = crowding(ds, Settings(perplexity=15, n_iter=200))
    assert set(out) == {"student", "gauss"} and out["student"]["model"].kernel == "student" and out["gauss"]["model"].kernel == "gauss"


def test_timing_sweep_has_the_expected_shape_and_tsne_is_the_slowest():
    rows = timing_sweep(ns=(100, 200), n_iter=100)
    assert [r["n"] for r in rows] == [100, 200] and all(r[key] > 0 for r in rows for key in ("tsne", "isomap", "lle", "pca"))
    assert rows[1]["tsne"] > rows[1]["isomap"] and rows[1]["tsne"] > rows[1]["lle"]
