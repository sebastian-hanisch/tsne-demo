"""Auswertung: findet t-SNE die wahren Faktoren zurück - und was geht dabei verloren? Alle Kennzahlen werden am Datensatz gemessen; die wahren latenten Faktoren z sind bekannt
(Lieferrouten-Erzeugung, inkl. Sonderfahrten). t-SNE, Isomap, LLE (Vergleichsverfahren mit ihren guten Einstellungen) und PCA werden mit denselben Messungen bewertet.

- **R² der wahren Faktoren**: Rekonstruktion von z aus den zwei Koordinaten per quadratischer Regression (monotone Umparametrisierungen werden nicht bestraft).
- **Abstandstreue** (gesamt / nahe Paare / ferne Paare): Pearson-Korrelation der Paarabstände in der Einbettung mit den Paarabständen der wahren Faktoren; "nah" = untere Hälfte der
  wahren Paarabstände, "fern" = obere Hälfte. t-SNE erhält Nachbarschaften, keine Abstände zwischen fernen Regionen.
- **Trustworthiness** (Venna & Kaski): bleiben Nachbarn Nachbarn?
- **KL-Divergenz** KL(P‖Q): das eigene Optimierungsziel von t-SNE."""

import time
from dataclasses import dataclass

import numpy as np

import tsne_constants as C
from tsne_algorithm import embed_new_naive, fit_tsne, procrustes_disparity
from tsne_isomap import fit_isomap, pairwise_distances, standardize
from tsne_lle import fit_lle
from tsne_scenario import generate_dataset


def trustworthiness(X_high, X_low, n_neighbors=C.TRUST_NEIGHBORS):
    """Trustworthiness (Venna & Kaski, 2001): Anteil der Nachbarn im Einbettungsraum, die auch im Originalraum echte Nachbarn sind, mit
    Rang-Strafe für eingeschleppte Fremde. 1 = perfekt. Eigene Implementierung, gegen sklearn geprüft (nur im Test)."""
    n = len(X_high)
    k = n_neighbors
    d_high = np.linalg.norm(X_high[:, None, :] - X_high[None, :, :], axis=-1)
    d_low = np.linalg.norm(X_low[:, None, :] - X_low[None, :, :], axis=-1)
    np.fill_diagonal(d_high, np.inf)
    np.fill_diagonal(d_low, np.inf)
    ranks_high = np.argsort(np.argsort(d_high, axis=1), axis=1) + 1            # Rang 1 = nächster Nachbar
    neighbors_low = np.argsort(d_low, axis=1)[:, :k]
    penalty = 0.0
    for i in range(n):
        r = ranks_high[i, neighbors_low[i]]
        penalty += float(np.maximum(r - k, 0).sum())
    return 1.0 - 2.0 / (n * k * (2 * n - 3 * k - 1)) * penalty


def _quad_features(e):
    return np.column_stack([e[:, 0], e[:, 1], e[:, 0] ** 2, e[:, 0] * e[:, 1], e[:, 1] ** 2, np.ones(len(e))])


def r2_quadratic(coords2, z):
    """R² der Rekonstruktion von z aus zwei Koordinaten (quadratische Regression, Mittel über die Faktoren, gewichtet mit ihrer Varianz)."""
    A = _quad_features(coords2)
    beta, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(1.0 - (z - A @ beta).var(0).sum() / z.var(0).sum())


def pca_project(X, n_components=2):
    Z = standardize(X)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ vt[:n_components].T


def distance_fidelity(coords2, z):
    iu = np.triu_indices(len(z), 1)
    return float(np.corrcoef(pairwise_distances(coords2)[iu], pairwise_distances(z)[iu])[0, 1])


def distance_fidelity_split(coords2, z):
    """(nahe Paare, ferne Paare): Abstandstreue getrennt für die untere und die obere Hälfte der wahren Paarabstände."""
    iu = np.triu_indices(len(z), 1)
    dz = pairwise_distances(z)[iu]
    dc = pairwise_distances(coords2)[iu]
    near = dz <= np.median(dz)
    return float(np.corrcoef(dz[near], dc[near])[0, 1]), float(np.corrcoef(dz[~near], dc[~near])[0, 1])


def make_dataset(n_tours, q, curvature, noise, outlier_pct, seed):
    return generate_dataset(n_tours, q, curvature, noise, outlier_pct, seed)


def _lr(learning_rate):
    return None if not learning_rate else float(learning_rate)


def _metrics(coords, z, Z):
    near, far = distance_fidelity_split(coords, z)
    return {"r2": r2_quadratic(coords, z), "fid": distance_fidelity(coords, z), "near": near, "far": far, "trust": trustworthiness(Z, coords)}


@dataclass(frozen=True)
class Settings:
    perplexity: float = C.DEFAULT_PERPLEXITY
    n_iter: int = C.DEFAULT_N_ITER
    learning_rate: float = C.DEFAULT_LR
    exaggeration: float = C.DEFAULT_EXAGGERATION
    kernel: str = C.DEFAULT_KERNEL
    init: str = C.DEFAULT_INIT                       # "pca" oder "random:<Start>"


def run_tsne(X, s):
    kind, _, start = s.init.partition(":")
    return fit_tsne(X, s.perplexity, s.n_iter, _lr(s.learning_rate), s.exaggeration, s.kernel, kind, int(start) if start else 0)


@dataclass(frozen=True)
class Analysis:
    tsne: object
    alt: object                      # bei Gauß-Kern: derselbe Lauf mit Student-t-Kern (Crowding-Vergleich), sonst None
    ref: object                      # bei großer Lernrate: derselbe Lauf mit automatischer Lernrate (Referenz für 'Lernrate zu hoch'), sonst None
    isomap: object
    lle: object
    iso_2d: np.ndarray
    pca_2d: np.ndarray
    iso_indices: np.ndarray
    metrics: dict                    # {"tsne", "isomap", "lle", "pca", "alt"?} -> {"r2","fid","near","far","trust"}
    snapshot_r2: dict                # Iteration -> R² der Einbettung zu diesem Zeitpunkt


LR_REFERENCE_FROM = 200.0        # ab dieser Lernrate rechnet die Analyse einen Referenzlauf mit automatischer Lernrate daneben
LR_HIGH_KL_FACTOR = 1.5          # 'Lernrate zu hoch': die KL-Divergenz endet mehr als 1.5-fach über der des Referenzlaufs


def analyse(dataset, settings):
    Z = standardize(dataset.X)
    model = run_tsne(dataset.X, settings)
    iso = fit_isomap(dataset.X, C.ISOMAP_K, 2)
    lle = fit_lle(dataset.X, C.LLE_K, 2, C.LLE_REG)
    pca2 = pca_project(dataset.X)
    iso2 = iso.embedding[:, :2]
    metrics = {
        "tsne": _metrics(model.embedding, dataset.z, Z),
        "isomap": _metrics(iso2, dataset.z[iso.indices], Z[iso.indices]),
        "lle": _metrics(lle.embedding[:, :2], dataset.z, Z),
        "pca": _metrics(pca2, dataset.z, Z),
    }
    alt = None
    if settings.kernel == "gauss":
        alt = run_tsne(dataset.X, Settings(**{**settings.__dict__, "kernel": "student"}))
        metrics["alt"] = _metrics(alt.embedding, dataset.z, Z)
        metrics["alt"]["kl"] = alt.kl
    ref = None
    if settings.learning_rate >= LR_REFERENCE_FROM:
        ref = run_tsne(dataset.X, Settings(**{**settings.__dict__, "learning_rate": 0.0}))
        metrics["ref"] = _metrics(ref.embedding, dataset.z, Z)
        metrics["ref"]["kl"] = ref.kl
    snap = {it: r2_quadratic(y, dataset.z) for it, y in model.snapshots.items() if it > 0}
    return Analysis(tsne=model, alt=alt, ref=ref, isomap=iso, lle=lle, iso_2d=iso2, pca_2d=pca2, iso_indices=iso.indices, metrics=metrics, snapshot_r2=snap)


PERPLEXITY_SMALL_BELOW = 10      # darunter: kleine Perplexität (im Sweep ist die Kurve ab 10 flach bis zackig, davor bricht sie ein)
PERPLEXITY_SMALL_MARGIN = 0.10


def perplexity_status(rows, perplexity):
    """-> "small", wenn die Perplexität unter 10 liegt und das R² der Sweep-Kurve an dieser Stelle deutlich unter dem Median der Perplexitäten ab 10 liegt, sonst None.
    Ein "zu große Perplexität" gibt es hier bewusst nicht: auf diesen Daten ist die Kurve für große Perplexitäten nicht verlässlich schlechter (siehe README)."""
    reference = float(np.median([r["r2"] for r in rows if r["perplexity"] >= PERPLEXITY_SMALL_BELOW]))
    cur = min(rows, key=lambda r: abs(np.log(r["perplexity"]) - np.log(max(perplexity, 1e-9))))
    if perplexity < PERPLEXITY_SMALL_BELOW and reference - cur["r2"] >= PERPLEXITY_SMALL_MARGIN:
        return "small"
    return None


def verdict(analysis, dataset, settings, sweep_rows=None):
    """Verdict-Kaskade (Warnungen zuerst) -> (Stufe, Code, Daten)."""
    m = analysis.metrics
    model = analysis.tsne
    kl = model.kl_history
    late = kl[int(0.8 * len(kl))]
    data = {"r2": m["tsne"]["r2"], "r2_iso": m["isomap"]["r2"], "r2_lle": m["lle"]["r2"], "r2_pca": m["pca"]["r2"], "far": m["tsne"]["far"], "far_pca": m["pca"]["far"],
            "far_iso": m["isomap"]["far"], "trust": m["tsne"]["trust"], "kl": model.kl, "perplexity": settings.perplexity, "n_iter": settings.n_iter,
            "outlier_pct": dataset.outlier_pct, "improvement": float((late - kl[-1]) / max(kl[-1], 1e-12)),
            }
    if "ref" in m:
        data.update({"kl_ref": m["ref"]["kl"], "trust_ref": m["ref"]["trust"], "r2_ref": m["ref"]["r2"]})
    if "alt" in m:
        data.update({"trust_alt": m["alt"]["trust"], "kl_alt": m["alt"]["kl"], "r2_alt": m["alt"]["r2"]})
    data["diverged_at"] = model.diverged_at
    if model.diverged_at:
        return "warning", "diverged", data
    if "ref" in m and model.kl > LR_HIGH_KL_FACTOR * m["ref"]["kl"]:
        return "warning", "lr_high", data
    if data["improvement"] > 0.03 and settings.n_iter < 500:
        return "warning", "not_converged", data
    if "alt" in m and m["alt"]["trust"] - m["tsne"]["trust"] >= 0.04:
        return "warning", "crowding", data
    if dataset.outlier_pct > 0 and m["pca"]["far"] - m["tsne"]["far"] >= 0.25 and m["pca"]["r2"] - m["tsne"]["r2"] >= 0.15:
        return "warning", "global_structure", data
    if sweep_rows is not None and perplexity_status(sweep_rows, settings.perplexity) == "small":
        return "warning", "perplexity_small", data
    if dataset.curvature == 0 and m["tsne"]["r2"] - m["pca"]["r2"] < 0.03:
        return "info", "no_advantage", data
    if m["tsne"]["r2"] - m["pca"]["r2"] >= 0.10:
        return "success", "tsne_wins", data
    return "info", "neutral", data


def perplexity_sweep(q, curvature, noise, outlier_pct, n_tours=C.SWEEP_N_TOURS, n_iter=C.SWEEP_N_ITER, perplexities=C.SWEEP_PERPLEXITIES, seeds=C.SWEEP_SEEDS):
    """Feste Sweep-Seeds, Standard-Einstellungen für Lernrate/Exaggeration/Kern/Init: je Perplexität mittleres R², Abstandstreue (ferne Paare), Trustworthiness, KL."""
    rows = []
    for perp in perplexities:
        acc = {"r2": [], "far": [], "trust": [], "kl": []}
        for seed in seeds:
            ds = make_dataset(n_tours, q, curvature, noise, outlier_pct, seed)
            model = fit_tsne(ds.X, perp, n_iter)
            m = _metrics(model.embedding, ds.z, standardize(ds.X))
            acc["r2"].append(m["r2"]), acc["far"].append(m["far"]), acc["trust"].append(m["trust"]), acc["kl"].append(model.kl)
        rows.append({"perplexity": int(perp), **{k: float(np.mean(v)) for k, v in acc.items()}})
    return rows


def convergence_rows(analysis):
    """(Iteration, R²) an den Schnappschüssen des aktuellen Laufs - ohne Extra-Rechnung."""
    return sorted(analysis.snapshot_r2.items())


def stability(dataset, settings, seeds=C.STABILITY_SEEDS):
    """Derselbe Datensatz, dieselben Einstellungen, verschiedene zufällige Start-Layouts: -> Einbettungen, R², KL, Procrustes-Abstände zum ersten Lauf."""
    runs = [run_tsne(dataset.X, Settings(**{**settings.__dict__, "init": f"random:{s}"})) for s in seeds]
    return {"seeds": list(seeds), "embeddings": [r.embedding for r in runs], "kl": [r.kl for r in runs],
            "r2": [r2_quadratic(r.embedding, dataset.z) for r in runs],
            "procrustes": [procrustes_disparity(r.embedding, runs[0].embedding) for r in runs]}


def out_of_sample(dataset, settings, fraction=C.HOLDOUT_FRACTION, k=C.OOS_K):
    """Die letzten `fraction` der Touren zurückhalten. t-SNE hat keine Abbildung: (a) Näherung `embed_new_naive`, (b) Neuberechnung mit allen Touren -
    wie stark verschieben sich dabei die bereits eingebetteten Trainings-Touren (Procrustes)?"""
    n = dataset.n
    n_test = max(10, int(round(fraction * n)))
    train, test = np.arange(n - n_test), np.arange(n - n_test, n)
    model = run_tsne(dataset.X[train], settings)
    y_test = embed_new_naive(model, dataset.X[test], k)
    beta, *_ = np.linalg.lstsq(_quad_features(model.embedding), dataset.z[train], rcond=None)
    z_test = dataset.z[test]
    resid = z_test - _quad_features(y_test) @ beta                                                     # nicht zentrieren: ein konstanter Versatz der Vorhersage zählt als Fehler
    r2_test = float(1 - (resid ** 2).sum() / ((z_test - z_test.mean(0)) ** 2).sum())
    full = run_tsne(dataset.X, settings)
    return {"train": train, "test": test, "model": model, "y_test": y_test, "r2_test": r2_test, "r2_train": r2_quadratic(model.embedding, dataset.z[train]),
            "shift": procrustes_disparity(full.embedding[train], model.embedding)}


def crowding(dataset, settings):
    """Student-t- gegen Gauß-Kern bei sonst gleichen Einstellungen -> beide Läufe mit Kennzahlen."""
    Z = standardize(dataset.X)
    out = {}
    for kernel in C.KERNELS:
        model = run_tsne(dataset.X, Settings(**{**settings.__dict__, "kernel": kernel}))
        out[kernel] = {"model": model, **_metrics(model.embedding, dataset.z, Z), "kl": model.kl}
    return out


def timing_sweep(ns=C.TIMING_NS, n_iter=C.TIMING_N_ITER, seed=100_000):
    """Gemessene Rechenzeit (Sekunden) von t-SNE, Isomap, LLE und PCA für wachsende n (eigene Messung, Rechner-abhängig)."""
    rows = []
    for n in ns:
        ds = generate_dataset(n, 2, 0.0, 0.1, 0, seed)
        out = {"n": int(n)}
        for name, fn in (("tsne", lambda: fit_tsne(ds.X, C.DEFAULT_PERPLEXITY, n_iter)), ("isomap", lambda: fit_isomap(ds.X, C.ISOMAP_K, 2)),
                         ("lle", lambda: fit_lle(ds.X, C.LLE_K, 2, C.LLE_REG)), ("pca", lambda: pca_project(ds.X))):
            t = time.perf_counter()
            fn()
            out[name] = time.perf_counter() - t
        rows.append(out)
    return rows
