"""Orakel-Test für das Out-of-sample-R²: `sklearn.metrics.r2_score` (varianzgewichtet) statt der eigenen Formel. Das alte Residuen-Varianz-Maß
(`resid.var(0)`) zentrierte die Residuen und verzieh damit einen konstanten Versatz der Vorhersage; richtig ist R² = 1 − SSE/SST ohne Zentrierung."""

import numpy as np
import pytest

from tsne_evaluation import Settings, make_dataset, out_of_sample

metrics = pytest.importorskip("sklearn.metrics")


def _quad(e):
    return np.column_stack([e[:, 0], e[:, 1], e[:, 0] ** 2, e[:, 0] * e[:, 1], e[:, 1] ** 2, np.ones(len(e))])


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_out_of_sample_r2_equals_sklearn_r2_score_and_penalises_offsets(seed):
    ds = make_dataset(90, 2, 1.0, 0.25, 0, seed)
    out = out_of_sample(ds, Settings(perplexity=10, n_iter=120))
    beta = np.linalg.pinv(_quad(out["model"].embedding)) @ ds.z[out["train"]]
    z_test, pred = ds.z[out["test"]], _quad(out["y_test"]) @ beta
    assert out["r2_test"] == pytest.approx(metrics.r2_score(z_test, pred, multioutput="variance_weighted"), abs=1e-6)
    # Der Versatz der Vorhersage ist hier nicht null: das alte, zentrierte Maß hätte ihn verziehen und wäre größer gewesen.
    old = 1 - (z_test - pred).var(0).sum() / z_test.var(0).sum()
    assert np.abs((z_test - pred).mean(0)).max() > 1e-3 and out["r2_test"] < old - 1e-6
