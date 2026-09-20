"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert)."""

import pytest

import tsne_constants as C
from tsne_evaluation import Settings, analyse, make_dataset, perplexity_sweep, verdict

_SWEEPS = {}


def _settings(p):
    return Settings(perplexity=float(p["perplexity"]), n_iter=p["n_iter"], learning_rate=float(p["learning_rate"]), exaggeration=float(p["exaggeration"]), kernel=p["kernel"], init=p["init"])


def _measure(p):
    dataset = make_dataset(p["n_tours"], p["q"], p["curvature"], p["noise"], p["outlier_pct"], p["seed"])
    s = _settings(p)
    a = analyse(dataset, s)
    key = (p["q"], p["curvature"], p["noise"], p["outlier_pct"])
    if key not in _SWEEPS:
        _SWEEPS[key] = perplexity_sweep(*key)
    code, data = verdict(a, dataset, s, _SWEEPS[key])[1:]
    return {"verdict": code, "r2": data["r2"], "r2_iso": data["r2_iso"], "r2_pca": data["r2_pca"], "far": data["far"], "far_iso": data["far_iso"], "far_pca": data["far_pca"],
            "trust": data["trust"], "kl": data["kl"], "trust_alt": data.get("trust_alt"), "kl_alt": data.get("kl_alt")}


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_TOURS_MIN <= p["n_tours"] <= C.N_TOURS_MAX and C.Q_MIN <= p["q"] <= C.Q_MAX
        assert C.CURVATURE_MIN <= p["curvature"] <= C.CURVATURE_MAX and C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX and C.OUTLIER_PCT_MIN <= p["outlier_pct"] <= C.OUTLIER_PCT_MAX
        assert C.PERPLEXITY_MIN <= p["perplexity"] <= min(C.PERPLEXITY_MAX, p["n_tours"] - 2) and C.N_ITER_MIN <= p["n_iter"] <= C.N_ITER_MAX
        assert p["learning_rate"] in C.LR_CHOICES and C.EXAGGERATION_MIN <= p["exaggeration"] <= C.EXAGGERATION_MAX and p["kernel"] in C.KERNELS and p["init"] in C.INITS


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if isinstance(expected, str):
            assert value == expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
