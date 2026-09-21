"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Schritt-Zustand, Zusatz-Experimente, Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import tsne_constants as C
from tsne_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None, timeout=300):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def test_default_renders_without_exception():
    at = _run()
    assert any("Perplexity" in h.value for h in at.subheader)
    assert not at.error


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    _run(lambda a: _apply(a, C.PRESETS[name]))


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MIN
        at.session_state["perplexity_slider"] = 98                    # n − 2: größte zulässige Perplexity
        at.session_state["q_slider"] = C.Q_MIN
        at.session_state["n_iter_slider"] = C.N_ITER_MIN
    _run(small)

    def perplexity_follows_n(at):
        at.session_state["perplexity_slider"] = 140
        at.session_state["n_tours_slider"] = C.N_TOURS_MIN            # Perplexity 140 > n − 2 -> wird geklemmt
    at = _run(perplexity_follows_n)
    assert at.session_state["perplexity_slider"] == 98

    def large(at):
        at.session_state["n_tours_slider"] = 400
        at.session_state["q_slider"] = C.Q_MAX
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["outlier_slider"] = C.OUTLIER_PCT_MAX
        at.session_state["kernel_select"] = "gauss"
        at.session_state["init_select"] = "random:3"
        at.session_state["lr_select"] = 2000.0
        at.session_state["exaggeration_slider"] = C.EXAGGERATION_MAX
        at.session_state["n_iter_slider"] = 300
    _run(large)


def test_step_state_resets_when_the_data_or_settings_change_and_survives_reruns():
    at = _run()
    at.session_state["tsne_step"] = 3
    at.run()
    assert not at.exception and at.session_state["tsne_step"] == 3
    at.session_state["perplexity_slider"] = 10
    at.run()
    assert not at.exception and at.session_state["tsne_step"] == 1


@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_every_step_renders(step):
    def setup(at):
        at.session_state["tsne_step"] = step
    _run(setup)


def test_snapshot_slider_survives_a_change_of_snapshot():
    at = _run(lambda a: a.session_state.__setitem__("tsne_step", 3))
    at.session_state["tsne_snap"] = 0
    at.run()
    assert not at.exception and at.session_state["tsne_snap"] == 0


def test_extra_experiments_run_on_demand():
    at = _run()
    for key in ("stability_start", "oos_start", "crowding_start"):
        [b for b in at.button if b.key == key][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["stability_on"] and at.session_state["oos_on"] and at.session_state["crowding_on"]


def test_gauss_kernel_shows_the_crowding_comparison_without_a_button():
    at = _run(lambda a: _apply(a, C.PRESETS["Gauß-Kern: Crowding"]))
    assert not [b for b in at.button if b.key == "crowding_start"]
    assert any(t.value.startswith("Mit q = 3") or "Mit q = 3" in t.value for t in at.caption)


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "tsne_visualization.py").read_text(encoding="utf-8")
    assert len(re.findall(r"return lock_axes\(fig\)", source)) == len(re.findall(r"^def build_", source, flags=re.M)) == 11
    assert len(re.findall(r"^\s+return fig$", source, flags=re.M)) == 1


def test_play_runs_through_all_frames_without_duplicate_chart_keys():
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    at = _run()
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
