"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe km_presets.py in kmeans-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import tsne_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "n_tours_slider": SettingSpec("n", int, C.DEFAULT_N_TOURS, C.N_TOURS_MIN, C.N_TOURS_MAX),
    "q_slider": SettingSpec("q", int, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "curvature_slider": SettingSpec("curv", float, C.DEFAULT_CURVATURE, C.CURVATURE_MIN, C.CURVATURE_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "outlier_slider": SettingSpec("out", int, C.DEFAULT_OUTLIER_PCT, C.OUTLIER_PCT_MIN, C.OUTLIER_PCT_MAX),
    "perplexity_slider": SettingSpec("perp", int, C.DEFAULT_PERPLEXITY, C.PERPLEXITY_MIN, C.PERPLEXITY_MAX),
    "n_iter_slider": SettingSpec("it", int, C.DEFAULT_N_ITER, C.N_ITER_MIN, C.N_ITER_MAX),
    "lr_select": SettingSpec("lr", float, C.DEFAULT_LR, min(C.LR_CHOICES), max(C.LR_CHOICES)),
    "exaggeration_slider": SettingSpec("ex", int, C.DEFAULT_EXAGGERATION, C.EXAGGERATION_MIN, C.EXAGGERATION_MAX),
    "kernel_select": SettingSpec("kern", _choice(C.KERNELS), C.DEFAULT_KERNEL),
    "init_select": SettingSpec("init", _choice(C.INITS), C.DEFAULT_INIT),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"n_tours": "n_tours_slider", "q": "q_slider", "curvature": "curvature_slider", "noise": "noise_slider", "outlier_pct": "outlier_slider",
               "perplexity": "perplexity_slider", "n_iter": "n_iter_slider", "learning_rate": "lr_select", "exaggeration": "exaggeration_slider", "kernel": "kernel_select",
               "init": "init_select", "seed": "seed_input"}


def snap_lr(value):
    return min(C.LR_CHOICES, key=lambda choice: abs(choice - value))


def perplexity_max(n_tours):
    """Obere Grenze der Perplexität: höchstens n − 2 (die Binärsuche braucht mehr Touren als Perplexität)."""
    return int(min(C.PERPLEXITY_MAX, n_tours - 2))


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["lr_select"] = snap_lr(st.session_state.get("lr_select", C.DEFAULT_LR))
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
