"""t-SNE an Lieferrouten-Kennzahlen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - t-SNE - und lässt
stattdessen das Beispiel wachsen. Viertes Stück der Dimensionsreduktion-Linie der "Konzepte"-Reihe: t-SNE behebt die Linearitätsschwäche der PCA
(pca-demo) über lokale probabilistische Nachbarschaftserhaltung - und hat dafür eigene Schwächen (keine globale Struktur, kein Out-of-sample,
Perplexity- und Optimierungs-Empfindlichkeit, O(n²)). Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import tsne_constants as C
from tsne_evaluation import (
    Settings, analyse, convergence_rows, crowding, make_dataset, out_of_sample, perplexity_sweep, stability, timing_sweep, verdict,
)
from tsne_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    perplexity_max,
    randomize_seed,
    sync_query_params,
)
from tsne_visualization import (
    build_affinity_bars,
    build_crowding,
    build_distance_fidelity,
    build_embedding,
    build_kernel_curves,
    build_kl_curve,
    build_neighbour_view,
    build_out_of_sample,
    build_perplexity_sweep,
    build_stability,
    build_timing,
)

st.set_page_config(page_title="t-SNE – Sebastian Hanisch", layout="wide")

STEP_LABELS = {
    1: "1 · Nachbarschaft (Gauß)",
    2: "2 · Zielraum-Kern (Student-t)",
    3: "3 · Optimierung",
    4: "4 · Ergebnis",
}


def _lr_label(value):
    return "auto" if value == 0 else f"{value:g}"


@st.cache_data(show_spinner=False)
def _dataset(n_tours, q, curvature, noise, outlier_pct, seed):
    return make_dataset(n_tours, q, curvature, noise, outlier_pct, seed)


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse(make_dataset(*data_params), settings)


@st.cache_data(show_spinner=False)
def _sweep(q, curvature, noise, outlier_pct):
    return perplexity_sweep(q, curvature, noise, outlier_pct)


@st.cache_data(show_spinner=False)
def _stability(data_params, settings):
    return stability(make_dataset(*data_params), settings)


@st.cache_data(show_spinner=False)
def _oos(data_params, settings):
    return out_of_sample(make_dataset(*data_params), settings)


@st.cache_data(show_spinner=False)
def _crowding(data_params, settings):
    return crowding(make_dataset(*data_params), settings)


st.title("🌌 t-SNE an Lieferrouten-Kennzahlen")
st.markdown(
    """
Dieselben **12 Kennzahlen je Lieferroute** wie in der PCA-, Isomap- und LLE-Demo - erzeugt aus wenigen versteckten Faktoren, aber mit **gekrümmter** Struktur, an der die PCA scheiterte.
**t-SNE** (t-distributed Stochastic Neighbor Embedding) geht einen dritten Weg: Es fragt für jede Tour, **wie wahrscheinlich sie jede andere als Nachbarn wählen würde** (eine Gauß-Verteilung,
deren Breite die **Perplexity** festlegt), und sucht dann 2-D-Punkte, deren Nachbarschafts-Wahrscheinlichkeiten (mit **schweren Rändern**, Student-t) möglichst gut dazu passen. Gemessen wird der
Unterschied mit der KL-Divergenz, minimiert per Gradientenabstieg. Was dabei gewonnen wird - und was verloren geht: **globale Abstände**, **neue Touren**, Stabilität, Rechenzeit - zeigt die Demo
direkt gegen PCA, Isomap und LLE. Wie das Verfahren funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - viertes Stück der "
    "Dimensionsreduktion-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel: t-SNE behebt die Linearitätsschwäche der PCA über lokale Nachbarschafts-Wahrscheinlichkeiten - "
    "ein dritter Ansatz neben Isomap (geodätisch) und LLE (Rekonstruktion), mit eigenen Schwächen und dem Startpunkt der Kette t-SNE → UMAP → PaCMAP."
)

with st.expander("So funktioniert t-SNE", expanded=True):
    st.markdown(
        """
t-SNE (van der Maaten & Hinton, 2008) besteht aus drei Schritten:

1. **Nachbarschafts-Wahrscheinlichkeiten**: für jede Tour eine Gauß-Verteilung über alle anderen Touren - nah heißt hohe Wahrscheinlichkeit. Die Breite wird so gewählt, dass die **Perplexity**
   (die "effektive Zahl der Nachbarn") einen Zielwert trifft; danach werden beide Richtungen zu einer gemeinsamen Verteilung *P* symmetrisiert.
2. **Zielraum mit schweren Rändern**: im 2-D-Bild gilt dieselbe Idee mit einem **Student-t-Kern** statt der Gauß-Glocke. Weil er langsamer abfällt, dürfen sich mittlere Abstände im Bild
   weiter voneinander entfernen als im Original - das löst das **Crowding-Problem** (zu wenig Platz für viele mittlere Abstände in nur 2 Dimensionen).
3. **Optimierung**: die Punkte werden per Gradientenabstieg verschoben, bis die Verteilung *Q* des Bildes zu *P* passt (KL-Divergenz). Eine **Early-Exaggeration**-Phase (*P* ×12) bildet zuerst
   grobe Gruppen; ein zu großer Schritt (**Lernrate**) oder zu wenige **Iterationen** verderben das Ergebnis.

Was t-SNE **nicht** tut: Abstände erhalten (nur Nachbarschaften - ferne Regionen und Extremwerte werden gestaucht), neue Touren einbetten (es gibt keine Abbildung, nur eine Neuberechnung) und
für jeden Start dasselbe Bild liefern (die Optimierung hat lokale Minima). Die Regler links zeigen jede dieser Eigenschaften live.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider("Anzahl Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=50)
    q = st.slider(
        "Wahre Anzahl versteckter Faktoren (q)", *bounds("q_slider"), key="q_slider",
        help="So viele echte Einflussgrößen erzeugen die 12 Kennzahlen. Mit mehr Faktoren als Zielraum-Dimensionen (2) wird das Crowding-Problem spürbar - und die Stichprobe dünner.",
    )
    curvature = st.slider(
        "Krümmung", *bounds("curvature_slider"), key="curvature_slider", step=0.05,
        help="0 = die Kennzahlen hängen linear von den Faktoren ab (dann hat t-SNE keinen Vorteil vor der PCA). Größer = die Touren liegen auf einer zunehmend gebogenen Fläche.",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen je Kennzahl.",
    )
    outlier_pct = st.slider(
        "Sonderfahrten (%)", *bounds("outlier_slider"), key="outlier_slider",
        help="Anteil der Touren mit extremem Zeitdruck-Faktor (zehnfach vergrößert). PCA und Isomap behalten die Größenordnung - t-SNE staucht die Extreme.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**t-SNE**")
    p_max = perplexity_max(int(n_tours))
    if st.session_state["perplexity_slider"] > p_max:
        st.session_state["perplexity_slider"] = p_max
    perplexity = st.slider(
        "Perplexity", C.PERPLEXITY_MIN, p_max, key="perplexity_slider",
        help="Die 'effektive Zahl der Nachbarn' je Tour. Zu klein: nur allerengste Nachbarn zählen, die Einbettung zerfällt in Fragmente. Die obere Grenze folgt der Tourenzahl.",
    )
    n_iter = st.slider(
        "Iterationen", *bounds("n_iter_slider"), key="n_iter_slider", step=50,
        help="Schritte des Gradientenabstiegs. Zu wenige: die Optimierung ist noch nicht konvergiert (die KL-Kurve unten fällt noch).",
    )
    learning_rate = st.select_slider(
        "Lernrate", options=C.LR_CHOICES, key="lr_select", format_func=_lr_label,
        help="Schrittweite des Gradientenabstiegs. auto = max(n / Exaggeration / 4, 50) wie in scikit-learn. Zu hoch: die Schritte überschießen, die Optimierung endet bei deutlich höherer KL-Divergenz (die Demo rechnet ab 200 einen Referenzlauf mit auto daneben).",
    )
    exaggeration = st.slider(
        "Early Exaggeration", *bounds("exaggeration_slider"), key="exaggeration_slider",
        help="Faktor auf P in der ersten Phase (ein Viertel der Iterationen, höchstens 250). Bei 1 statt 12 fiel im Test (Seed 7, 300 Touren) die Abstandstreue ferner Paare von 0.69 auf 0.33.",
    )
    kernel = st.radio(
        "Zielraum-Kern", C.KERNELS, key="kernel_select", format_func=lambda k: C.KERNEL_LABELS[k],
        help="Student-t ist t-SNE. Der Gauß-Kern ist das ursprüngliche (symmetrische) SNE: das Crowding-Problem tritt auf, sobald es mehr versteckte Faktoren als 2 gibt.",
    )
    init = st.selectbox(
        "Initialisierung", C.INITS, key="init_select", format_func=lambda i: C.INIT_LABELS[i],
        help="Startlayout. PCA ist deterministisch (Standard). Zufällige Starts zeigen, wie stark das Ergebnis vom Start abhängt (siehe Stabilität unten).",
    )

    st.button("🎲 Neue Touren generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren.")

sync_query_params({
    "n_tours_slider": int(n_tours), "q_slider": int(q), "curvature_slider": curvature, "noise_slider": noise, "outlier_slider": int(outlier_pct), "perplexity_slider": int(perplexity),
    "n_iter_slider": int(n_iter), "lr_select": learning_rate, "exaggeration_slider": int(exaggeration), "kernel_select": kernel, "init_select": init, "seed_input": int(seed),
})

data_params = (int(n_tours), int(q), float(curvature), float(noise), int(outlier_pct), int(seed))
settings = Settings(perplexity=float(perplexity), n_iter=int(n_iter), learning_rate=float(learning_rate), exaggeration=float(exaggeration), kernel=kernel, init=init)
with st.spinner("Berechne Nachbarschafts-Wahrscheinlichkeiten und optimiere die Einbettung..."):
    dataset = _dataset(*data_params)
    analysis = _analysis(data_params, settings)
model = analysis.tsne
metrics = analysis.metrics
z_color = dataset.z[:, 0]
iso_idx = analysis.iso_indices
data_key = data_params + (settings,)
snap_iters = sorted(model.snapshots)
sweep_rows = _sweep(int(q), float(curvature), float(noise), int(outlier_pct))
level, code, vd = verdict(analysis, dataset, settings, sweep_rows)

# --- t-SNE in Aktion -----------------------------------------------------------------------------------------------------

st.markdown("## 🎯 t-SNE in Aktion")
st.caption(
    "Die 2-D-Ansicht in Schritt 1 zeigt die Touren in den ersten beiden Hauptkomponenten (Farbe = versteckter Faktor 1) - nur als Zeichenfläche; t-SNE selbst rechnet in allen 12 Dimensionen."
)
if "tsne_step" not in st.session_state or st.session_state.get("tsne_step_owner") != data_key:
    st.session_state["tsne_step"] = 1
    st.session_state["tsne_snap"] = snap_iters[-1]
    st.session_state["tsne_step_owner"] = data_key
step_col, play_col = st.columns([5, 1])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="tsne_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

view = analysis.pca_2d
centre = view.mean(0)
focus = int(np.argmin(((view - centre) ** 2).sum(1)))
p_row = model.cond[focus]
order = np.argsort(-p_row)
top = order[:40]
if st.session_state.get("tsne_snap") not in model.snapshots:
    st.session_state["tsne_snap"] = snap_iters[-1]
if step == 3 and not auto_play:
    snap_it = st.select_slider("Iteration", options=snap_iters, key="tsne_snap", format_func=lambda i: "Start (PCA/Zufall)" if i == 0 else f"Iteration {i}")
else:
    snap_it = st.session_state.get("tsne_snap", snap_iters[-1])
    if snap_it not in model.snapshots:
        snap_it = snap_iters[-1]
r2_points = convergence_rows(analysis)

view_slot = st.empty()


def _render(current_step, iteration=None):
    if current_step == 1:
        with view_slot.container():
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(build_neighbour_view(view, z_color, focus, top, p_row[top]), width="stretch", key="tsne_view_1")
            c2.markdown("**Nachbarschafts-Wahrscheinlichkeiten dieser Tour**")
            c2.plotly_chart(build_affinity_bars(p_row[order], model.perplexity, float(model.sigma[focus])), width="stretch", key="tsne_affinity_bars")
    elif current_step == 2:
        with view_slot.container():
            c1, c2 = st.columns(2)
            c1.markdown("**Kern im Zielraum**")
            c1.plotly_chart(build_kernel_curves(model.kernel), width="stretch", key="tsne_kernel_curves")
            c2.markdown("**Startlayout (Iteration 0)**")
            c2.plotly_chart(build_embedding(model.snapshots[0], z_color, "Koordinate 1", "Koordinate 2"), width="stretch", key="tsne_start_layout")
    elif current_step == 3:
        it = snap_it if iteration is None else iteration
        with view_slot.container():
            c1, c2 = st.columns([2, 3])
            c1.markdown(f"**Einbettung nach Iteration {it}**")
            c1.plotly_chart(build_embedding(model.snapshots[it], z_color, "Koordinate 1", "Koordinate 2"), width="stretch", key="tsne_snapshot")
            c2.markdown("**Optimierung**")
            c2.plotly_chart(build_kl_curve(model.kl_history, model.exaggeration_iters, marker=max(it, 1), r2_points=r2_points), width="stretch", key="tsne_kl_curve")
    else:
        with view_slot.container():
            c1, c2 = st.columns(2)
            c1.markdown("**t-SNE: Einbettung**")
            c1.plotly_chart(build_embedding(model.embedding, z_color, "t-SNE-Koordinate 1", "t-SNE-Koordinate 2"), width="stretch", key="tsne_embed_step")
            c2.markdown("**Zum Vergleich: PCA**")
            c2.plotly_chart(build_embedding(analysis.pca_2d, z_color, "PC1", "PC2"), width="stretch", key="pca_embed_step")


if auto_play:
    for s in STEP_LABELS:
        if s == 3:
            for it in snap_iters:
                _render(3, it)
                time.sleep(0.35)
        else:
            _render(s)
            time.sleep(1.0)
    step = 4
else:
    _render(step)

if step == 1:
    st.caption(
        f"Die gewählte Tour (Stern, nahe der Mitte): die Gauß-Breite σ = {model.sigma[focus]:.2f} (in z-Einheiten) ist so gewählt, dass ihre Nachbarschaft **{model.perplexity:g} effektive Nachbarn** hat. "
        f"Die nächsten 10 Nachbarn tragen zusammen {p_row[order[:10]].sum() * 100:.0f} % der Wahrscheinlichkeit, die nächsten 40 {p_row[top].sum() * 100:.0f} %. Ferne Touren spielen praktisch keine Rolle."
    )
elif step == 2:
    st.caption(
        "Im Zielraum wird die Ähnlichkeit zweier Punkte aus ihrem Abstand berechnet. Die Student-t-Kurve (orange) fällt viel langsamer als die Gauß-Glocke (blau) - bei Abstand 4 liegt sie noch bei 6 % "
        "statt praktisch 0. Deshalb dürfen mittlere Abstände im Bild groß werden, ohne dass es 'teuer' wird: das ist die Antwort auf das Crowding-Problem. "
        f"Aktuell gewählt: **{C.KERNEL_LABELS[model.kernel]}**. Das Startlayout ({C.INIT_LABELS[settings.init]}) liegt extrem eng beieinander (Streuung 1e-4) - die Optimierung entfaltet es."
    )
elif step == 3:
    st.caption(
        f"Grau hinterlegt: die **Early-Exaggeration**-Phase ({model.exaggeration_iters} Iterationen, P × {model.exaggeration:g}) - sie bildet grobe Gruppen, bevor die feine Anordnung folgt. "
        f"Lernrate {model.learning_rate:g}, Momentum 0.5 → 0.8. KL-Divergenz am Ende: {model.kl:.2f}. Fällt die Kurve am Ende noch, ist die Optimierung nicht konvergiert."
    )
else:
    st.caption("Farbe = versteckter Faktor 1. Verläuft sie in der t-SNE-Einbettung glatt und ohne Überlappung, hat t-SNE die Fläche entrollt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was t-SNE gefunden hat - und die anderen drei Verfahren auf denselben Daten")
st.caption(
    f"Vergleichsverfahren mit ihren guten Einstellungen (aus den Demos davor): Isomap k = {C.ISOMAP_K}, LLE k = {C.LLE_K}, Regularisierung {C.LLE_REG:g}, PCA auf z-Werten."
)
if len(iso_idx) < dataset.n:
    st.warning(f"⚠️ Der Isomap-Graph ist nicht zusammenhängend: nur {len(iso_idx)} von {dataset.n} Touren sind in der Isomap-Einbettung enthalten (siehe Isomap-Demo).")
m1, m2, m3, m4 = st.columns(4)
m1.metric("R² der wahren Faktoren", f"{metrics['tsne']['r2']:.2f}", delta=f"{metrics['tsne']['r2'] - metrics['isomap']['r2']:+.2f} ggü. Isomap", delta_color="normal",
          help="Wie gut lassen sich die versteckten Faktoren aus den zwei Koordinaten zurückgewinnen (quadratische Regression). Isomap mit derselben Messung im Delta.")
m2.metric("Abstandstreue ferner Paare", f"{metrics['tsne']['far']:.2f}", delta=f"{metrics['tsne']['far'] - metrics['isomap']['far']:+.2f} ggü. Isomap", delta_color="normal",
          help="Korrelation der Paarabstände in der Einbettung mit den Paarabständen der wahren Faktoren, nur für die obere Hälfte der wahren Abstände. t-SNE erhält Nachbarschaften, keine großen Abstände.")
m3.metric("Trustworthiness", f"{metrics['tsne']['trust']:.2f}", delta=f"{metrics['tsne']['trust'] - metrics['isomap']['trust']:+.2f} ggü. Isomap", delta_color="normal",
          help=f"Nachbarschaft erhalten: Anteil der Nachbarn in der 2-D-Einbettung, die auch im Originalraum Nachbarn sind (k = {C.TRUST_NEIGHBORS}); 1 = perfekt.")
m4.metric("KL-Divergenz", f"{model.kl:.2f}", help="t-SNEs eigenes Optimierungsziel KL(P‖Q): wie gut die Nachbarschafts-Wahrscheinlichkeiten des Bildes zu denen der Daten passen (kleiner = besser).")

row1 = st.columns(2)
row2 = st.columns(2)
with row1[0]:
    st.markdown("**t-SNE**")
    st.plotly_chart(build_embedding(model.embedding, z_color, "t-SNE-Koordinate 1", "t-SNE-Koordinate 2"), width="stretch", key="tsne_embedding")
with row1[1]:
    st.markdown("**Isomap (Vergleich)**")
    st.plotly_chart(build_embedding(analysis.iso_2d, z_color[iso_idx], "Isomap-Koordinate 1", "Isomap-Koordinate 2"), width="stretch", key="iso_embedding")
with row2[0]:
    st.markdown("**LLE (Vergleich)**")
    st.plotly_chart(build_embedding(analysis.lle.embedding[:, :2], z_color, "LLE-Koordinate 1", "LLE-Koordinate 2"), width="stretch", key="lle_embedding")
with row2[1]:
    st.markdown("**PCA (Vergleich)**")
    st.plotly_chart(build_embedding(analysis.pca_2d, z_color, "PC1", "PC2"), width="stretch", key="pca_embedding")

st.table({
    "Verfahren": ["t-SNE", "Isomap", "LLE", "PCA"],
    "R² der Faktoren": [f"{metrics[k]['r2']:.2f}" for k in ("tsne", "isomap", "lle", "pca")],
    "Abstandstreue (gesamt)": [f"{metrics[k]['fid']:.2f}" for k in ("tsne", "isomap", "lle", "pca")],
    "nahe Paare": [f"{metrics[k]['near']:.2f}" for k in ("tsne", "isomap", "lle", "pca")],
    "ferne Paare": [f"{metrics[k]['far']:.2f}" for k in ("tsne", "isomap", "lle", "pca")],
    "Trustworthiness": [f"{metrics[k]['trust']:.2f}" for k in ("tsne", "isomap", "lle", "pca")],
})

st.markdown("---")

# --- Perplexity und Konvergenz ---------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von der Perplexity ab?")
st.markdown(
    """
Die **Perplexity** ist t-SNEs wichtigster Regler: sie legt fest, wie viele Nachbarn jede Tour 'sieht'. Klein - und die Einbettung zerfällt in Fragmente; groß - und die Nachbarschaften werden global.
Live für Ihr aktuelles Szenario über **feste Sweep-Seeds** (unabhängig vom Demo-Seed) geprüft, nicht behauptet:
"""
)
if code == "diverged":
    st.error(
        f"⛔ **Die Optimierung ist divergiert** (Iteration {vd['diverged_at']}): die Schritte waren so groß, dass die Punkte ins Unendliche flogen. Gezeigt wird der letzte endliche Stand - "
        "R² und Trustworthiness sind hier bedeutungslos. Im Test (Gauß-Kern, q = 4, Seed 7) verhinderten das eine Lernrate von 10 statt 50 oder eine Early Exaggeration von 4 statt 12."
    )
elif code == "lr_high":
    st.warning(
        f"⚠️ **Lernrate zu hoch**: die Optimierung endet bei einer KL-Divergenz von {vd['kl']:.2f}, ein Referenzlauf mit automatischer Lernrate erreicht {vd['kl_ref']:.2f} - die Schritte überschießen. "
        f"R² der Faktoren {vd['r2']:.2f} gegen {vd['r2_ref']:.2f}, Trustworthiness {vd['trust']:.2f} gegen {vd['trust_ref']:.2f}. Eine kleinere Lernrate (oder 'auto') beruhigt die Optimierung. "
        "(Bei großer Lernrate ist das Ergebnis chaotisch: je nach Datensatz und Rechner schwanken R² und Trustworthiness stark - die hohe KL-Divergenz ist das verlässliche Zeichen.)"
    )
elif code == "not_converged":
    st.warning(
        f"⚠️ **Noch nicht konvergiert**: nach {vd['n_iter']} Iterationen fällt die KL-Divergenz in den letzten 20 % der Iterationen noch um {vd['improvement'] * 100:.0f} % - R² der Faktoren erst {vd['r2']:.2f} "
        f"(Isomap {vd['r2_iso']:.2f}). Mehr Iterationen verbessern das Ergebnis (Kurve in Schritt 3 der Demo oben)."
    )
elif code == "crowding":
    st.warning(
        f"⚠️ **Crowding-Problem**: mit dem Gauß-Kern liegt die Trustworthiness bei {vd['trust']:.2f} statt {vd['trust_alt']:.2f} mit dem Student-t-Kern, die KL-Divergenz bei {vd['kl']:.2f} statt {vd['kl_alt']:.2f}. "
        "Bei mehr versteckten Faktoren als 2 Zielraum-Dimensionen gibt es im Bild zu wenig Platz für die vielen mittleren Abstände - der Student-t-Kern schafft ihn."
    )
elif code == "global_structure":
    st.warning(
        f"⚠️ **Keine globale Struktur**: mit {vd['outlier_pct']} % Sonderfahrten liegt das R² der Faktoren bei {vd['r2']:.2f} (PCA {vd['r2_pca']:.2f}, Isomap {vd['r2_iso']:.2f}), die Abstandstreue ferner Paare bei "
        f"{vd['far']:.2f} (PCA {vd['far_pca']:.2f}). t-SNE erhält Nachbarschaften, aber nicht die Größe von Abständen - die Extreme werden an den Rand der Wolke gestaucht. (Das R² wird hier von den "
        "Sonderfahrten dominiert - genau darum geht es.)"
    )
elif code == "perplexity_small":
    st.warning(
        f"⚠️ **Perplexity zu klein**: mit {vd['perplexity']:g} zählen nur die allerengsten Nachbarn, die Einbettung zerfällt in Fragmente - R² der Faktoren {vd['r2']:.2f}, Abstandstreue ferner Paare {vd['far']:.2f} "
        "(die Kurve unten zeigt, wie sie ab etwa 10 wieder steigt)."
    )
elif code == "no_advantage":
    st.info(f"ℹ️ **Kein Vorteil vor der PCA**: die Daten sind gerade (Krümmung 0) - R² der Faktoren {vd['r2']:.2f} (t-SNE) gegen {vd['r2_pca']:.2f} (PCA); ferne Paare {vd['far']:.2f} gegen {vd['far_pca']:.2f}.")
elif code == "tsne_wins":
    st.success(
        f"✅ **t-SNE entrollt die Fläche**: R² der Faktoren {vd['r2']:.2f} gegen {vd['r2_pca']:.2f} bei der PCA (Isomap {vd['r2_iso']:.2f}, LLE {vd['r2_lle']:.2f}), Trustworthiness {vd['trust']:.2f}. "
        f"Bei fernen Paaren bleibt es hinter Isomap zurück: Abstandstreue {vd['far']:.2f} gegen {vd['far_iso']:.2f}."
    )
else:
    st.info(f"t-SNE erreicht R² {vd['r2']:.2f} (PCA {vd['r2_pca']:.2f}, Isomap {vd['r2_iso']:.2f}) - kein klarer Gewinn und kein klarer Bruch.")

st.plotly_chart(build_perplexity_sweep(sweep_rows, float(perplexity)), width="stretch", key="perplexity_sweep")
best = max(sweep_rows, key=lambda r: r["r2"])
st.caption(
    f"Gleiche Daten-Einstellungen (q = {dataset.q}, Krümmung {curvature:.2f}, Rauschen {noise:.2f}, Sonderfahrten {int(outlier_pct)} %), nur die Perplexity wächst; {C.SWEEP_N_ITER} Iterationen, Lernrate auto, Exaggeration 12, "
    f"PCA-Start; Mittel über {len(C.SWEEP_SEEDS)} feste Seeds mit je {C.SWEEP_N_TOURS} Touren. Bestes R²: {best['r2']:.2f} bei Perplexity {best['perplexity']} - "
    "die Kurve ist ab etwa 10 **flach und zackig**, nur der Einbruch bei sehr kleinen Werten ist verlässlich."
)

st.markdown("**Konvergenz** (aktueller Lauf)")
st.plotly_chart(build_kl_curve(model.kl_history, model.exaggeration_iters, r2_points=r2_points), width="stretch", key="convergence_curve")
st.caption(
    f"KL-Divergenz je Iteration und R² der wahren Faktoren an den {len(r2_points)} Schnappschüssen dieses Laufs. KL am Ende: {model.kl:.2f}; grau: Early-Exaggeration-Phase ({model.exaggeration_iters} Iterationen). "
    "Die KL-Divergenz wird mit dem echten P gemessen, nicht mit dem übertriebenen."
)

st.markdown("---")

# --- Keine globale Struktur ------------------------------------------------------------------------------------------------

st.markdown("## 📏 t-SNE erhält Nachbarschaften, keine Abstände")
rng = np.random.default_rng(0)
m_iso = len(iso_idx)
pa = rng.integers(0, m_iso, size=min(1500, m_iso * (m_iso - 1) // 2))
pb = rng.integers(0, m_iso, size=len(pa))
keep = pa != pb
pa, pb = pa[keep], pb[keep]
ga, gb = iso_idx[pa], iso_idx[pb]


def _norm(d):
    return d / d.mean()


latent = _norm(np.linalg.norm(dataset.z[ga] - dataset.z[gb], axis=1))
tsne_d = _norm(np.linalg.norm(model.embedding[ga] - model.embedding[gb], axis=1))
iso_d = _norm(np.linalg.norm(analysis.iso_2d[pa] - analysis.iso_2d[pb], axis=1))
pca_d = _norm(np.linalg.norm(analysis.pca_2d[ga] - analysis.pca_2d[gb], axis=1))
st.plotly_chart(build_distance_fidelity(latent, [("t-SNE", tsne_d, "#d62728"), ("Isomap", iso_d, "#1f77b4"), ("PCA", pca_d, "#2ca02c")]), width="stretch", key="distance_fidelity")
st.caption(
    f"Jeder Punkt ein Tourenpaar: Abstand in der 2-D-Einbettung gegen den Abstand der wahren Faktoren. Auf der gestrichelten Diagonale wäre die Einbettung abstandstreu. "
    f"Korrelation für **nahe** Paare: t-SNE {metrics['tsne']['near']:.2f}, Isomap {metrics['isomap']['near']:.2f}, PCA {metrics['pca']['near']:.2f} - für **ferne** Paare: t-SNE {metrics['tsne']['far']:.2f}, "
    f"Isomap {metrics['isomap']['far']:.2f}, PCA {metrics['pca']['far']:.2f}. Wie weit entfernte Regionen auseinander liegen, ist t-SNE freigestellt - Sonderfahrten (Regler links) machen das drastisch sichtbar."
)

st.markdown("---")

# --- Stabilität ------------------------------------------------------------------------------------------------------------

st.markdown("## 🔀 Stabilität: derselbe Datensatz, verschiedene Starts")
st.caption(
    "Die Optimierung hat lokale Minima - ein anderes zufälliges Startlayout kann ein anderes Bild liefern. Vier Läufe mit zufälligen Starts (die Initialisierung links spielt hier keine Rolle); "
    "die Abweichung wird per Procrustes-Abstand nach bester Drehung/Spiegelung gemessen (0 = gleiches Bild, 1 = unabhängig)."
)
if st.button("🔀 Vier zufällige Starts rechnen", key="stability_start"):
    st.session_state["stability_on"] = True
if st.session_state.get("stability_on"):
    with st.spinner("Rechne vier t-SNE-Läufe..."):
        stab = _stability(data_params, settings)
    labels = [f"Start {s + 1}: R² {r:.2f}, KL {k:.2f}" for s, r, k in zip(stab["seeds"], stab["r2"], stab["kl"])]
    st.plotly_chart(build_stability(stab["embeddings"], z_color, labels), width="stretch", key="stability_plot")
    worst = max(stab["procrustes"][1:])
    st.caption(
        f"Procrustes-Abstand zum ersten Lauf: {', '.join(f'{p:.2f}' for p in stab['procrustes'][1:])}; R² zwischen {min(stab['r2']):.2f} und {max(stab['r2']):.2f}. "
        + ("Auf dieser einfachen Fläche finden alle Starts dasselbe Bild (bis auf Drehung)." if worst < 0.05 else "Die Läufe unterscheiden sich sichtbar: das Ergebnis hängt vom Start ab - deshalb ist die PCA-Initialisierung der Standard.")
    )

st.markdown("---")

# --- Out-of-sample ---------------------------------------------------------------------------------------------------------

st.markdown("## 🆕 Neue Touren einbetten (Out-of-sample)")
st.caption(
    "t-SNE hat **keine Abbildung** für neue Touren: die Punkte sind freie Parameter der Optimierung. Zwei Wege: (a) eine **Näherung** - die neue Tour bekommt den gewichteten Mittelwert der Koordinaten "
    f"ihrer {C.OOS_K} nächsten Trainings-Touren (kein Teil von t-SNE, nur eine Behelfslösung), oder (b) alles **neu rechnen** - dann verschieben sich aber auch die bereits eingebetteten Touren. "
    f"Test: die letzten {C.HOLDOUT_FRACTION * 100:.0f} % der Touren zurückhalten (LLE der Demo davor kann das über Rekonstruktionsgewichte)."
)
if st.button("🆕 Neue Touren testen", key="oos_start"):
    st.session_state["oos_on"] = True
if st.session_state.get("oos_on"):
    with st.spinner("Rechne t-SNE ohne und mit den neuen Touren..."):
        oos = _oos(data_params, settings)
    st.plotly_chart(build_out_of_sample(oos["model"].embedding, dataset.z[oos["train"], 0], oos["y_test"], dataset.z[oos["test"], 0]), width="stretch", key="oos_plot")
    st.caption(
        f"Sterne = zurückgehaltene Touren, mit der Näherung eingebettet: R² der wahren Faktoren **{oos['r2_test']:.2f}** für die neuen, {oos['r2_train']:.2f} für die Trainings-Touren. "
        f"Beim Neu-Rechnen mit allen Touren verschieben sich die Trainings-Touren um einen Procrustes-Abstand von **{oos['shift']:.2f}** (0 = unverändert). "
        "Über 6 feste Seeds (Standardeinstellungen) lag das R² der Näherung zwischen 0.67 und 0.95, die Verschiebung zwischen 0.01 und 0.47."
    )

st.markdown("---")

# --- Crowding --------------------------------------------------------------------------------------------------------------

st.markdown("## 🌀 Das Crowding-Problem: Student-t gegen Gauß")
st.caption(
    "Wenn die Fläche mehr als 2 Dimensionen hat (q ≥ 3), müssen mittlere Abstände in ein 2-D-Bild gezwängt werden - mit dem Gauß-Kern des ursprünglichen SNE staut sich alles in der Mitte, "
    "der schwere Rand des Student-t-Kerns schafft Platz. Beide Kerne bei sonst gleichen Einstellungen:"
)
if kernel == "gauss":
    crowd = {"gauss": {"model": model, **metrics["tsne"], "kl": model.kl}, "student": {"model": analysis.alt, **metrics["alt"]}}
else:
    if st.button("🌀 Beide Kerne vergleichen", key="crowding_start"):
        st.session_state["crowding_on"] = True
    crowd = None
    if st.session_state.get("crowding_on"):
        with st.spinner("Rechne beide Kerne..."):
            crowd = _crowding(data_params, settings)
if crowd is not None:
    st.plotly_chart(build_crowding(crowd["student"]["model"].embedding, crowd["gauss"]["model"].embedding, z_color), width="stretch", key="crowding_plot")
    st.table({
        "Kern": ["Student-t", "Gauß"],
        "KL-Divergenz": [f"{crowd[k]['kl']:.2f}" for k in ("student", "gauss")],
        "Trustworthiness": [f"{crowd[k]['trust']:.2f}" for k in ("student", "gauss")],
        "R² der Faktoren": [f"{crowd[k]['r2']:.2f}" for k in ("student", "gauss")],
    })
    st.caption(
        f"Mit q = {dataset.q} versteckten Faktoren: Trustworthiness {crowd['student']['trust']:.2f} (Student-t) gegen {crowd['gauss']['trust']:.2f} (Gauß). "
        + ("Bei q = 2 passt die Fläche in 2 Dimensionen - dort zeigt sich der Unterschied kaum (im Test Trustworthiness 0.99 gegen 0.98, R² 0.93 gegen 0.95)." if dataset.q <= 2 else "Bei q = 3 lagen im Test (Seed 7) Trustworthiness 0.97 gegen 0.90, KL 0.51 gegen 0.82 und R² 0.48 gegen 0.38 - bei q = 4 kann der Gauß-Kern schon in der Early-Exaggeration-Phase divergieren.")
    )

st.markdown("---")

# --- Rechenzeit ------------------------------------------------------------------------------------------------------------

st.markdown("## ⏱️ Rechenzeit: t-SNE gegen die anderen")
st.caption(
    "t-SNE berechnet in **jeder** Iteration alle n² Paar-Ähnlichkeiten - bei 750 Iterationen sind das 750 Durchläufe über eine n×n-Matrix. Isomap braucht einmal kürzeste Wege und eine Eigenzerlegung, "
    "LLE einmal eine Eigenzerlegung, die PCA nur eine 12×12-Zerlegung. (Schnellere Näherungen für große Datensätze gibt es, sie stehen hier nicht.)"
)
if "timing_rows" not in st.session_state:
    if st.button("⏱️ Rechenzeit messen (ca. 15 s)", key="timing_start", help=f"Misst t-SNE ({C.TIMING_N_ITER} Iterationen), Isomap, LLE und PCA für n = {', '.join(str(n) for n in C.TIMING_NS)} auf diesem Rechner."):
        with st.spinner("Messe..."):
            st.session_state["timing_rows"] = timing_sweep()
        st.rerun()
else:
    rows = st.session_state["timing_rows"]
    st.plotly_chart(build_timing(rows), width="stretch", key="timing_chart")
    st.table({
        "Touren n": [r["n"] for r in rows],
        "t-SNE": [f"{r['tsne']:.2f} s" for r in rows],
        "Isomap": [f"{r['isomap']:.3f} s" for r in rows],
        "LLE": [f"{r['lle']:.3f} s" for r in rows],
        "PCA": [f"{r['pca'] * 1000:.2f} ms" for r in rows],
        "t-SNE / Isomap": [f"{r['tsne'] / max(r['isomap'], 1e-9):.0f}×" for r in rows],
    })
    ns = np.array([r["n"] for r in rows], dtype=float)
    ts = np.array([r["tsne"] for r in rows])
    exponent = float(np.polyfit(np.log(ns[-3:]), np.log(ts[-3:]), 1)[0])
    st.caption(f"Gemessen auf diesem Rechner (Wandzeit, {C.TIMING_N_ITER} Iterationen, ein Lauf je n): über die letzten drei Punkte wächst die t-SNE-Zeit etwa mit n^{exponent:.1f} (theoretisch n² je Iteration; die gemessene Steigung hängt von Rechner und Speicher ab).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Affinitäten im Originalraum.** Für Punkte $x_i \in \mathbb{R}^{12}$ (z-Werte) ist die bedingte Wahrscheinlichkeit, dass $i$ den Punkt $j$ als Nachbarn wählt,

$$
p_{j|i} = \frac{\exp(-\lVert x_i - x_j \rVert^2 / 2\sigma_i^2)}{\sum_{l \ne i} \exp(-\lVert x_i - x_l \rVert^2 / 2\sigma_i^2)} .
$$

Die Breite $\sigma_i$ wird per Binärsuche so bestimmt, dass die **Perplexity** $\mathrm{Perp}(P_i) = 2^{H(P_i)}$ mit $H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$ einen Zielwert trifft. Symmetrisiert:
$p_{ij} = (p_{j|i} + p_{i|j}) / 2n$.

**Affinitäten im Zielraum.** Für Bildpunkte $y_i \in \mathbb{R}^2$ mit Student-t-Kern (ein Freiheitsgrad):
$q_{ij} = (1 + \lVert y_i - y_j \rVert^2)^{-1} \big/ \sum_{k \ne l} (1 + \lVert y_k - y_l \rVert^2)^{-1}$. Der ursprüngliche SNE-Kern ist $\exp(-\lVert y_i - y_j \rVert^2)$ (Gauß).

**Zielfunktion und Gradient.** $\mathrm{KL}(P \Vert Q) = \sum_{i \ne j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$, minimiert per Gradientenabstieg mit

$$
\frac{\partial\, \mathrm{KL}}{\partial y_i} = 4 \sum_{j} (p_{ij} - q_{ij})\,(y_i - y_j)\,(1 + \lVert y_i - y_j \rVert^2)^{-1}
$$

(beim Gauß-Kern ohne den letzten Faktor; beide Gradienten sind per finite Differenzen im Test geprüft). Dazu Momentum (0.5, nach der Early Exaggeration 0.8), adaptive Gains (delta-bar-delta),
und in der ersten Phase $P \to 12\,P$ (**Early Exaggeration**).

**Crowding-Problem.** In 2 Dimensionen gibt es weniger Platz für mittlere Abstände als in 12 - mit gleich schmalen Kernen in beiden Räumen zieht die Optimierung alle mittleren Punkte in die Mitte.
Der schwere Rand des Student-t-Kerns ($\sim d^{-2}$ statt $e^{-d^2}$) lässt mittlere Abstände im Bild wachsen, ohne die KL-Divergenz stark zu erhöhen.

**Grenzen.** (1) *Keine globale Struktur*: KL misst Nachbarschaften; Abstände zwischen fernen Punkten (und Extremwerte) sind nicht Teil des Ziels. (2) *Kein Out-of-sample*: die Punkte $y_i$ sind freie Parameter,
es gibt keine Abbildung $x \mapsto y$. (3) *Lokale Minima*: das Ergebnis hängt vom Start ab. (4) *Perplexity, Iterationen, Lernrate* sind zu wählen; (5) $O(n^2)$ Zeit und Speicher je Iteration in dieser exakten Variante.
Die Achsen und Abstände im Bild haben keine feste Bedeutung; Clustergrößen und Abstände zwischen Clustern sollten nicht interpretiert werden.

**Trustworthiness** (Venna & Kaski, 2001): $T = 1 - \frac{2}{nk(2n - 3k - 1)} \sum_i \sum_{j \in U_i} (r(i,j) - k)$ mit $U_i$ = Nachbarn in der Einbettung, die im Originalraum keine sind, und $r(i,j)$ ihrem Originalrang.
**Abstandstreue** = Pearson-Korrelation der Paarabstände der 2-D-Einbettung mit den Paarabständen der wahren Faktoren (nah/fern: untere/obere Hälfte der wahren Abstände).
**Procrustes-Abstand**: $1 - (\sum s_i)^2$ mit $s_i$ den Singulärwerten von $A^\top B$ nach Zentrierung und Normierung beider Einbettungen (wie `scipy.spatial.procrustes`).

Implementiert in `tsne_algorithm.py` (Affinitäten, Gradient, Optimierung, Näherung für neue Touren), `tsne_isomap.py` / `tsne_lle.py` (Vergleichsverfahren, wortgleich aus isomap-demo / lle-demo),
`tsne_scenario.py` (Lieferrouten-Generator, wortgleich aus pca-demo) und `tsne_evaluation.py` (Kennzahlen, Sweep, Verdict, Zeitmessung).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
