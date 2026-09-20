"""Plotly-Visualisierungen der t-SNE-Demo: Nachbarschaft einer Tour (Gauß-Affinitäten), Kernel-Kurven, Einbettungen, KL-Verlauf, Perplexity-Sweep, Abstandstreue, Stabilität,
Out-of-sample, Crowding und Rechenzeit. Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios: keine Zoom-/Pan-Gesten im Chart)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, GREEN, RED, GRAY = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _scatter(coords, color, name="Touren", size=7, showscale=False, label="latenter Faktor 1", opacity=1.0):
    return go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers", name=name, hoverinfo="skip",
        marker=dict(color=color, colorscale="Viridis", size=size, showscale=showscale, opacity=opacity, line=dict(width=0.5, color="white"),
                    colorbar=dict(title=label) if showscale else None),
    )


def build_neighbour_view(coords, color, focus, neighbours, probabilities):
    """2-D-Ansicht (erste zwei Hauptkomponenten): gewählte Tour und ihre Nachbarn, Markergröße = Affinität p_j|i."""
    sizes = 8 + 34 * probabilities / max(probabilities.max(), 1e-12)
    fig = go.Figure(_scatter(coords, color, showscale=True, opacity=0.5))
    fig.add_trace(go.Scatter(x=coords[neighbours, 0], y=coords[neighbours, 1], mode="markers", name="Nachbarn (Größe = Affinität)", hoverinfo="skip",
                             marker=dict(color=ORANGE, size=sizes, line=dict(width=1, color="#14233B"))))
    fig.add_trace(go.Scatter(x=[coords[focus, 0]], y=[coords[focus, 1]], mode="markers", name="gewählte Tour", hoverinfo="skip",
                             marker=dict(color=RED, size=14, symbol="star", line=dict(width=1, color="#14233B"))))
    fig.update_xaxes(title="PC1 (Ansicht)")
    fig.update_yaxes(title="PC2 (Ansicht)")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_affinity_bars(sorted_probabilities, perplexity, sigma):
    """Affinitäten p_j|i der nächsten Nachbarn einer Tour (absteigend) und die Summe: die Perplexität ist die 'effektive Zahl' dieser Nachbarn."""
    shown = sorted_probabilities[:40]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=list(range(1, len(shown) + 1)), y=shown, marker_color=BLUE, name="Affinität p_j|i", hovertemplate="Nachbar %{x}: %{y:.3f}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=list(range(1, len(shown) + 1)), y=np.cumsum(sorted_probabilities)[:len(shown)], mode="lines", line=dict(color=ORANGE, width=3), name="Summe",
                             hovertemplate="Summe bis %{x}: %{y:.2f}<extra></extra>"), secondary_y=True)
    fig.add_vline(x=perplexity, line_dash="dash", line_color=GRAY, annotation_text=f"Perplexity {perplexity:g}", annotation_position="top right")
    fig.update_xaxes(title=f"Nachbarn dieser Tour, nach Nähe geordnet (Gauß-Breite σ = {sigma:.2f})")
    fig.update_yaxes(title_text="Affinität", secondary_y=False)
    fig.update_yaxes(title_text="Summe", range=[0, 1.02], secondary_y=True)
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_kernel_curves(current):
    """Wie schnell fällt die Ähnlichkeit im Zielraum mit dem Abstand? Student-t (schwere Ränder) gegen Gauß (Ränder fallen sofort ab) - beide unnormiert."""
    d = np.linspace(0, 6, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=1.0 / (1.0 + d ** 2), mode="lines", name="Student-t: 1 / (1 + d²)", line=dict(color=ORANGE, width=4 if current == "student" else 2)))
    fig.add_trace(go.Scatter(x=d, y=np.exp(-d ** 2), mode="lines", name="Gauß: exp(−d²)", line=dict(color=BLUE, width=4 if current == "gauss" else 2)))
    fig.update_xaxes(title="Abstand d im Zielraum")
    fig.update_yaxes(title="Ähnlichkeit (unnormiert, log)", type="log", range=[-6, 0.1])
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_embedding(coords, color, title_x, title_y, label="latenter Faktor 1", height=380):
    fig = go.Figure(_scatter(coords, color, showscale=True, label=label))
    fig.update_xaxes(title=title_x)
    fig.update_yaxes(title=title_y)
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=20, b=10))
    return lock_axes(fig)


def build_kl_curve(kl_history, exaggeration_iters, marker=None, r2_points=None):
    """KL(P‖Q) je Iteration (Early-Exaggeration-Phase grau hinterlegt) und, aus den Schnappschüssen, das R² der wahren Faktoren."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("KL-Divergenz KL(P‖Q)", "R² der wahren Faktoren"))
    its = np.arange(1, len(kl_history) + 1)
    fig.add_trace(go.Scatter(x=its, y=kl_history, mode="lines", line=dict(color=BLUE, width=3), showlegend=False, hovertemplate="Iteration %{x}: %{y:.3f}<extra></extra>"), row=1, col=1)
    if r2_points:
        fig.add_trace(go.Scatter(x=[p[0] for p in r2_points], y=[p[1] for p in r2_points], mode="lines+markers", line=dict(color=ORANGE, width=3), showlegend=False,
                                 hovertemplate="Iteration %{x}: %{y:.2f}<extra></extra>"), row=1, col=2)
    for col in (1, 2):
        if exaggeration_iters > 0:
            fig.add_vrect(x0=0, x1=exaggeration_iters, fillcolor="rgba(140,140,140,0.15)", line_width=0, row=1, col=col,
                          annotation_text="Early Exaggeration" if col == 1 else None, annotation_position="top left")
        if marker is not None:
            fig.add_vline(x=marker, line_dash="dot", line_color=RED, row=1, col=col)
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(type="log", col=1)
    fig.update_yaxes(range=[0, 1.02], col=2)
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_perplexity_sweep(rows, current):
    ps = [r["perplexity"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("R² der wahren Faktoren", "Abstandstreue ferner Paare und Trustworthiness"))
    fig.add_trace(go.Scatter(x=ps, y=[r["r2"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="R²"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ps, y=[r["far"] for r in rows], mode="lines+markers", line=dict(color=RED, width=3), name="Abstandstreue (fern)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ps, y=[r["trust"] for r in rows], mode="lines+markers", line=dict(color=GREEN, width=3), name="Trustworthiness"), row=1, col=2)
    for col in (1, 2):
        fig.add_vline(x=current, line_dash="dot", line_color=GRAY, row=1, col=col)
    fig.update_xaxes(title_text="Perplexity", type="log", tickvals=ps, ticktext=[str(p) for p in ps])
    fig.update_yaxes(range=[-0.1, 1.02])
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_distance_fidelity(latent_pairs, panels):
    """Paarabstände der 2-D-Einbettung gegen die Abstände der wahren Faktoren (je auf Mittelwert 1 normiert). `panels`: [(Titel, Abstände, Farbe)]; auf der Diagonalen ist die Einbettung abstandstreu."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels])
    for col, (_, pairs, color) in enumerate(panels, start=1):
        top = float(max(latent_pairs.max(), pairs.max())) * 1.05
        fig.add_trace(go.Scatter(x=latent_pairs, y=pairs, mode="markers", marker=dict(color=color, size=5, opacity=0.35), hoverinfo="skip", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=GRAY, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=col)
    fig.update_xaxes(title_text="Abstand der wahren Faktoren (normiert)")
    fig.update_yaxes(title_text="Abstand in der Einbettung (normiert)", col=1)
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_stability(embeddings, color, labels):
    """Vier Läufe mit verschiedenen zufälligen Start-Layouts (2 × 2)."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=labels, vertical_spacing=0.14)
    for i, emb in enumerate(embeddings):
        fig.add_trace(go.Scatter(x=emb[:, 0], y=emb[:, 1], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(color=color, colorscale="Viridis", size=5, line=dict(width=0.3, color="white"))), row=i // 2 + 1, col=i % 2 + 1)
    fig.update_layout(template="plotly_white", height=560, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_out_of_sample(train_embedding, train_color, test_embedding, test_color):
    fig = go.Figure(_scatter(train_embedding, train_color, showscale=True, name="Trainings-Touren", opacity=0.6))
    fig.add_trace(go.Scatter(x=test_embedding[:, 0], y=test_embedding[:, 1], mode="markers", name="neue Touren (Näherung)", hoverinfo="skip",
                             marker=dict(color=test_color, colorscale="Viridis", cmin=float(train_color.min()), cmax=float(train_color.max()), size=12, symbol="star",
                                         line=dict(width=1.5, color="#14233B"))))
    fig.update_xaxes(title="t-SNE-Koordinate 1")
    fig.update_yaxes(title="t-SNE-Koordinate 2")
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_crowding(student, gauss, color):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Student-t-Kern (t-SNE)", "Gauß-Kern (ursprüngliches SNE)"))
    for col, emb in enumerate((student, gauss), start=1):
        fig.add_trace(go.Scatter(x=emb[:, 0], y=emb[:, 1], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(color=color, colorscale="Viridis", size=6, line=dict(width=0.3, color="white"))), row=1, col=col)
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_timing(rows):
    ns = np.array([r["n"] for r in rows], dtype=float)
    fig = go.Figure()
    for key, label, color in (("tsne", "t-SNE", RED), ("isomap", "Isomap", BLUE), ("lle", "LLE", ORANGE), ("pca", "PCA", GREEN)):
        fig.add_trace(go.Scatter(x=ns, y=[max(r[key], 1e-6) for r in rows], mode="lines+markers", name=label, line=dict(color=color, width=3)))
    fig.update_xaxes(title="Anzahl Touren n", type="log")
    fig.update_yaxes(title="Rechenzeit (s)", type="log")
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)
