"""Defaults, Slider-Grenzen und Presets für die t-SNE-Demo. Merkmale und Erzeugungs-Konstanten sind wortgleich aus pca-demo übernommen
(dieselben Lieferrouten - dieselbe gekrümmte Fläche, an der PCA scheiterte); alles Übrige ist neu."""

# --- Merkmale: 12 Kennzahlen je Tour in 4 Gruppen zu je 3 (Name, Einheit, Mittelwert, typische Streuung in Einheiten) ------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_FEATURES = len(FEATURES)
FEATURE_NAMES = tuple(f[0] for f in FEATURES)
FEATURE_LABELS = tuple(f"{f[0]} [{f[1]}]" for f in FEATURES)
GROUPS = ("Größe", "Zeitdruck", "Verkehr", "Sonderfälle")     # je 3 aufeinanderfolgende Merkmale
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_FEATURES))

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 100, 600
DEFAULT_Q = 2
Q_MIN, Q_MAX = 1, 4
DEFAULT_CURVATURE = 1.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_OUTLIER_PCT = 0
OUTLIER_PCT_MIN, OUTLIER_PCT_MAX = 0, 10
DEFAULT_PERPLEXITY = 30
PERPLEXITY_MIN, PERPLEXITY_MAX = 2, 150            # die obere Grenze folgt zusätzlich der Tourenzahl (Perplexität < n − 1)
DEFAULT_N_ITER = 750
N_ITER_MIN, N_ITER_MAX = 50, 1000
LR_CHOICES = (0.0, 10.0, 50.0, 200.0, 500.0, 1000.0, 2000.0)     # 0 = auto (max(n / Exaggeration / 4, 50), wie scikit-learn)
DEFAULT_LR = 0.0
DEFAULT_EXAGGERATION = 12
EXAGGERATION_MIN, EXAGGERATION_MAX = 1, 30
KERNELS = ("student", "gauss")
KERNEL_LABELS = {"student": "Student-t (t-SNE)", "gauss": "Gauß (ursprüngliches SNE)"}
INITS = ("pca",) + tuple(f"random:{i}" for i in range(5))
INIT_LABELS = {"pca": "PCA (deterministisch)", **{f"random:{i}": f"zufällig, Start {i + 1}" for i in range(5)}}
DEFAULT_KERNEL, DEFAULT_INIT = "student", "pca"
DEFAULT_SEED = 7

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
OUTLIER_SCALE = 10.0                   # Sonderfahrten: latenter Faktor um diesen Faktor vergrößert
CROSS_LOADING = 0.15                   # kleine Querladungen zwischen Merkmalsgruppen
WITHIN_LOADINGS = (0.95, 0.9, 0.85)    # Ladung der drei Merkmale einer Gruppe auf ihren Faktor
CURVATURE_FREQUENCY = 1.6              # Frequenz der sin/cos-Terme der Krümmung
CURVATURE_AMPLITUDE = 2.0              # Länge jeder Spalte der Krümmungsmatrix (in z-Einheiten bei Krümmung 1)
LAYOUT_SEED = 20240915                 # feste Ladungs- und Krümmungsmatrizen (unabhängig vom Seed der Touren)


# --- Auswertung --------------------------------------------------------------------------------------------------------
TRUST_NEIGHBORS = 10
ISOMAP_K = 10                                     # Vergleichsverfahren mit ihren guten Einstellungen (siehe isomap-demo / lle-demo)
LLE_K, LLE_REG = 14, 1e-2
SWEEP_SEEDS = tuple(100_000 + i for i in range(3))                 # feste Sweep-Seeds, unabhängig vom Demo-Seed
SWEEP_PERPLEXITIES = (3, 5, 10, 20, 30, 50, 80)
SWEEP_N_TOURS = 200
SWEEP_N_ITER = 500
STABILITY_SEEDS = (0, 1, 2, 3)
HOLDOUT_FRACTION = 0.2
OOS_K = 10
TIMING_NS = (100, 200, 400, 600)
TIMING_N_ITER = 500
SNAPSHOT_COUNT = 12

_BASE = {"n_tours": DEFAULT_N_TOURS, "q": DEFAULT_Q, "curvature": DEFAULT_CURVATURE, "noise": DEFAULT_NOISE, "outlier_pct": DEFAULT_OUTLIER_PCT, "perplexity": DEFAULT_PERPLEXITY,
         "n_iter": DEFAULT_N_ITER, "learning_rate": DEFAULT_LR, "exaggeration": DEFAULT_EXAGGERATION, "kernel": DEFAULT_KERNEL, "init": DEFAULT_INIT, "seed": DEFAULT_SEED}
PRESETS = {
    "Gekrümmte Fläche: t-SNE entrollt": {**_BASE},
    "Sonderfahrten: t-SNE staucht die Extreme": {**_BASE, "outlier_pct": 5},
    "Perplexity zu klein": {**_BASE, "perplexity": 3},
    "Zu wenige Iterationen": {**_BASE, "n_iter": 50},
    "Lernrate zu hoch": {**_BASE, "learning_rate": 1000.0},
    "Gauß-Kern: Crowding": {**_BASE, "q": 3, "kernel": "gauss"},
}
PRESET_HELP = {
    "Gekrümmte Fläche: t-SNE entrollt": "Dieselbe gebogene Fläche wie in den Demos davor: t-SNE gewinnt die versteckten Faktoren gut zurück (R² ≈ 0.93 gegen 0.50 der PCA; Isomap 0.98, LLE 0.96) - bei fernen Tourenpaaren aber nur mit Abstandstreue 0.69 (Isomap 0.91).",
    "Sonderfahrten: t-SNE staucht die Extreme": "5 % Sonderfahrten mit extremen Werten: PCA (R² ≈ 0.76) und Isomap (0.75) behalten die Größenordnung, t-SNE fällt auf 0.12 - es erhält Nachbarschaften, keine Abstände, und drückt die Extreme an den Rand der Wolke (Abstandstreue ferner Paare 0.14 gegen 0.95 bei der PCA).",
    "Perplexity zu klein": "Mit Perplexity 3 schaut jede Tour nur auf etwa drei Nachbarn: die Einbettung zerfällt in Fragmente (R² ≈ 0.59 statt 0.93, Abstandstreue ferner Paare 0.07).",
    "Zu wenige Iterationen": "Nach nur 50 Iterationen hat die Optimierung noch nicht konvergiert: die KL-Divergenz liegt bei etwa 0.49 statt 0.31 und das R² bei 0.64 - die KL-Kurve fällt noch steil.",
    "Lernrate zu hoch": "Mit Lernrate 1000 (statt automatisch 50) überschießen die Gradientenschritte: die KL-Divergenz steigt wieder (1.6 statt 0.31), R² fällt auf 0.26, Trustworthiness auf 0.84.",
    "Gauß-Kern: Crowding": "Drei versteckte Faktoren, aber nur 2 Dimensionen zum Einbetten - mit dem Gauß-Kern des ursprünglichen SNE staut sich alles in der Mitte (Crowding): Trustworthiness 0.90 statt 0.97 mit Student-t, KL 0.82 statt 0.51.",
}
PRESET_EXPECTED_BANDS = {
    "Gekrümmte Fläche: t-SNE entrollt": {"verdict": "tsne_wins", "r2": (0.88, 0.98), "r2_iso": (0.94, 1.0), "r2_pca": (0.4, 0.6), "far": (0.55, 0.8), "far_iso": (0.85, 1.0)},
    "Sonderfahrten: t-SNE staucht die Extreme": {"verdict": "global_structure", "r2": (0.0, 0.3), "r2_pca": (0.6, 0.9), "r2_iso": (0.6, 0.9), "far": (0.0, 0.3), "far_pca": (0.85, 1.0)},
    "Perplexity zu klein": {"verdict": "perplexity_small", "r2": (0.45, 0.72), "far": (-0.1, 0.25)},
    "Zu wenige Iterationen": {"verdict": "not_converged", "r2": (0.5, 0.78), "kl": (0.4, 0.6)},
    "Lernrate zu hoch": {"verdict": "lr_high", "r2": (0.1, 0.45), "trust": (0.75, 0.92), "kl": (1.0, 2.5)},
    "Gauß-Kern: Crowding": {"verdict": "crowding", "trust": (0.85, 0.94), "trust_alt": (0.94, 1.0), "kl": (0.6, 1.1), "kl_alt": (0.4, 0.65)},
}
