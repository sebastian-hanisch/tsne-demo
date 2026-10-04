# t-SNE an Lieferrouten-Kennzahlen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-tsne-demo.streamlit.app/)**

Viertes Stück der **Dimensionsreduktion-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **t-SNE** – an einem wachsenden Beispiel. Vehikel: **dieselben 12
Lieferrouten-Kennzahlen wie in [pca-demo](../pca-demo), [isomap-demo](../isomap-demo) und [lle-demo](../lle-demo)** (der Generator ist wortgleich kopiert und per Test gegen dessen Ausgabe eingefroren), erzeugt
aus wenigen versteckten Faktoren – dieselbe gekrümmte Fläche, an der PCA scheiterte. PCA, Isomap und LLE stehen als Vergleich daneben.

**Einordnung in die Reihe (die Kanten des Graphen):** t-SNE **behebt die Linearitätsschwäche der PCA** – wie Isomap und LLE, aber mit einem **dritten, probabilistischen** Mechanismus: statt geodätischer Abstände
(Isomap) oder lokaler Rekonstruktion (LLE) vergleicht es **Nachbarschafts-Wahrscheinlichkeiten** (Gauß im Original, Student-t im Bild). Es hat dafür **eigene Schwächen**, die die Demo live zeigt, und ist der Anfang
der Kette t-SNE → UMAP → PaCMAP:
```
pca-demo → isomap-demo   (global-geodätisch)
pca-demo → lle-demo      (Kontrast zu Isomap: lokal-linear)
pca-demo → tsne-demo     (probabilistisch; Schwäche: keine globale Struktur, kein Out-of-sample, Start-/Perplexity-/Optimierungs-Empfindlichkeit, O(n²) je Iteration)
tsne-demo → umap-demo → pacmap-demo | autoencoder-demo   (weitere Stücke, alle gebaut)
```

## Was die Demo zeigt

1. **t-SNE in Aktion** (Schritt-Slider + Abspielen): Gauß-Nachbarschaft einer Tour (Affinitäten, σ, Perplexity als "effektive Nachbarzahl") → Zielraum-Kern (Student-t gegen Gauß) und Startlayout →
   **Optimierung** (Schnappschüsse durch die Iterationen, KL-Kurve, Early-Exaggeration-Phase, R² je Schnappschuss) → Ergebnis neben der PCA.
2. **Was t-SNE gefunden hat – und die anderen drei Verfahren auf denselben Daten:** vier Einbettungen, R² der wahren Faktoren, Abstandstreue (gesamt/nah/fern), Trustworthiness, KL-Divergenz.
3. **📐 Wie stark hängt das Ergebnis von der Perplexity ab?** (live über feste Sweep-Seeds ab 100000, unabhängig vom Demo-Seed) plus Konvergenz-Kurve, mit Verdict (divergiert → Lernrate zu hoch → nicht
   konvergiert → Crowding → keine globale Struktur → Perplexity zu klein → kein Vorteil → t-SNE gewinnt).
4. **📏 Abstände** (nah/fern), **🔀 Stabilität** (vier zufällige Starts, Procrustes-Abstand), **🆕 Out-of-sample** (Näherung gegen Neuberechnung), **🌀 Crowding** (Student-t gegen Gauß-Kern), **⏱️ Rechenzeit** – die
   Experimente mit Knopf (Rechenzeit, Stabilität, Out-of-sample, Crowding) laufen auf Abruf und werden bei Änderungen neu berechnet.

Regler: Touren, wahre Faktoren q, Krümmung, Rauschen, **Sonderfahrten** (der Generator aus pca-demo), Perplexity, Iterationen, Lernrate, Early Exaggeration, Zielraum-Kern, Initialisierung.

Messwerte (Seed 7, 300 Touren, q = 2, Perplexity 30, 750 Iterationen, wenn nicht anders angegeben; die Presets prüfen sie mit Bändern):

| Situation | Messung |
|---|---|
| Gekrümmte Fläche | R² der wahren Faktoren **0.93** (t-SNE) gegen 0.98 (Isomap), 0.96 (LLE), 0.50 (PCA); Trustworthiness 0.99; **Abstandstreue ferner Paare 0.69** gegen 0.91 (Isomap), 0.74 (LLE), 0.38 (PCA) |
| 5 % Sonderfahrten | t-SNE **R² 0.12**, PCA 0.76, Isomap 0.75, LLE 0.49; Abstandstreue ferner Paare 0.14 gegen 0.95 (PCA) |
| Perplexity 3 | R² **0.59**, Abstandstreue ferner Paare 0.07 (Fragmente) |
| 50 Iterationen | R² **0.64**, KL 0.49 statt 0.31; mit 100 Iterationen 0.85, mit 500 0.93 |
| Lernrate 2000 (auto = 50) | KL **1.7** statt 0.31, Trustworthiness 0.85, R² 0.20 (über 4 Datensätze: KL 1.2–1.7, Trustworthiness 0.85–0.91, R² 0.18–0.51 – das R² schwankt stark, die hohe KL nicht) |
| q = 3, Gauß-Kern | Trustworthiness **0.90** gegen 0.97 (Student-t), KL 0.82 gegen 0.51, R² 0.38 gegen 0.48 |
| Gerade Daten (Krümmung 0) | **kein Vorteil**: R² 0.94 gegen 0.98 (PCA) |

**Sonderfahrten** (Anteil der Touren mit zehnfach vergrößertem Zeitdruck-Faktor; das R² wird hier von den Extremen dominiert – genau das ist der Punkt): R² t-SNE / PCA / Isomap / LLE bei 0 %: 0.93 / 0.50 / 0.98 / 0.96;
1 %: 0.61 / 0.67 / 0.69 / 0.81; 2 %: 0.45 / 0.65 / 0.65 / **0.25**; 5 %: 0.12 / 0.76 / 0.75 / 0.49; 10 %: 0.16 / 0.75 / 0.78 / 0.41. t-SNE bricht schon bei 1 % ein; LLE ist bei 2 % und 10 % ebenfalls instabil.
Die Perplexity ändert daran nichts (bei 5 %: R² 0.12 / 0.12 / 0.09 für Perplexity 5 / 30 / 100).

**Perplexity-Fenster** (gekrümmte Daten, 3 feste Seeds × 200 Touren, 500 Iterationen): R² 0.58 (3), 0.75 (5), **0.92 (10)**, 0.86 (20), 0.90 (30), 0.93 (50), 0.88 (80); Abstandstreue ferner Paare 0.00 (3), 0.26 (5), 0.65 (10), 0.61 (20), 0.59 (30),
0.73 (50), 0.65 (80). Nur der Einbruch unterhalb von etwa 10 ist verlässlich, danach ist die Kurve flach und zackig – kein zweiter Einbruch bei großen Perplexitäten (siehe "Was nicht funktioniert hat").

**Lernrate und Early Exaggeration:** Lernrate 2 … 50: R² 0.93–0.94; 200: 0.90; 1000: 0.26 (Seed 7; über die Seeds 7–10: 0.26 / 0.59 / 0.72 / 0.35); 5000: ≈ 0. Zu kleine Lernraten schaden bei 750 Iterationen nicht (die adaptiven Gains gleichen aus). Early Exaggeration 1: R² 0.86,
Abstandstreue ferner Paare **0.33** (bei 12: 0.69); 4: 0.69; 30: 0.61 – die Übertreibung sortiert zuerst die groben Gruppen, ohne sie geht globale Ordnung verloren.

**Crowding** (Perplexity 30, Student-t / Gauß): q = 2: R² 0.93 / 0.95, Trustworthiness 0.99 / 0.98, KL 0.31 / 0.31 – kein Unterschied, die Fläche passt in 2 Dimensionen; q = 3: R² 0.48 / 0.38, Trustworthiness 0.97 / 0.90,
KL 0.51 / 0.82; q = 4: der Gauß-Kern **divergiert** schon in Iteration 96 (Early-Exaggeration-Phase; KL 18.3, Trustworthiness 0.53), Student-t bleibt stabil (R² 0.21, Trustworthiness 0.91, KL 0.80). Mit Lernrate 10 statt 50
oder Early Exaggeration 4 statt 12 läuft der Gauß-Kern bei q = 4 durch (KL 1.03).

**Stabilität** (vier zufällige Starts, Perplexity 30): q = 2 sauber – alle vier Läufe liefern dasselbe Bild (Procrustes-Abstand 0.00 zum PCA-Start); q = 3: R² 0.36–0.56, Procrustes-Abstand zum PCA-Start 0.46–0.81;
q = 2 mit Rauschen 0.8: R² 0.73–0.79, Procrustes 0.21–0.51.

**Out-of-sample** (letzte 20 % zurückgehalten, 6 feste Seeds): die Näherung (gewichteter Mittelwert der 10 nächsten Trainings-Touren – kein Teil von t-SNE) erreicht R² 0.67–0.95; beim Neu-Rechnen mit allen Touren
verschieben sich die bereits eingebetteten um einen Procrustes-Abstand von 0.01–0.47. (LLE kann neue Touren über Rekonstruktionsgewichte einbetten: Median 0.94, siehe lle-demo.)

**Rechenzeit** (lokale Messung, 500 Iterationen, ein Lauf je n): n = 100: t-SNE 0.09 s, Isomap 3 ms, LLE 5 ms; n = 400: 3.1 s, 0.19 s, 0.05 s; **n = 600: 7 s, 0.6 s, 0.12 s** (PCA 0.2 ms) – t-SNE rechnet in jeder Iteration alle n² Paare.

## Modell und Verfahren

- **Generator** (`tsne_scenario.py`): wortgleich aus pca-demo; latente Faktoren, 12 Merkmale in 4 Gruppen, Krümmung `κ·B·h(z)`, Rauschen, Sonderfahrten. t-SNE arbeitet immer auf z-Werten.
- **t-SNE** (`tsne_algorithm.py`, numpy, ohne sklearn, exakte O(n²)-Variante): Gauß-Affinitäten mit Perplexity per Binärsuche über log β (alle Zeilen gleichzeitig), symmetrisiert; Student-t-Kern (oder Gauß-Kern für
  das ursprüngliche SNE, um `exp(−d²)` gegen Unterlauf verschoben); Gradientenabstieg auf KL(P‖Q) mit Momentum 0.5 → 0.8, Gains (delta-bar-delta), Early Exaggeration ×12 für min(250, n_iter/4) Iterationen, Lernrate
  `auto` = max(n/12/4, 50) (also 50 für alle n der Demo), PCA-Start auf Std 1e-4 oder zufällig (Start-Nummer entkoppelt vom Datensatz-Seed). Divergiert die Optimierung, wird der letzte endliche Stand
  eingefroren und gemeldet.
- **Isomap** (`tsne_isomap.py`) und **LLE** (`tsne_lle.py`): wortgleich aus isomap-demo / lle-demo kopiert (nur Vergleichsverfahren, feste gute Einstellungen: k = 10 bzw. k = 14 mit Regularisierung 0.01).
- **Auswertung** (`tsne_evaluation.py`): R² der wahren Faktoren aus den zwei Koordinaten per quadratischer Regression; Abstandstreue = Pearson-Korrelation der Paarabstände mit den Faktor-Paarabständen, getrennt für
  die untere (nah) und obere Hälfte (fern) der wahren Abstände; Trustworthiness (Venna & Kaski, eigene Implementierung); KL-Divergenz; Procrustes-Abstand; Sweeps und Zeitmessung.

## Was nicht funktioniert hat / Grenzen

- **Kein zweiter Einbruch bei großer Perplexity:** ein Einzel-Seed-Lauf (q = 3, Perplexity 100: R² 0.30 gegen 0.56 bei Perplexity 5) sah nach "zu große Perplexity" aus, ließ sich über die festen Sweep-Seeds aber nicht
  reproduzieren (q = 3: 0.48, 0.53, 0.45, 0.38, 0.46, 0.50, 0.47 für Perplexity 3 … 80). Das Verdict kennt deshalb nur "zu klein".
- **Zu kleine Lernrate ist kein Fehlerfall** (Lernrate 2 liefert bei 750 Iterationen dasselbe wie 50) – die Demo zeigt nur "zu hoch".
- **Cluster-Modus verworfen:** die geplante Zusatzdatenmenge mit echten Clustern wurde nicht gebaut (Vehikel bleibt dieselbe Fläche); die fehlende globale Struktur zeigt sich stattdessen an den Sonderfahrten.
- **Gleiche Implementierung wie sklearn nur bei gleichem Zeitplan und größerem n:** mit 300 Touren stimmen KL (0.3098) und Einbettung mit `sklearn.manifold.TSNE(method="exact")` überein; mit 120 Touren landen beide in
  verschiedenen lokalen Minima (KL 0.21 gegen 0.30) – die Optimierung ist chaotisch. Der Test prüft deshalb nur den Fall mit 300 Touren.
- **Gauß-Kern:** `exp(−d²)` lief bei weit gestreuten Punkten in den Unterlauf (NaN); jetzt verschoben. Mit Lernrate 2000 divergiert er trotzdem – die Demo meldet das statt abzustürzen.
- **Grenzen (Text):** Achsen und Abstände im Bild haben keine feste Bedeutung; Clustergrößen und Abstände zwischen Clustern sollten nicht interpretiert werden. Der exakte O(n²)-Algorithmus ist auf ≈ 600 Touren
  ausgelegt; größere Datensätze brauchen Näherungen (Barnes-Hut/FFT), die hier nicht gebaut sind.

## Verifikation

- t-SNE gegen `sklearn.manifold.TSNE` (exakt, 300 Touren): KL-Divergenz und Einbettung bis auf Drehung gleich; gemeinsame Affinitäten gegen `sklearn.manifold._t_sne._joint_probabilities` (2e-6).
- Zeilen-Perplexität = Zielwert (1e-6), P symmetrisch mit Summe 1; Gradient (Student-t und Gauß) gegen finite Differenzen (1e-7); Gauß-Kern ohne Unterlauf; PCA-Start deterministisch, zufälliger Start hängt nur
  von der Start-Nummer ab; Divergenz-Erkennung; Procrustes gegen `scipy.spatial.procrustes`.
- Generator bit-identisch zu pca-demo, Isomap-Kopie bit-identisch zu isomap-demo (eingefrorene Referenzwerte), LLE-Kopie: zentriert, konstanter Eigenvektor, Fehler bei k > d ohne Regularisierung;
  Trustworthiness gegen `sklearn.manifold.trustworthiness` (1e-9).
- Sweeps über feste Seeds deterministisch; Verdict-Codes (inkl. Divergenz); die Aussage im Hilfetext der Early Exaggeration ist als Test hinterlegt; alle 6 Presets in kalibrierten Bändern; AppTest-Rauchtests
  (Default, jedes Preset, jeder Schritt, Randgrößen, Perplexity folgt n, Experimente auf Abruf, Gauß-Vergleich ohne Knopf), Achsensperre aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Perplexity/Konvergenz, Abstände, Stabilität, Out-of-sample, Crowding, Rechenzeit, Mathe |
| `tsne_algorithm.py` | t-SNE von Grund auf (Affinitäten, Gradient, Optimierung, Näherung für neue Touren, Procrustes) |
| `tsne_isomap.py`, `tsne_lle.py` | Vergleichsverfahren (wortgleich aus isomap-demo / lle-demo) |
| `tsne_scenario.py`, `tsne_constants.py` | Lieferrouten-Generator (wortgleich aus pca-demo), Konstanten, Presets |
| `tsne_evaluation.py` | Kennzahlen, Sweeps, Verdict, Stabilität, Out-of-sample, Crowding, Zeitmessung |
| `tsne_presets.py`, `tsne_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | sklearn-/scipy-Kreuzvergleiche, Generator-Referenz, Auswertung, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Dimensionsreduktion: von PCA bis Autoencoder](https://sebastianhanisch.net/konzepte-dimensionsreduktion.html).
