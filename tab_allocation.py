"""Onglet « Exercice 3 - Track record » : portefeuille à allocation dynamique du 01/01/2026 au 30/10/2026.

Exercice indépendant du reste de l'application : autre jeu de données (fichier « Exercice allocation 261113 »),
sans lien avec l'univers d'actions, le score ni les filtres de la barre latérale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px_
import plotly.graph_objects as go
import streamlit as st

import allocation as al
import data_utils as du

C_NAVY, C_BLUE, C_LIGHT, C_TEAL, C_PALE, C_RED, C_GREY = (du.C_NAVY, du.C_BLUE, du.C_LIGHT, du.C_TEAL, du.C_PALE,
                                                            du.C_RED, du.C_GREY)
TEMPLATE = "plotly_white"
EXCEL = Path(__file__).parent / "resultats" / "track_record_exercice_3.xlsx"   # version finale mise en forme
EXCEL_NAME = "Partie 3 - Track record d'un portefeuille dynamique.xlsx"
STRICT = Path(__file__).parent / "data" / "variante_plafond_quotidien.csv"
NAMES = {"opt": "Dynamique optimal", "static": "Statique optimal (0 réallocation)", "ew": "Équipondéré (29)"}
COLORS = {"opt": C_NAVY, "static": C_TEAL, "ew": C_GREY}
RULES = {"Toutes les places ouvertes (Europe et États-Unis)": "commun",
         "Au moins une place ouverte": "cotation",
         "Tous les jours calendaires": "calendaire"}
ETF = ["WELE GY Equity", "WELD GY Equity", "IXN US Equity"]
FAMILY_COLORS = {"Indices sectoriels européens": C_BLUE, "Indices américains": C_TEAL, "ETF": C_LIGHT}


@st.cache_data
def _prices():
    return al.load_prices()


@st.cache_data
def _solve(cap: float, k: int, rule: str):
    return al.optimise(_prices(), max_rebal=k, cap=cap, rule=rule)


@st.cache_data
def _frontier(cap: float, rule: str, k_max: int = 20):
    return al.frontier(_prices(), k_max=k_max, cap=cap, rule=rule), al.daily_oracle(_prices(), cap=cap)


@st.cache_data
def _checks(cap: float, rule: str, n_draws: int = 200_000, seed: int = 0):
    """Contrôles de l'optimum : recherche exhaustive (1 et 2 réallocations) et tirages aléatoires (10)."""
    px = _prices()
    p = al.period_prices(px)
    nodes = pd.DatetimeIndex([al.START]).append(al.candidate_dates(px, rule=rule)).append(pd.DatetimeIndex([al.END]))
    G = al.segment_values(p.loc[nodes].to_numpy(), cap)
    n = len(nodes)
    inner = np.arange(1, n - 1)
    b1 = np.max(G[0, inner] + G[inner, n - 1])
    i, j = np.triu_indices(n - 2, k=1)
    i, j = i + 1, j + 1
    b2 = np.max(G[0, i] + G[i, j] + G[j, n - 1])
    rng = np.random.default_rng(seed)
    rnd, chunk = -np.inf, 20_000
    for _ in range(n_draws // chunk):          # 10 dates distinctes par tirage, par paquets
        draws = np.sort(np.argpartition(rng.random((chunk, len(inner))), 10, axis=1)[:, :10] + 1, axis=1)
        path = np.hstack([np.zeros((chunk, 1), int), draws, np.full((chunk, 1), n - 1)])
        rnd = max(rnd, np.max(G[path[:, :-1], path[:, 1:]].sum(axis=1)))
    return {"exh1": np.exp(max(b1, G[0, n - 1])) - 1, "exh2": np.exp(max(b2, b1, G[0, n - 1])) - 1,
            "rand10": np.exp(rnd) - 1, "n_dates": n - 2}


def _fmt_pct(x, d=1):
    return f"{x * 100:,.{d}f} %".replace(",", " ").replace(".", ",")


# sens favorable de chaque indicateur : +1 plus haut = mieux, −1 plus bas = mieux
SENS = {"Performance totale": 1, "Performance annualisée": 1, "Volatilité annualisée": -1, "Ratio de Sharpe": 1,
        "Perte maximale (max drawdown)": 1, "Ratio de Calmar": 1, "Jours de hausse": 1, "Pire jour": 1,
        "Meilleur jour": 1}
KEY_ROWS = ["Performance totale", "Volatilité annualisée", "Ratio de Sharpe", "Perte maximale (max drawdown)"]
REF_CSS = "background-color: #F2F4F7; color: #4A5A70"


def _comp_table(comp: pd.DataFrame, rows: list[str]):
    """Valeur et écart avec l'équipondéré (1re colonne) ; vert favorable, rouge défavorable (même code que l'onglet 8)."""
    num = comp.loc[rows]
    ref = num.iloc[:, 0]
    txt = num.copy().astype(object)
    css = pd.DataFrame("", index=num.index, columns=num.columns)
    for r in num.index:
        ratio = r.startswith("Ratio")
        f = (lambda x: f"{x:.2f}".replace(".", ",")) if ratio else _fmt_pct
        txt.loc[r, num.columns[0]] = f(ref[r])
        css.loc[r, num.columns[0]] = REF_CSS
        scale = 0.5 * max(num.loc[r].abs().max(), 1e-9)
        for c in num.columns[1:]:
            v, d = num.loc[r, c], num.loc[r, c] - ref[r]
            gap = f"{d:+.2f}".replace(".", ",") if ratio else f"{d * 100:+.1f} pt".replace(".", ",")
            txt.loc[r, c] = f"{f(v)}   ({gap})"
            inten = min(1.0, abs(d) / scale)
            if inten >= 0.05:
                rgb = "0, 161, 154" if SENS.get(r, 0) * d > 0 else "214, 69, 61"
                css.loc[r, c] = (f"background-color: rgba({rgb}, {0.12 + 0.55 * inten:.2f});"
                                 + (" font-weight: 700" if inten > 0.5 else ""))
    return txt.style.apply(lambda _: css, axis=None)


def _family(t: str) -> str:
    if t in ETF:
        return "ETF"
    return "Indices américains" if t in al.US_MARKET else "Indices sectoriels européens"


def render(show, recap):
    px = _prices()
    p = al.period_prices(px)
    md = al.market_days(px).loc[p.index]

    # ------------------------------------------------------------------ Présentation de l'exercice
    st.subheader("Track record d'un portefeuille à allocation dynamique")
    st.markdown(f"""<div class='recap' style='margin-top:.2rem'>
<b class='t'>Exercice indépendant, nouveau jeu de données</b>
Fichier <b>« Exercice allocation 261113 »</b> : cours de clôture quotidiens de {px.shape[1]} sous-jacents (indices
sectoriels européens STOXX, indices américains, ETF), en devise de cotation.
<b class='t' style='margin-top:.6rem'>Consigne</b>
Construire, du 01/01/2026 au 30/10/2026, le track record du portefeuille à allocation dynamique qui maximise la
performance totale réalisée, avec <b>20 % maximum par sous-jacent</b> et <b>10 réallocations au plus</b>
(allocation de départ non comptée).
</div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------------ Jeu de données
    st.subheader("Sous-jacents disponibles")
    perf = (p.iloc[-1] / p.iloc[0] - 1).sort_values()
    k = st.columns(4)
    k[0].metric("Sous-jacents", px.shape[1])
    k[1].metric("Jours de cotation sur la période", int(md.cotation.iloc[1:].sum()))
    k[2].metric("Performance médiane", _fmt_pct(perf.median()))
    k[3].metric("Meilleur sous-jacent", _fmt_pct(perf.iloc[-1], 0), al.label(perf.index[-1]), delta_color="off")
    d = pd.DataFrame({"Sous-jacent": [f"{al.label(t)} ({t})" for t in perf.index], "Performance": perf.values,
                      "Famille": [_family(t) for t in perf.index]})
    fig = px_.bar(d, x="Performance", y="Sous-jacent", color="Famille", orientation="h", template=TEMPLATE,
                  color_discrete_map=FAMILY_COLORS, category_orders={"Sous-jacent": d["Sous-jacent"].tolist()[::-1]})
    fig.update_traces(hovertemplate="%{y} : %{x:+.1%}<extra></extra>")
    fig.update_layout(height=640, xaxis_tickformat="+.0%", yaxis_title="", xaxis_title="",
                      title="Performance du 01/01 au 30/10/2026", legend=dict(orientation="h", y=-0.06, title=""),
                      margin=dict(l=10))
    fig.update_yaxes(tickfont=dict(size=11))
    show(fig)
    st.markdown(f"Écart de {(perf.iloc[-1] - perf.iloc[0]) * 100:.0f} points entre le meilleur ({al.label(perf.index[-1])}) "
               f"et le moins bon sous-jacent ({al.label(perf.index[0])}). Indices de prix, dividendes non inclus ; "
               "Russell 2000, S&P Communication Services et iShares Global Tech cotés en dollars.")

    with st.expander("Paramètres (consigne pré-remplie)", expanded=False):
        c = st.columns(4)
        cap = c[0].slider("Poids maximum par sous-jacent", 0.05, 0.50, al.CAP, 0.05, format="%.2f",
                          help="0,20 = 20 %. Le plafond fixe le nombre minimal de lignes : 1 / plafond.")
        k_max = c[1].slider("Nombre maximal de réallocations", 0, 20, al.MAX_REBAL)
        cost = c[2].number_input("Frais de transaction (pb du montant échangé)", 0.0, 100.0, 0.0, 5.0,
                                 help="Appliqués à chaque réallocation sur Σ|nouveau poids − poids avant réallocation|.")
        rule_lab = c[3].selectbox("Dates de réallocation autorisées", list(RULES))
        rule = RULES[rule_lab]
    if abs(cap - al.CAP) > 1e-9 or k_max != al.MAX_REBAL:
        st.info("Paramètres différents de la consigne (20 %, 10 réallocations) : résultats indicatifs.")

    sol = _solve(cap, k_max, rule)
    static = _solve(cap, 0, rule)
    ports = {"opt": sol.weights, "static": static.weights, "ew": al.equal_weight(px)}
    navs, segs = {}, {}
    for key, w in ports.items():
        navs[key], segs[key] = al.backtest(px, w, cost_bps=cost if key == "opt" else 0.0)
    rf = al.rf_estr()
    mets = {key: al.metrics(nav, px, rf=rf) for key, nav in navs.items()}
    m, e = mets["opt"], mets["ew"]

    # ------------------------------------------------------------------ Track record
    st.subheader("Track record")
    kc = st.columns(5)
    kc[0].metric("Performance totale", _fmt_pct(m["Performance totale"], 0),
                 f"{(m['Performance totale'] - e['Performance totale']) * 100:+.0f} pt vs équipondéré")
    kc[1].metric("Performance annualisée", _fmt_pct(m["Performance annualisée"], 0))
    kc[2].metric("Volatilité annualisée", _fmt_pct(m["Volatilité annualisée"]),
                 f"{(m['Volatilité annualisée'] - e['Volatilité annualisée']) * 100:+.1f} pt vs équipondéré",
                 delta_color="inverse")
    kc[3].metric("Perte maximale", _fmt_pct(m["Perte maximale (max drawdown)"]),
                 f"{(m['Perte maximale (max drawdown)'] - e['Perte maximale (max drawdown)']) * 100:+.1f} pt vs équipondéré")
    n_used = int((sol.weights > 0).any(axis=1).sum())
    kc[4].metric("Réallocations utilisées", f"{sol.n_rebal} / {k_max}",
                 help=f"{n_used} sous-jacents différents détenus au moins une fois sur la période.")

    fig = go.Figure()
    for key in ["ew", "static", "opt"]:
        fig.add_trace(go.Scatter(x=navs[key].index, y=navs[key].values, name=NAMES[key], mode="lines",
                                 line=dict(color=COLORS[key], width=2.6 if key == "opt" else 1.8),
                                 hovertemplate="%{x|%d/%m/%Y} : %{y:.1f}<extra>" + NAMES[key] + "</extra>"))
    for i, d_ in enumerate(sol.dates[1:], start=1):
        fig.add_vline(x=d_, line=dict(color=C_PALE, width=1, dash="dot"))
        fig.add_annotation(x=d_, y=1.0, yref="paper", text=f"R{i}", showarrow=False, yanchor="bottom",
                           font=dict(size=10, color=C_NAVY))
    fig.update_layout(template=TEMPLATE, height=470, title="Valeur liquidative (base 100) et dates de réallocation (R1 à "
                      f"R{sol.n_rebal})", yaxis_title="VL", xaxis_title="", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.12), margin=dict(t=70))
    show(fig)

    comp = pd.DataFrame({NAMES[kk]: mets[kk] for kk in ["ew", "static", "opt"]})   # équipondéré = référence
    turnover = segs["opt"]["Rotation"].sum()
    st.markdown("**Indicateurs clés**")
    st.dataframe(_comp_table(comp, KEY_ROWS), width="stretch")
    st.markdown("**Détail**")
    st.dataframe(_comp_table(comp, [r for r in comp.index if r not in KEY_ROWS]), width="stretch")
    st.caption("Entre parenthèses : écart avec l'équipondéré. Vert : écart favorable ; rouge : défavorable. Statique optimal : meilleur portefeuille sans réallocation, "
               f"mêmes contraintes. Rotation cumulée du dynamique (achats + ventes) : {_fmt_pct(turnover, 0)}. "
               "Volatilité : rendements des jours de cotation, annualisés (√252). Sharpe : taux sans risque = €STR "
               f"moyen de la période, soit {_fmt_pct(rf, 2)} en taux annuel ([source]({al.ESTR_SOURCE})). "
               + (f"Frais de {cost:.0f} pb déduits pour le portefeuille dynamique." if cost else
                  "Performances brutes de frais."))

    # ------------------------------------------------------------------ Allocation dans le temps
    st.subheader("Composition au fil des réallocations")
    bounds = sol.dates + [al.END]
    rows = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        w = sol.weights[a]
        R = p.loc[b] / p.loc[a] - 1
        for t in w[w > 0].index:
            rows.append({"Sous-jacent": f"{t} · {al.label(t)}", "Début": a, "Fin": b, "Poids": w[t],
                         "Performance sur la période": R[t], "Période": "Départ" if i == 0 else f"R{i}"})
    tl = pd.DataFrame(rows)
    order = tl.groupby("Sous-jacent")["Début"].min().sort_values().index.tolist()   # premier détenu en haut
    fig = px_.timeline(tl, x_start="Début", x_end="Fin", y="Sous-jacent", color="Performance sur la période",
                       color_continuous_scale=[C_PALE, C_BLUE, C_NAVY], template=TEMPLATE,
                       hover_data={"Période": True, "Poids": ":.0%", "Performance sur la période": ":+.1%",
                                   "Sous-jacent": False}, category_orders={"Sous-jacent": order})
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1], autorange=True, title="")
    fig.update_layout(height=max(420, 26 * len(order)), coloraxis_colorbar=dict(title="Perf.", tickformat="+.0%"),
                      title="Périodes de détention (20 % par ligne) ; couleur : performance du sous-jacent sur la période")
    for d_ in sol.dates[1:]:
        fig.add_vline(x=d_, line=dict(color="#C5CED9", width=1, dash="dot"))
    show(fig)

    per = al.periods_table(px, sol.weights)
    contrib = al.contributions(px, sol.weights).sort_values()
    contrib = contrib[contrib.abs() > 1e-9]
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        lab_ = [f"{r['Période'].replace('Réallocation ', 'R')}<br>{pd.Timestamp(r['Début']):%d/%m}" for _, r in per.iterrows()]
        fig.add_bar(x=lab_, y=per["Portefeuille"], name="Dynamique optimal", marker_color=C_NAVY,
                    hovertemplate="%{y:+.1%}<extra>Dynamique optimal</extra>")
        fig.add_bar(x=lab_, y=per["Équipondéré (29)"], name="Équipondéré (29)", marker_color=C_GREY,
                    hovertemplate="%{y:+.1%}<extra>Équipondéré</extra>")
        fig.update_layout(template=TEMPLATE, barmode="group", height=max(420, 19 * len(contrib)), yaxis_tickformat="+.0%",
                          title="Performance de chaque période", legend=dict(orientation="h", y=-0.2), bargap=0.25)
        show(fig, c1)
    with c2:
        fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h", marker_color=C_BLUE,
                               hovertemplate="%{y} : %{x:.1f} pts<extra></extra>"))
        fig.update_layout(template=TEMPLATE, height=max(420, 19 * len(contrib)),
                          title="Contribution à la performance (points de VL)", xaxis_title="Points", margin=dict(l=10))
        fig.update_yaxes(tickmode="array", tickvals=list(contrib.index), tickfont=dict(size=10))
        show(fig, c2)

    # ------------------------------------------------------------------ Valeur des réallocations
    st.subheader("Apport des réallocations")
    fr, oracle = _frontier(cap, rule)
    c1, c2 = st.columns([3, 2])
    with c1:
        colors = [C_NAVY if i <= al.MAX_REBAL else C_PALE for i in fr.index]
        fig = go.Figure(go.Bar(x=fr.index, y=fr.values, marker_color=colors,
                               hovertemplate="%{x} réallocation(s) au plus : %{y:+.1%}<extra></extra>"))
        fig.update_layout(template=TEMPLATE, height=380, yaxis_tickformat="+.0%", xaxis_title="Nombre maximal de "
                          "réallocations", title="Performance maximale selon le nombre de réallocations "
                          "(bleu foncé : dans la consigne)")
        show(fig, c1)
    with c2:
        gains = fr.diff().dropna()
        chk = _checks(cap, rule)
        st.markdown(f"""
* **Sans réallocation**, le meilleur portefeuille atteint {_fmt_pct(fr[0], 0)} : l'essentiel de la performance vient
  du choix des dates.
* **Gain par réallocation** : de {gains.iloc[:5].min() * 100:.0f} à {gains.iloc[:5].max() * 100:.0f} points pour les
  5 premières, environ {gains.iloc[5:10].mean() * 100:.0f} points pour les suivantes. Le gain marginal reste élevé :
  la limite de 10 réallocations est contraignante.
* **Borne théorique** : réallocation quotidienne sans limite, {_fmt_pct(oracle, 0)}.
* **Contrôle de l'optimum** : recherche exhaustive pour 1 et 2 réallocations ({_fmt_pct(chk['exh1'])} et
  {_fmt_pct(chk['exh2'])}, identiques à l'algorithme) ; meilleur de 200 000 jeux de 10 dates tirés au hasard parmi
  {chk['n_dates']} : {_fmt_pct(chk['rand10'], 0)}, contre {_fmt_pct(fr[min(10, fr.index.max())], 0)}.
""")

    # ------------------------------------------------------------------ Synthèse, méthode et limites
    best_tickers = per["Sous-jacents détenus (20 % chacun)"].str.split(", ").explode().value_counts()
    perf0 = al.backtest(px, sol.weights)[0].iloc[-1] / 100 - 1
    perf10 = al.backtest(px, sol.weights, cost_bps=10)[0].iloc[-1] / 100 - 1
    recap([
        f"<b>Résultat</b> : {_fmt_pct(m['Performance totale'], 0)} du 01/01 au 30/10/2026 (VL {navs['opt'].iloc[-1]:.0f}), "
        f"contre {_fmt_pct(mets['static']['Performance totale'])} sans réallocation et "
        f"{_fmt_pct(e['Performance totale'])} pour l'équipondéré.",
        f"<b>Risque</b> : volatilité {_fmt_pct(m['Volatilité annualisée'])} (équipondéré "
        f"{_fmt_pct(e['Volatilité annualisée'])}) et perte maximale {_fmt_pct(m['Perte maximale (max drawdown)'])} : "
        "la concentration sur 5 lignes accroît la volatilité, mais l'allocation ex post évite les fortes baisses.",
        f"<b>Rotation</b> : {sol.n_rebal} réallocations, presque toujours complètes (rotation cumulée "
        f"{_fmt_pct(turnover, 0)}) ; sous-jacents les plus souvent détenus : "
        + ", ".join(f"{t} ({n}×)" for t, n in best_tickers.head(4).items()) + ".",
        f"<b>Frais</b> : à 10 pb du montant échangé, la valeur finale baisse de "
        f"{_fmt_pct(1 - (1 + perf10) / (1 + perf0))} ; performance de {_fmt_pct(perf10, 0)} au lieu de "
        f"{_fmt_pct(perf0, 0)}.",
    ], methode=(
        "Entre deux dates d'allocation, les positions sont conservées et les poids évoluent avec les prix ; le plafond "
        "de 20 % s'applique à chaque date d'allocation. Sur une période, la valeur finale est linéaire dans les poids : "
        "l'optimum place 20 % sur les 5 sous-jacents qui progressent le plus. Les dates sont choisies par programmation "
        "dynamique sur (date, nombre de réallocations), ce qui donne l'optimum global. Réallocation possible les jours "
        "où toutes les places sont ouvertes ; le résultat est identique avec tous les jours calendaires."
    ))
    drift = al.max_drift(px, sol.weights)
    strict_txt = ""
    if STRICT.exists():
        sv = pd.read_csv(STRICT)
        strict_txt = (f" Avec un plafond respecté chaque jour, l'optimum atteint "
                      f"{_fmt_pct(float(np.prod(1 + sv['perf']) - 1), 0)} (mêmes dates).")
    st.markdown(f"""
**Limites**
- Les poids évoluent entre deux réallocations : un sous-jacent peut dépasser 20 % en cours de période (au plus
  {_fmt_pct(drift[0])}, {drift[1]} le {drift[2]:%d/%m/%Y}).{strict_txt}
- Indices de prix (dividendes non inclus) ; sous-jacents en dollars agrégés sans conversion de change.
- Exécution au cours de clôture, sans contrainte de liquidité ; performances brutes de frais par défaut.
""")
    c1, c2 = st.columns(2)
    if EXCEL.exists():
        c1.download_button("Télécharger le classeur Excel (track record en formules)", EXCEL.read_bytes(), EXCEL_NAME,
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    tr = pd.DataFrame({NAMES[kk]: v for kk, v in navs.items()})
    c2.download_button("Télécharger le track record (CSV)", tr.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
                       "track_record_portefeuille_dynamique.csv", "text/csv")
