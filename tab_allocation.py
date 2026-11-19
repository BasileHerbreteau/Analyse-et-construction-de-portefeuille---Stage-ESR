"""Onglet « Allocation dynamique » : track record ex post du 01/01/2026 au 30/10/2026 (exercice séparé)."""
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
EXCEL = Path(__file__).parent / "resultats" / "track_record_allocation_dynamique.xlsx"
STRICT = Path(__file__).parent / "data" / "variante_plafond_quotidien.csv"
NAMES = {"opt": "Dynamique optimal", "static": "Statique optimal (0 réallocation)", "ex": "Exemple du fichier",
         "ew": "Équipondéré (29)"}
COLORS = {"opt": C_NAVY, "static": C_TEAL, "ex": C_LIGHT, "ew": C_GREY}
RULES = {"Toutes les places ouvertes (Europe et États-Unis)": "commun",
         "Au moins une place ouverte": "cotation",
         "Tous les jours calendaires": "calendaire"}


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
    return f"{x * 100:,.{d}f}\u00a0%".replace(",", "\u00a0").replace(".", ",")


def render(show, recap):
    px = _prices()
    st.subheader("Allocation dynamique : track record du 01/01/2026 au 30/10/2026")
    st.caption("Exercice séparé, fichier « Exercice allocation 261113 » : 29 sous-jacents (indices sectoriels STOXX, "
               "Russell 2000, indice sectoriel américain, ETF), prix quotidiens en devise de cotation. Consigne : maximiser "
               "la performance totale réalisée avec 20 % maximum par sous-jacent et 10 réallocations au plus "
               "(l'allocation de départ ne compte pas).")

    with st.expander("Paramètres (la consigne est pré-remplie)", expanded=False):
        c = st.columns(4)
        cap = c[0].slider("Poids maximum par sous-jacent", 0.05, 0.50, al.CAP, 0.05, format="%.2f",
                          help="0,20 = 20 %. Le plafond fixe le nombre minimal de lignes : 1 / plafond.")
        k = c[1].slider("Nombre maximal de réallocations", 0, 20, al.MAX_REBAL)
        cost = c[2].number_input("Frais de transaction (pb du montant échangé)", 0.0, 100.0, 0.0, 5.0,
                                 help="Appliqués à chaque réallocation sur Σ|nouveau poids − poids dérivé|.")
        rule_lab = c[3].selectbox("Dates de réallocation autorisées", list(RULES))
        rule = RULES[rule_lab]
    if abs(cap - al.CAP) > 1e-9 or k != al.MAX_REBAL:
        st.info("Paramètres différents de la consigne (20 %, 10 réallocations) : résultats indicatifs.")

    sol = _solve(cap, k, rule)
    static = _solve(cap, 0, rule)
    _, ex_w = al.load_example()
    ports = {"opt": sol.weights, "static": static.weights, "ex": ex_w, "ew": al.equal_weight(px)}
    navs, segs = {}, {}
    for key, w in ports.items():
        navs[key], segs[key] = al.backtest(px, w, cost_bps=cost if key in ("opt", "ex") else 0.0)
    mets = {key: al.metrics(nav, px) for key, nav in navs.items()}
    m, e = mets["opt"], mets["ew"]

    # ------------------------------------------------------------------ Indicateurs clés
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
    kc[4].metric("Réallocations utilisées", f"{sol.n_rebal} / {k}",
                 help=f"{n_used} sous-jacents différents détenus au moins une fois sur la période.")

    # ------------------------------------------------------------------ Track record
    fig = go.Figure()
    for key in ["ew", "ex", "static", "opt"]:
        fig.add_trace(go.Scatter(x=navs[key].index, y=navs[key].values, name=NAMES[key], mode="lines",
                                 line=dict(color=COLORS[key], width=2.6 if key == "opt" else 1.8),
                                 hovertemplate="%{x|%d/%m/%Y} : %{y:.1f}<extra>" + NAMES[key] + "</extra>"))
    for i, d in enumerate(sol.dates[1:], start=1):
        fig.add_vline(x=d, line=dict(color=C_PALE, width=1, dash="dot"))
        fig.add_annotation(x=d, y=1.0, yref="paper", text=f"R{i}", showarrow=False, yanchor="bottom",
                           font=dict(size=10, color=C_NAVY))
    fig.update_layout(template=TEMPLATE, height=470, title="Valeur liquidative (base 100) et dates de réallocation (R1 à "
                      f"R{sol.n_rebal})", yaxis_title="VL", xaxis_title="", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.12), margin=dict(t=70))
    show(fig)

    comp = pd.DataFrame({NAMES[kk]: v for kk, v in mets.items()})
    pct_rows = [r for r in comp.index if not r.startswith("Ratio")]
    fmt = {r: (lambda x: _fmt_pct(x)) for r in pct_rows}
    disp = comp.copy().astype(object)
    for r in comp.index:
        disp.loc[r] = [fmt[r](x) if r in fmt else f"{x:.2f}".replace(".", ",") for x in comp.loc[r]]
    turnover = segs["opt"]["Rotation"].sum()
    disp.loc["Rotation cumulée (achats + ventes)"] = [
        _fmt_pct(segs[kk]["Rotation"].sum(), 0) if segs[kk]["Rotation"].notna().any() else "-" for kk in mets]
    st.dataframe(disp, width="stretch")
    st.caption("Volatilité : rendements des jours de cotation × √252. Sharpe calculé avec un taux sans risque de 0 %. "
               "Statique optimal : meilleur portefeuille sans réallocation, mêmes règles. "
               + (f"Frais de {cost:.0f} pb déduits pour le dynamique et l'exemple." if cost else "Performances brutes de frais."))

    # ------------------------------------------------------------------ Allocation dans le temps
    st.markdown("#### Qui est détenu, et quand")
    bounds = sol.dates + [al.END]
    p = al.period_prices(px)
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
                      title="Périodes de détention (20 % chacun) ; couleur = performance du sous-jacent sur la période")
    for d in sol.dates[1:]:
        fig.add_vline(x=d, line=dict(color="#C5CED9", width=1, dash="dot"))
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
    per_disp = per.copy()
    for c_ in ["Portefeuille", "Équipondéré (29)"]:
        per_disp[c_] = per_disp[c_].map(lambda x: f"{x * 100:+.1f} %".replace(".", ","))
    for c_ in ["Début", "Fin"]:
        per_disp[c_] = pd.to_datetime(per_disp[c_]).dt.strftime("%d/%m/%Y")
    st.dataframe(per_disp, hide_index=True, width="stretch", height=36 * (len(per_disp) + 1) + 4)

    # ------------------------------------------------------------------ Valeur des réallocations
    st.markdown("#### Combien rapporte chaque réallocation ?")
    fr, oracle = _frontier(cap, rule)
    c1, c2 = st.columns([3, 2])
    with c1:
        colors = [C_NAVY if i <= al.MAX_REBAL else C_PALE for i in fr.index]
        fig = go.Figure(go.Bar(x=fr.index, y=fr.values, marker_color=colors,
                               hovertemplate="%{x} réallocation(s) au plus : %{y:+.1%}<extra></extra>"))
        if STRICT.exists() and abs(cap - al.CAP) < 1e-9:
            sv = pd.read_csv(STRICT)
            v = float(np.prod(1 + sv["perf"]) - 1)
            fig.add_hline(y=v, line=dict(color=C_TEAL, dash="dash"),
                          annotation_text=f"Plafond respecté chaque jour, 10 réallocations : {v:+.0%}",
                          annotation_position="top left")
        fig.update_layout(template=TEMPLATE, height=380, yaxis_tickformat="+.0%", xaxis_title="Nombre maximal de "
                          "réallocations", title="Meilleure performance atteignable (bleu foncé : dans la consigne)")
        show(fig, c1)
    with c2:
        gains = fr.diff().dropna()
        chk = _checks(cap, rule)
        st.markdown(f"""
* **Sans réallocation**, le meilleur portefeuille ne fait que {_fmt_pct(fr[0], 0)} : choisir les bons moments compte
  plus que choisir les bons sous-jacents une fois pour toutes.
* **Chaque réallocation** rapporte : de {gains.iloc[:5].min() * 100:.0f} à {gains.iloc[:5].max() * 100:.0f} points pour
  les 5 premières, environ {gains.iloc[5:10].mean() * 100:.0f} points pour les suivantes. Le gain marginal baisse mais
  ne s'annule pas : la limite de 10 réallocations est très contraignante.
* **Borne théorique** : en réallouant chaque jour vers les 5 meilleurs du lendemain, on atteindrait
  {_fmt_pct(oracle, 0)}. L'écart mesure tout ce que la consigne interdit.
* **Contrôle de l'optimum** : recherche exhaustive pour 1 et 2 réallocations ({_fmt_pct(chk['exh1'])} et
  {_fmt_pct(chk['exh2'])}, identiques à l'algorithme) ; 200 000 jeux de 10 dates tirés au hasard parmi
  {chk['n_dates']} dates possibles : meilleur tirage {_fmt_pct(chk['rand10'], 0)}, contre
  {_fmt_pct(fr[min(10, fr.index.max())], 0)} pour l'algorithme.
""")

    # ------------------------------------------------------------------ Méthode et limites
    best_tickers = per["Sous-jacents détenus (20 % chacun)"].str.split(", ").explode().value_counts()
    perf0 = al.backtest(px, sol.weights)[0].iloc[-1] / 100 - 1
    perf10 = al.backtest(px, sol.weights, cost_bps=10)[0].iloc[-1] / 100 - 1
    recap([
        f"<b>Résultat</b> : {_fmt_pct(m['Performance totale'], 0)} du 01/01 au 30/10/2026 (VL {navs['opt'].iloc[-1]:.0f}), "
        f"contre {_fmt_pct(e['Performance totale'])} pour l'équipondéré et {_fmt_pct(mets['ex']['Performance totale'])} "
        f"pour l'allocation d'exemple du fichier.",
        f"<b>Risque</b> : volatilité {_fmt_pct(m['Volatilité annualisée'])} (équipondéré {_fmt_pct(e['Volatilité annualisée'])}),"
        f" perte maximale {_fmt_pct(m['Perte maximale (max drawdown)'])} : concentrer sur 5 lignes augmente la volatilité,"
        " mais l'optimum ex post évite les grandes baisses.",
        f"<b>Rotation</b> : {sol.n_rebal} réallocations, presque toutes complètes (rotation cumulée "
        f"{_fmt_pct(turnover, 0)}) ; sous-jacents les plus souvent retenus : "
        + ", ".join(f"{t} ({n}×)" for t, n in best_tickers.head(4).items()) + ".",
        f"<b>Frais</b> : à 10 pb du montant échangé, la performance passerait de {_fmt_pct(perf0, 0)} à "
        f"{_fmt_pct(perf10, 0)} : les frais ne changent pas la conclusion.",
    ], methode=[
        "Entre deux dates, le portefeuille est conservé tel quel (les poids dérivent avec les prix) ; le plafond de 20 % "
        "s'applique à chaque date d'allocation, comme dans les règles d'indices.",
        "Sur une période donnée, la valeur finale est linéaire dans les poids : l'optimum met 20 % sur les 5 sous-jacents "
        "qui montent le plus (vérifié par programmation linéaire).",
        "Les dates sont choisies par programmation dynamique sur (date, nombre de réallocations faites) : toutes les "
        "combinaisons sont couvertes implicitement, l'optimum est global et se calcule en quelques millisecondes.",
        "Réallocation permise les jours où toutes les places sont ouvertes, pour ne pas trader sur un prix figé ; "
        "le résultat est le même avec tous les jours calendaires.",
    ])
    drift = al.max_drift(px, sol.weights)
    strict_txt = ""
    if STRICT.exists():
        sv = pd.read_csv(STRICT)
        strict_txt = (f" Si le plafond doit être respecté chaque jour, l'optimum (programme linéaire par période, mêmes "
                      f"dates) donne encore {_fmt_pct(float(np.prod(1 + sv['perf']) - 1), 0)}, avec "
                      f"{int(sv['nb_lignes'].median())} lignes par période.")
    st.markdown(f"""
**Limites**
- Optimum **ex post** : il suppose de connaître les prix futurs. C'est la borne haute permise par les contraintes,
  pas une stratégie investissable ; en temps réel, il faudrait des signaux (momentum, valorisation…) et un backtest
  hors échantillon.
- Les poids dérivent entre deux réallocations : un sous-jacent peut dépasser 20 % en cours de période (au plus
  {_fmt_pct(drift[0])}, {drift[1]} le {drift[2]:%d/%m/%Y}).{strict_txt}
- Indices de prix (dividendes non inclus) ; 3 sous-jacents cotés en dollars agrégés sans conversion de change.
- Exécution supposée au cours de clôture, sans contrainte de liquidité ; frais non déduits par défaut.
""")
    c1, c2 = st.columns(2)
    if EXCEL.exists():
        c1.download_button("Télécharger le classeur Excel (track record en formules, paramètres de la consigne)",
                           EXCEL.read_bytes(), EXCEL.name,
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    tr = pd.DataFrame({NAMES[kk]: v for kk, v in navs.items()})
    c2.download_button("Télécharger le track record (CSV)", tr.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
                       "track_record_allocation_dynamique.csv", "text/csv")
