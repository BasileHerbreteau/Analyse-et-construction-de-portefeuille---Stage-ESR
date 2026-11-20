"""Onglet « Bonus - Historique Bloomberg ».

Les onglets 1 à 8 n'utilisent pas d'historique de prix : la volatilité y est estimée à partir du fichier Excel et les
corrélations sont supposées. Cet onglet confronte ces hypothèses à 104 semaines de prix Bloomberg en dollars
(jusqu'au 13/11/2026), puis reconstruit les portefeuilles avec un modèle à un facteur fondé sur les bêtas.
Rien ici ne modifie les onglets principaux.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px_
import plotly.graph_objects as go
import streamlit as st

import data_utils as du
import portfolio as pf
import scoring as sc

TEMPLATE = "plotly_white"
C_NAVY, C_BLUE, C_TEAL, C_GREY, C_LIGHT = du.C_NAVY, du.C_BLUE, du.C_TEAL, du.C_GREY, du.C_LIGHT
E_, M_ = "Volatilité estimée (%)", "Volatilité mesurée (%)"


@st.cache_data(show_spinner="Optimisation avec le modèle à un facteur…")
def _build_1f(d: pd.DataFrame, eligible: tuple, n_total: int, cap: float, cov_flat: tuple,
              rescue: tuple | None = None) -> pd.Series:
    """Même sélection que l'onglet 8 (quotas, score, filtre ESG), pondération au Sharpe maximal avec la covariance
    du modèle à un facteur."""
    n = len(d)
    cov = np.array(cov_flat).reshape(n, n)
    res = None if rescue is None else pd.Series(rescue, index=d.index)
    sel = pf.select(d, n_total, pd.Series(eligible, index=d.index), rescue=res)
    pos = d.index.get_indexer(sel.index)
    w = pf.max_sharpe(sel["mu"].to_numpy(float), cov[np.ix_(pos, pos)], cap)
    out = pd.Series(0.0, index=d.index)
    out[sel.index] = w
    return out


def _blocks(corr: np.ndarray, sub: np.ndarray, sec: np.ndarray) -> list[tuple[float, int]]:
    iu = np.triu_indices(len(sub), 1)
    ss, se = (sub[:, None] == sub[None, :])[iu], (sec[:, None] == sec[None, :])[iu]
    c = corr[iu]
    return [(float(c[m].mean()), int(m.sum())) for m in (ss, se & ~ss, ~se)]


def render(ctx: dict, show, recap):
    full = ctx["full"]
    betas, px = du.load_betas(), du.load_weekly_prices()
    if (betas is None or px is None or not full["ticker"].isin(betas["ticker"]).all()
            or not full["ticker"].isin(px.columns).all()):
        st.info("Données Bloomberg absentes : fichiers data/bloomberg_betas.csv et data/prix_hebdo_usd.csv.")
        return
    b = betas.set_index("ticker")
    tick = full["ticker"]
    beta = tick.map(b["beta_usd_raw"]).to_numpy(float)
    vol_h = tick.map(b["vol_hist_usd"])
    ret = px.pct_change().dropna()
    R = ret[tick.tolist()].to_numpy(float)
    sig_m = float(ret["MXWO"].std() * np.sqrt(52))
    span = f"{len(ret)} semaines, du {px.index[0]:%d/%m/%Y} au {px.index[-1]:%d/%m/%Y}"

    st.caption(f"Onglet complémentaire. Les onglets 1 à 8 n'utilisent aucun historique de prix : la volatilité y est "
               f"estimée à partir du fichier Excel et les corrélations sont supposées. Ici, ces hypothèses sont "
               f"comparées aux prix hebdomadaires en dollars extraits de Bloomberg ({span}), puis les portefeuilles "
               f"sont reconstruits avec les bêtas. Rien dans cet onglet ne modifie les onglets principaux.")

    # ------------------------------------------------------------------ volatilité
    st.subheader("Volatilité : estimation du fichier et mesure sur l'historique")
    dv = pd.DataFrame({"ticker": tick, "name": full["name"], "sector": full["sector"],
                       E_: full["vol"], M_: vol_h.to_numpy(float)}, index=full.index)
    sp = dv[E_].corr(dv[M_], method="spearman")
    gap = (dv[E_] - dv[M_]).abs().median()
    k = st.columns(4)
    k[0].metric("Médiane estimée", f"{dv[E_].median():.1f} %")
    k[1].metric("Médiane mesurée", f"{dv[M_].median():.1f} %")
    k[2].metric("Corrélation des rangs", f"{sp:.2f}")
    k[3].metric("Écart absolu médian", f"{gap:.1f} pt")
    fig = px_.scatter(dv, x=E_, y=M_, color="sector", hover_name="ticker", color_discrete_map=du.SECTOR_COLORS,
                      template=TEMPLATE)
    lim = [0, float(max(dv[E_].max(), dv[M_].max())) * 1.05]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Égalité", line=dict(color=C_GREY, dash="dash")))
    fig.update_layout(height=480, title="Volatilité annualisée par titre : estimée (fichier) contre mesurée (Bloomberg)",
                      legend=dict(orientation="h", y=-0.18))
    show(fig)
    over = dv.assign(d=dv[E_] - dv[M_]).sort_values("d")
    st.caption("Estimée : écart-type annualisé de trois rendements du fichier sur des périodes disjointes, rapproché de "
               "la médiane de la sous-industrie (onglet 2). Mesurée : écart-type des rendements hebdomadaires × √52. "
               "Au-dessus de la diagonale, le fichier sous-estime le risque du titre ; en dessous, il le surestime.")

    # ------------------------------------------------------------------ bêtas
    st.subheader("Bêtas contre le MSCI World")
    st.markdown(
        "Le bêta mesure la sensibilité d'un titre aux mouvements du marché. Quand le MSCI World varie de 1 %, un titre "
        "de bêta 1,40 varie en moyenne de 1,40 % dans le même sens : il amplifie les hausses comme les baisses de 40 %. "
        "Le bêta ne mesure que le risque de marché : bêta = corrélation avec le marché × volatilité du titre ÷ "
        "volatilité du marché. Un titre très volatil mais peu lié au marché peut donc avoir un bêta modéré.")
    db = full[["ticker", "name", "country", "subindustry"]].assign(
        **{"Bêta mesuré": beta, "Bêta ajusté": tick.map(b["beta_usd_adj"]).to_numpy(float),
           "Bêta Bloomberg (indice local)": tick.map(b["beta_local_bbg_raw"]).to_numpy(float),
           "Corrélation avec le MSCI World": tick.map(b["corr_mxwo"]).to_numpy(float), M_: vol_h.to_numpy(float),
           E_: full["vol"].to_numpy(float)})
    med = db.groupby("subindustry")["Bêta mesuré"].median().sort_values()
    fig = go.Figure(go.Bar(x=med.values, y=med.index, orientation="h", marker_color=C_BLUE,
                           text=[f"{v:.2f}" for v in med.values], textposition="outside"))
    fig.add_vline(x=1, line=dict(color=C_GREY, dash="dash"), annotation_text="Marché (bêta 1)")
    fig.update_layout(template=TEMPLATE, height=max(420, 24 * len(med) + 120), xaxis_title="Bêta mesuré médian",
                      title="Bêta médian par sous-industrie")
    show(fig)
    with st.expander("Bêtas et volatilités de chaque titre"):
        st.dataframe(db.rename(columns={"ticker": "Ticker", "name": "Société", "country": "Pays",
                                        "subindustry": "Sous-industrie"})
                     .sort_values("Bêta mesuré", ascending=False).style.format(precision=2),
                     hide_index=True, width="stretch", height=420)
        st.caption("Bêta mesuré : pente de la régression des rendements hebdomadaires en dollars du titre sur ceux du "
                   "MSCI World. Ajusté : 0,67 × mesuré + 0,33 (méthode de Blume, utilisée par Bloomberg), plus adapté "
                   "à une prévision car les bêtas extrêmes se rapprochent de 1 avec le temps. Bêta Bloomberg par "
                   "défaut : mesuré contre l'indice local du titre, non comparable d'un pays à l'autre.")
        c1, c2 = st.columns(2)
        c1.download_button("Télécharger les bêtas (CSV)", betas.to_csv(index=False).encode(), "betas_bloomberg.csv",
                           "text/csv")
        c2.download_button("Télécharger les prix hebdomadaires (CSV)", px.to_csv().encode(), "prix_hebdo_usd.csv",
                           "text/csv")

    # ------------------------------------------------------------------ corrélations
    st.subheader("Corrélations : hypothèse et mesure")
    blk = _blocks(np.corrcoef(R, rowvar=False), full["subindustry"].to_numpy(object), full["sector"].to_numpy(object))
    rho = ctx["RHO"]
    st.dataframe(pd.DataFrame({
        "Paire de titres": ["Même sous-industrie", "Même secteur, autre sous-industrie", "Secteurs différents"],
        "Hypothèse des onglets 7 et 8": list(rho), "Mesurée (Bloomberg)": [m for m, _ in blk],
        "Nombre de paires": [n for _, n in blk]}).style.format(precision=2), hide_index=True, width="stretch")
    st.caption("Corrélation moyenne des rendements hebdomadaires sur chaque groupe de paires. Les niveaux supposés sont "
               "plus élevés que les niveaux mesurés : c'est une marge de prudence, les corrélations augmentant en "
               "marché baissier.")

    # ------------------------------------------------------------------ portefeuilles
    st.subheader("Portefeuilles reconstruits avec les bêtas")
    st.markdown("Si l'historique avait été disponible dès le départ, le risque aurait pu être modélisé avec un "
                "**modèle à un facteur** : chaque titre est exposé au marché selon son bêta, le reste de son risque lui "
                "est propre.")
    st.latex(r"\Sigma = \sigma_m^2\,\beta\beta^\top + D, \qquad D_{ii} = \sigma_i^2 - \beta_i^2\sigma_m^2")
    st.markdown(f"σ_m = {sig_m:.1%} (volatilité mesurée du MSCI World), σ_i = volatilité mesurée du titre, β = bêta "
                "mesuré. Le rendement attendu garde la même formule, avec la volatilité mesurée dans l'alpha "
                "(μ = prime de risque + IC × σ × score standardisé). La sélection est inchangée : 30 places réparties "
                "par sous-industrie, score supérieur à la médiane, filtre ESG, plafond de 5 %. Seuls les paramètres de "
                "risque changent.")
    vol_dec = vol_h.to_numpy(float) / 100
    S1 = sig_m ** 2 * np.outer(beta, beta) + np.diag(np.maximum(vol_dec ** 2 - beta ** 2 * sig_m ** 2, 1e-8))
    S_emp = np.cov(R, rowvar=False) * 52
    mu_b = sc.expected_return(full, full["Score alpha"], vol_h, ctx["rf"], ctx["lam"], ctx["ic"])["mu"]
    d = full.copy()
    d["mu"] = mu_b
    var = {"Sans ESG": _build_1f(d, tuple(ctx["elig_all"]), ctx["N_TOTAL"], ctx["CAP"], tuple(S1.ravel()))}
    if ctx.get("elig_esg") is not None:
        var["Avec filtre ESG"] = _build_1f(d, tuple(ctx["elig_esg"]), ctx["N_TOTAL"], ctx["CAP"], tuple(S1.ravel()),
                                           tuple(ctx["rescue"]))
    ports = ctx["ports"]
    rows, W = [], {}
    W["Équipondéré univers (99 titres)"] = ctx["EW"].reindex(full.index).fillna(0.0)
    for nm in var:
        lbl = nm[0].lower() + nm[1:]
        W[f"Portefeuille {lbl} (onglet 8)"] = ports[nm]["Poids"].reindex(full.index).fillna(0.0)
        W[f"Portefeuille {lbl} (bêtas)"] = var[nm]
    if ctx.get("W_CAP") is not None:
        W[ctx["CAPW_NAME"]] = ctx["W_CAP"].reindex(full.index).fillna(0.0)
    mu_v = mu_b.to_numpy(float)
    for nm, w in W.items():
        wv = w.to_numpy(float)
        r_, v_ = float(wv @ mu_v), float(np.sqrt(wv @ S1 @ wv) * 100)
        rows.append({"Portefeuille": nm, "Lignes": int((wv > 1e-6).sum()), "Rendement attendu (%)": r_,
                     "Volatilité, modèle à un facteur (%)": v_, "Sharpe ex ante": r_ / v_,
                     "Volatilité mesurée (%)": float(np.sqrt(wv @ S_emp @ wv) * 100),
                     "Bêta mesuré": float(wv @ beta)})
    tab = pd.DataFrame(rows).set_index("Portefeuille")
    st.dataframe(tab.style.format(precision=2).format({"Lignes": "{:.0f}"}), width="stretch")
    st.caption("Tous les portefeuilles sont évalués avec le même modèle (rendements attendus avec la volatilité mesurée, "
               "covariance à un facteur) pour être comparables. Volatilité mesurée : covariance historique complète "
               "des 104 semaines, sans modèle. Limite du modèle à un facteur : il ne capte que le lien commun au "
               "marché et sous-estime les co-mouvements propres à une sous-industrie ; les quotas par sous-industrie "
               "limitent cet effet.")

    pick = st.radio("Comparer les poids", list(var), horizontal=True, key="bbg_pick")
    lbl = pick[0].lower() + pick[1:]
    a_, c_ = W[f"Portefeuille {lbl} (onglet 8)"], W[f"Portefeuille {lbl} (bêtas)"]
    keep = (a_ > 1e-6) | (c_ > 1e-6)
    cmp_ = pd.DataFrame({"ticker": tick[keep], "Onglet 8": a_[keep] * 100, "Avec les bêtas": c_[keep] * 100})
    cmp_ = cmp_.sort_values(["Onglet 8", "Avec les bêtas"])
    fig = go.Figure([go.Bar(y=cmp_["ticker"], x=cmp_["Onglet 8"], name="Onglet 8 (hypothèses du fichier)",
                            orientation="h", marker_color=C_LIGHT),
                     go.Bar(y=cmp_["ticker"], x=cmp_["Avec les bêtas"], name="Avec les bêtas (historique Bloomberg)",
                            orientation="h", marker_color=C_NAVY)])
    fig.update_layout(barmode="group", template=TEMPLATE, height=max(500, 22 * len(cmp_) + 140), xaxis_title="Poids (%)",
                      title=f"Poids du portefeuille {lbl}", legend=dict(orientation="h", y=-0.12))
    show(fig)

    pts = [f"<b>Volatilité</b> : médiane estimée {dv[E_].median():.1f} % contre {dv[M_].median():.1f} % mesurée ; "
           f"corrélation des rangs {sp:.2f}. L'estimation à partir de trois rendements donne le bon niveau de risque"
           + (", mais classe moins bien les titres entre eux." if sp < 0.8 else " et classe correctement les titres."),
           f"<b>Corrélations</b> : mesurées {blk[0][0]:.2f} / {blk[1][0]:.2f} / {blk[2][0]:.2f} contre "
           f"{rho[0]:.2f} / {rho[1]:.2f} / {rho[2]:.2f} supposées (sous-industrie / secteur / autres) : l'hypothèse "
           "des onglets 7 et 8 est " + ("prudente." if all(r >= m for r, (m, _) in zip(rho, blk)) else "à revoir.")]
    e_b = tab.iloc[0]["Bêta mesuré"]
    for nm in var:
        lbl = nm[0].lower() + nm[1:]
        a_, c_ = W[f"Portefeuille {lbl} (onglet 8)"], W[f"Portefeuille {lbl} (bêtas)"]
        A, C = set(tick[a_ > 1e-6]), set(tick[c_ > 1e-6])
        ta_, tc_ = tab.loc[f"Portefeuille {lbl} (onglet 8)"], tab.loc[f"Portefeuille {lbl} (bêtas)"]
        pts.append(f"<b>Portefeuille {lbl}</b> : avec les bêtas, {len(A & C)} des {len(A)} lignes sont conservées"
                   + (f" ; sortent {', '.join(sorted(A - C))}" if A - C else "")
                   + (f" ; entrent {', '.join(sorted(C - A))} ({c_[tick.isin(C - A)].sum():.0%} du portefeuille)"
                      if C - A else "")
                   + f". Sharpe ex ante {tc_['Sharpe ex ante']:.2f} contre {ta_['Sharpe ex ante']:.2f} pour le "
                   "portefeuille de l'onglet 8 évalué avec le même modèle (écart attendu, la variante étant optimisée "
                   f"pour ce modèle) ; bêta {tc_['Bêta mesuré']:.2f} contre {ta_['Bêta mesuré']:.2f} "
                   f"(équipondéré {e_b:.2f}).")
    pts.append("Le cœur de la sélection tient sur données réelles : l'historique change surtout les poids et ajoute "
               "quelques lignes, sans remettre en cause les titres retenus dans l'onglet 8." if all(
                   len(set(tick[W[f'Portefeuille {n[0].lower() + n[1:]} (onglet 8)'] > 1e-6])
                       & set(tick[var[n] > 1e-6])) >= 0.75 * (W[f'Portefeuille {n[0].lower() + n[1:]} (onglet 8)']
                                                               > 1e-6).sum() for n in var)
               else "L'historique modifie sensiblement la composition : les hypothèses du fichier pèsent sur la sélection.")
    recap(pts, ["Bêta mesuré = cov(r_i, r_m) / var(r_m) sur les rendements hebdomadaires en dollars.",
                "Modèle à un facteur : Σ = σ_m² ββᵀ + D, D diagonale des risques propres (Sharpe, 1963).",
                "Pondération : maximisation du Sharpe ex ante, 0 ≤ w ≤ 5 %, Σw = 1, sur les 30 titres présélectionnés."])
