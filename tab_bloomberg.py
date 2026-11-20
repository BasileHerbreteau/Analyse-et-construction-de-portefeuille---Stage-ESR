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
C_NAVY, C_BLUE, C_TEAL, C_GREY, C_LIGHT, C_RED = du.C_NAVY, du.C_BLUE, du.C_TEAL, du.C_GREY, du.C_LIGHT, du.C_RED
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

    st.caption(f"Information externe au fichier Excel, sans effet sur les onglets 1 à 8. Prix hebdomadaires en dollars "
               f"(Bloomberg, {span}) ; marché : MSCI World.")

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
    st.caption("Estimée : 3 rendements disjoints du fichier, rapprochés de la médiane de la sous-industrie (onglet 2). "
               "Mesurée : rendements hebdomadaires annualisés (√52). Au-dessus de la diagonale, risque sous-estimé "
               "par le fichier.")

    # ------------------------------------------------------------------ bêtas
    st.subheader("Bêtas contre le MSCI World")
    rho_m = tick.map(b["corr_mxwo"]).to_numpy(float)
    db = full[["ticker", "name", "country", "subindustry"]].assign(
        **{"Bêta mesuré": beta, "Bêta ajusté": tick.map(b["beta_usd_adj"]).to_numpy(float),
           "Bêta Bloomberg (indice local)": tick.map(b["beta_local_bbg_raw"]).to_numpy(float),
           "Corrélation avec le MSCI World": rho_m, "Part de variance expliquée par le marché (R²)": rho_m ** 2,
           M_: vol_h.to_numpy(float), "Volatilité spécifique (%)": vol_h.to_numpy(float) * np.sqrt(1 - rho_m ** 2),
           E_: full["vol"].to_numpy(float)})
    med = db.groupby("subindustry")["Bêta mesuré"].median().sort_values()
    fig = go.Figure(go.Bar(x=med.values, y=med.index, orientation="h", marker_color=C_BLUE,
                           text=[f"{v:.2f}" for v in med.values], textposition="outside"))
    fig.add_vline(x=1, line=dict(color=C_GREY, dash="dash"), annotation_text="Marché (bêta 1)", annotation_position="bottom right")
    fig.update_layout(template=TEMPLATE, height=max(420, 24 * len(med) + 120), xaxis_title="Bêta mesuré médian",
                      title="Bêta médian par sous-industrie")
    show(fig)
    with st.expander("Bêtas et volatilités de chaque titre"):
        st.dataframe(db.rename(columns={"ticker": "Ticker", "name": "Société", "country": "Pays",
                                        "subindustry": "Sous-industrie"})
                     .sort_values("Bêta mesuré", ascending=False).style.format(precision=2),
                     hide_index=True, width="stretch", height=420)
        st.caption("Bêta mesuré : régression des rendements hebdomadaires en dollars sur le MSCI World. Ajusté : "
                   "0,67 × mesuré + 0,33 (Blume). Bêta Bloomberg : contre l'indice local, non comparable entre pays. "
                   "Volatilité spécifique : écart-type annualisé des résidus de la régression, σ × √(1 − R²).")
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
    st.caption("Moyenne des corrélations de rendements hebdomadaires par groupe de paires. Hypothèse supérieure à la "
               "mesure : marge de prudence assumée (onglet 7).")

    # ------------------------------------------------------------------ portefeuilles
    st.subheader("Portefeuilles reconstruits avec les bêtas")
    st.latex(r"\Sigma = \sigma_m^2\,\beta\beta^\top + D, \qquad D_{ii} = \sigma_{\varepsilon,i}^2 = \sigma_i^2 - "
             r"\beta_i^2\sigma_m^2")
    st.caption(f"Modèle de marché à un facteur (Sharpe, 1963). σ_m = {sig_m:.1%} (MSCI World) ; σ_i, β_i et risque "
               "spécifique σ_ε mesurés sur l'historique. Rendement attendu : même formule, volatilité mesurée dans "
               "l'alpha. Sélection inchangée : quotas par sous-industrie, score > médiane, filtre ESG, plafond de 5 %.")
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
    st.caption("Tous les portefeuilles évalués avec le même modèle. Volatilité mesurée : covariance historique complète, "
               "sans modèle. Limite : le modèle à un facteur ignore les co-mouvements propres à une sous-industrie, "
               "effet atténué par les quotas.")

    pick = st.radio("Portefeuille", list(var), horizontal=True, key="bbg_pick")
    lbl = pick[0].lower() + pick[1:]
    a_, c_ = W[f"Portefeuille {lbl} (onglet 8)"] * 100, W[f"Portefeuille {lbl} (bêtas)"] * 100
    dw = pd.DataFrame({"ticker": tick, "name": full["name"], "avant": a_, "apres": c_, "delta": c_ - a_})
    held = dw[(dw["avant"] > 1e-4) | (dw["apres"] > 1e-4)]
    chg = held[held["delta"].abs() >= 0.1].sort_values("delta")
    same = held[held["delta"].abs() < 0.1]
    statut = np.where(chg["avant"] < 1e-4, "entrée", np.where(chg["apres"] < 1e-4, "sortie", ""))
    txt = [f"{a:.1f} → {c:.1f} %" + (f" ({s_})" if s_ else "") for a, c, s_ in zip(chg["avant"], chg["apres"], statut)]
    up = (chg["delta"] > 0).to_numpy()
    fig = go.Figure()
    for m_, nm_, col_ in ((up, "Poids en hausse", C_TEAL), (~up, "Poids en baisse", C_RED)):
        fig.add_trace(go.Bar(
            y=chg["ticker"][m_], x=chg["delta"][m_], orientation="h", name=nm_, marker_color=col_,
            text=[t for t, k_ in zip(txt, m_) if k_], textposition="outside", cliponaxis=False,
            customdata=np.c_[chg["name"][m_], chg["avant"][m_], chg["apres"][m_]],
            hovertemplate="<b>%{y}</b> %{customdata[0]}<br>Onglet 8 : %{customdata[1]:.1f} %<br>"
                          "Avec les bêtas : %{customdata[2]:.1f} %<br>Variation : %{x:+.1f} pt<extra></extra>"))
    span_x = float(chg["delta"].abs().max()) if len(chg) else 1.0
    fig.add_vline(x=0, line=dict(color=C_GREY, width=1))
    fig.update_layout(template=TEMPLATE, barmode="relative", bargap=0.35, height=max(320, 30 * len(chg) + 150),
                      title=f"Portefeuille {lbl} : lignes dont le poids change avec les bêtas",
                      xaxis=dict(title="Variation de poids (points de %)", range=[-span_x * 1.75, span_x * 1.75],
                                 zeroline=False),
                      yaxis=dict(categoryorder="array", categoryarray=chg["ticker"].tolist()),
                      legend=dict(orientation="h", y=-0.18))
    show(fig)
    st.caption(f"{len(chg)} lignes modifiées sur {len(held)} détenues dans l'une ou l'autre version. "
               f"{len(same)} lignes inchangées (écart < 0,1 pt), non représentées : "
               f"{', '.join(same.sort_values('apres', ascending=False)['ticker'])}.")

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
                   + f". Sharpe ex ante {tc_['Sharpe ex ante']:.2f} contre {ta_['Sharpe ex ante']:.2f} pour la version "
                   "de l'onglet 8 sous le même modèle (variante optimisée sur ce modèle) ; bêta "
                   f"{tc_['Bêta mesuré']:.2f} contre {ta_['Bêta mesuré']:.2f} "
                   f"(équipondéré {e_b:.2f}).")
    pts.append("Le cœur de la sélection tient sur données réelles : l'historique modifie surtout les poids et ajoute "
               "quelques lignes." if all(
                   len(set(tick[W[f'Portefeuille {n[0].lower() + n[1:]} (onglet 8)'] > 1e-6])
                       & set(tick[var[n] > 1e-6])) >= 0.75 * (W[f'Portefeuille {n[0].lower() + n[1:]} (onglet 8)']
                                                               > 1e-6).sum() for n in var)
               else "L'historique modifie sensiblement la composition : les hypothèses du fichier pèsent sur la sélection.")
    recap(pts, ["Bêta mesuré = cov(r_i, r_m) / var(r_m) sur les rendements hebdomadaires en dollars.",
                "Modèle à un facteur : Σ = σ_m² ββᵀ + D, D diagonale des variances spécifiques (Sharpe, 1963).",
                "Pondération : maximisation du Sharpe ex ante, 0 ≤ w ≤ 5 %, Σw = 1, sur les 30 titres présélectionnés."])
