"""Application Streamlit : analyse de l'univers « universe of stock 261113 ».

Lecture « en entonnoir » : on part de la composition de l'univers, puis d'une
synthèse globale, avant d'affiner par secteur, par facteur, puis par titre.

Lancer en local :  streamlit run app.py
"""
from __future__ import annotations

import base64
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as an
import data_utils as du
import portfolio as pf
import scoring as sc
from dictionary import COLUMNS, DERIVED_DOC

st.set_page_config(page_title="Univers d'actions – Analyse et portefeuille", page_icon="📊", layout="wide")

# Logo : fichier local assets/logo_amundi.png s'il existe (à déposer dans le dépôt), sinon image du site Amundi
LOGO_FILE = Path(__file__).parent / "assets" / "logo_amundi.png"
if LOGO_FILE.exists():
    LOGO_URL = "data:image/png;base64," + base64.b64encode(LOGO_FILE.read_bytes()).decode()
    LOGO_SRC = str(LOGO_FILE)
else:
    LOGO_URL = LOGO_SRC = "https://static.amundi.com/web-assets/awf-assets/corporate/logo.png"
C_NAVY, C_BLUE, C_LIGHT, C_TEAL, C_PALE, C_RED, C_GREY = (du.C_NAVY, du.C_BLUE, du.C_LIGHT, du.C_TEAL,
                                                            du.C_PALE, du.C_RED, du.C_GREY)
C_ACCENT = C_TEAL
st.logo(LOGO_SRC, size="large", link="https://about.amundi.com/")
st.markdown(f"""
<style>
h1, h2, h3 {{ color: {C_NAVY}; }}
div[data-testid="stMetricValue"] {{ color: {C_NAVY}; }}
div[data-testid="stMetric"] {{ background: #F3F7FC; border-left: 4px solid {C_BLUE};
    padding: .5rem .8rem; border-radius: 6px; }}
button[role="tab"] p {{ font-weight: 600; }}
.amundi-band {{ background: linear-gradient(90deg, {C_NAVY} 0%, {C_BLUE} 70%, {C_LIGHT} 100%);
    color: white; border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 1.2rem; }}
.amundi-band img {{ height: 46px; background: white; border-radius: 6px; padding: 6px 10px; }}
.amundi-band h1 {{ color: white; margin: 0; font-size: 1.9rem; }}
.amundi-band p {{ color: #DCE9F7; margin: .2rem 0 0 0; }}
</style>""", unsafe_allow_html=True)

PLOT_CFG = {"displaylogo": False}
TEMPLATE = "plotly_white"
SECTOR_COLORS = du.SECTOR_COLORS
FR = {"sector": "Secteur", "subindustry": "Sous-industrie", "country": "Pays", "ticker": "Ticker",
      "variable": "Variable", "value": "Valeur"}
GROUP_FR = {"sector": "Secteur", "subindustry": "Sous-industrie", "country": "Pays"}


def show(fig, container=st):
    """Affiche une figure Plotly avec des intitulés en français."""
    lt = fig.layout.legend.title.text
    if lt in FR:
        fig.update_layout(legend_title_text=FR[lt])
    fig.for_each_xaxis(lambda a: a.update(title_text=FR.get(a.title.text, a.title.text)))
    fig.for_each_yaxis(lambda a: a.update(title_text=FR.get(a.title.text, a.title.text)))
    container.plotly_chart(fig, width="stretch", config=PLOT_CFG)


def lab(c: str) -> str:
    return du.LABELS.get(c, c)


def intro(texte: str):
    st.markdown(f"<div style='border-left:4px solid #0062A3;padding:.4rem .9rem;margin-bottom:1rem;"
                f"color:#444'>{texte}</div>", unsafe_allow_html=True)


def recap(points: list[str], notions: list[str]):
    """Encadré de synthèse : points clés chiffrés + notions mathématiques / financières utilisées."""
    pts = "".join(f"<li>{x}</li>" for x in points)
    nts = "".join(f"<li>{x}</li>" for x in notions)
    st.markdown(
        "<div style='background:#EEF4FB;border:1px solid #C9DBF0;border-radius:8px;padding:.8rem 1.1rem;"
        "margin:1.2rem 0 .4rem 0'>"
        f"<div style='font-weight:600;margin-bottom:.3rem'>📌 Points clés</div><ul style='margin:0 0 .5rem 0'>{pts}</ul>"
        f"<div style='font-weight:600;margin-bottom:.3rem'>📐 Notions utilisées</div><ul style='margin:0'>{nts}</ul>"
        "</div>", unsafe_allow_html=True)


def suite(texte: str):
    st.markdown(f"<div style='margin-top:1.5rem;color:#666;font-style:italic'>➡️ {texte}</div>",
                unsafe_allow_html=True)


@st.cache_data
def get_data():
    raw = du.load_raw()
    df = du.prepare(raw)
    esg = du.load_esg()
    if esg is not None:
        df = df.merge(esg[["ticker", "esg_score", "esg_scale", "esg_provider", "esg_date", "source_url"]],
                      on="ticker", how="left").rename(columns={"esg_score": "esg"})
    return raw, df, esg


@st.cache_data(show_spinner="Optimisation du portefeuille (Sharpe maximal)…")
def build_portfolio(d: pd.DataFrame, eligible: tuple, n_total: int, cap: float, rhos: tuple):
    sel = pf.select(d, n_total, pd.Series(eligible, index=d.index))
    R = an.block_correlation(sel, *rhos)
    cov = an.covariance(sel["vol"], R)
    w = pf.max_sharpe(sel["mu"].to_numpy(float), cov, cap)
    sel["Poids"] = w
    sel["Contribution au risque"] = pf.risk_contributions(w, cov)
    return sel


@st.cache_data(show_spinner="Calcul de la frontière efficiente…")
def frontier_cached(mu: tuple, cov_flat: tuple, n: int, cap: float):
    cov = np.array(cov_flat).reshape(n, n)
    return an.efficient_frontier(np.array(mu), cov, cap, n_points=22)


raw, df_full, esg = get_data()
HAS_ESG = esg is not None

# --------------------------------------------------------------------------- #
# Barre latérale
# --------------------------------------------------------------------------- #
st.sidebar.header("Filtres")
sectors = st.sidebar.multiselect("Secteur GICS", sorted(df_full.sector.unique()),
                                 default=sorted(df_full.sector.unique()))
sub_opts = sorted(df_full.loc[df_full.sector.isin(sectors), "subindustry"].unique())
subs = st.sidebar.multiselect("Sous-industrie", sub_opts, default=sub_opts)
countries = st.sidebar.multiselect("Pays", sorted(df_full.country.unique()),
                                   default=sorted(df_full.country.unique()))

st.sidebar.header("Valeurs extrêmes")
wins = st.sidebar.checkbox("Écrêter les extrêmes (winsorisation)", value=True,
                           help="Chaque variable est ramenée dans ses quantiles [p, 1−p]. "
                                "Aucune ligne n'est supprimée ; le tableau brut reste intact.")
p = st.sidebar.slider("Quantile d'écrêtage p", 0.0, 0.10, 0.02, 0.01, disabled=not wins)

with st.sidebar.expander("Paramètres risque / rendement", expanded=False):
    data_date = st.date_input("Date des données", dt.date(2026, 11, 13),
                              help="Déduite du nom du fichier (261113 = 2026-11-13). Sert à mesurer la durée du YTD.")
    vol_method = st.radio("Volatilité", ["multi", "amp1m"],
                          format_func={"multi": "Multi-périodes (recommandé)",
                                       "amp1m": "Amplitude 1 mois annualisée"}.get)
    alpha = st.slider("Poids de l'estimation individuelle (vs médiane de la sous-industrie)",
                      0.0, 1.0, 0.5, 0.1,
                      help="1 = estimation du titre seule ; 0 = médiane de sa sous-industrie. "
                           "0,5 réduit le bruit d'une estimation faite sur 3 observations.")
    rf = st.number_input("Taux sans risque (%)", 0.0, 10.0, 0.0, 0.25,
                         help="Absent du fichier : 0 par défaut. Il est retranché de l'upside (la prime de risque "
                              "est déjà un rendement en excès du sans-risque).")

with st.sidebar.expander("Critère agrégé & rendement attendu", expanded=False):
    z_level = st.radio("z-scores calculés par rapport à", ["subindustry", "sector", "univers"],
                       format_func={"subindustry": "la sous-industrie (pairs directs, recommandé)",
                                    "sector": "le secteur", "univers": "tout l'univers"}.get)
    k_shrink = st.slider("Rétrécissement vers le secteur (k, titres fictifs)", 0.0, 15.0, 5.0, 1.0,
                         help="Pour une sous-industrie de n titres, moyenne et variance = (n·groupe + k·secteur)/(n + k). "
                              "Évite des z-scores extrêmes calculés sur 2 ou 3 pairs.")
    lam = st.slider("Part de l'upside brut dans le niveau de rendement (λ)", 0.0, 1.0, 0.0, 0.05,
                    help="0 (recommandé) : le niveau vient de la prime de risque implicite et l'upside n'entre que par "
                         "son z-score (information relative). 0,5 : moyenne 50/50 prime / upside brut, qui reprend le "
                         "biais d'optimisme des objectifs de cours.")
    ic = st.slider("Coefficient d'information (IC) du score", 0.0, 0.30, 0.10, 0.01,
                   help="Corrélation supposée entre le score et le rendement futur. Alpha = IC × volatilité × z-score "
                        "(règle de Grinold). 0,05–0,10 est l'ordre de grandeur usuel.")

# --------------------------------------------------------------------------- #
# Calculs sur l'univers complet (avant filtres)
# --------------------------------------------------------------------------- #
ALL_NUM = du.NUMERIC + du.DERIVED
base = du.winsorize(df_full, ALL_NUM, p, 1 - p) if wins else df_full.copy()
if vol_method == "multi":
    vol_raw = an.multi_period_vol(df_full, pd.Timestamp(data_date))
else:
    vol_raw = df_full["vol_proxy_1m"]
base["vol"] = an.shrink_to_group(vol_raw, df_full["subindustry"], alpha)
# Critère agrégé : z-scores par facteur (relatifs aux pairs) -> piliers -> score final
ZF = sc.factor_zscores(base, level=z_level, k=k_shrink)
SCORES = sc.composite_score(ZF)
for c_ in SCORES.columns:
    base[c_] = SCORES[c_]
# Rendement attendu ex ante = upside + prime de risque + alpha issu des z-scores
MU_PARTS = sc.expected_return(base, SCORES["Score alpha"], base["vol"], rf, lam, ic)
MU_COMP = ["Ancrage prime de risque", "Ancrage upside (niveau)", "Alpha z-scores"]
base["mu"] = MU_PARTS["mu"]
for c_ in MU_COMP:
    base[c_] = MU_PARTS[c_]
# repère : la moyenne 50/50 prime / upside brut (première version)
MU_5050 = sc.expected_return(base, SCORES["Score alpha"], base["vol"], rf, 0.5, ic)["mu"]
base["sharpe_ante"] = base["mu"] / base["vol"]
base["sharpe_post"] = (base["tr_prev_year"] - rf) / base["vol"]
RISK = ["vol", "mu", "sharpe_ante", "sharpe_post", "Score final"]
du.LABELS.setdefault("Score final", "Score final (critère agrégé)")
ESG_COLS = ["esg"] if HAS_ESG else []
if HAS_ESG:
    # médiane ESG de la sous-industrie, calculée sur les titres notés uniquement
    base["esg_med_sub"] = base.groupby("subindustry")["esg"].transform("median")
    base["esg_above"] = np.where(base["esg"].isna(), np.nan, (base["esg"] > base["esg_med_sub"]).astype(float))
ALL_NUM = ALL_NUM + RISK + ESG_COLS

mask = df_full.sector.isin(sectors) & df_full.subindustry.isin(subs) & df_full.country.isin(countries)
df = base.loc[mask].copy()
df_raw_f = df_full.loc[mask].copy()
st.sidebar.caption(f"{len(df)} titres sélectionnés sur {len(df_full)}.")
if len(df) < 3:
    st.warning("Sélection trop restreinte : élargissez les filtres.")
    st.stop()

st.markdown(f"""<div class="amundi-band"><img src="{LOGO_URL}" alt="Amundi" style="color:#002855;font-weight:700;font-size:1.3rem">
<div><h1>Univers d'actions : analyse et construction de portefeuille</h1>
<p>99 valeurs Technologie et Communication · critère agrégé vs Rule of 40 · portefeuille Sharpe max ≤ 5 % par titre · filtre ESG</p></div></div>""",
            unsafe_allow_html=True)
st.caption("Source : fichier « universe of stock 261113 » (99 titres, 17 colonnes)"
           + (" + scores ESG sourcés (voir onglet ESG)." if HAS_ESG else ". Scores ESG : en attente."))

tabs = st.tabs([
    "1 · L'univers", "2 · Synthèse", "3 · Rendement & risque", "4 · Valorisation & attentes",
    "5 · Croissance & Rule of 40", "6 · ESG", "7 · Relations entre facteurs", "8 · Critère agrégé vs Rule of 40",
    "9 · Rendement attendu & Sharpe", "10 · Frontière efficiente", "11 · Fiche titre", "12 · Portefeuille final", "Annexe · Données & méthodes",
])

# =========================================================================== #
# 1. L'univers : composition
# =========================================================================== #
with tabs[0]:
    intro("<b>Étape 1 – Poser la situation.</b> Avant toute analyse, de quoi est fait l'univers ? "
          "Combien de titres, dans quels secteurs, sous-secteurs et pays ?")
    k = st.columns(4)
    k[0].metric("Titres", len(df))
    k[1].metric("Secteurs", df.sector.nunique())
    k[2].metric("Sous-industries", df.subindustry.nunique())
    k[3].metric("Pays", df.country.nunique())

    c1, c2 = st.columns(2)
    with c1:
        sec = df.sector.value_counts().rename_axis("sector").reset_index(name="n")
        fig = px.pie(sec, names="sector", values="n", color="sector", color_discrete_map=SECTOR_COLORS,
                     hole=0.45, title="Répartition par secteur (nombre de titres)")
        fig.update_traces(textinfo="label+percent+value", marker_line_color="white", marker_line_width=2)
        fig.update_layout(showlegend=False)
        show(fig, c1)
    with c2:
        fig = px.sunburst(df.assign(n=1), path=["sector", "subindustry"], values="n", color="sector",
                          color_discrete_map=SECTOR_COLORS,
                          title="Secteur → sous-industrie (cliquer pour zoomer)")
        fig.update_traces(insidetextorientation="radial", marker_line_color="white")
        show(fig, c2)

    subc = df.groupby(["sector", "subindustry"]).size().reset_index(name="n").sort_values("n")
    fig = px.bar(subc, x="n", y="subindustry", color="sector", orientation="h",
                 color_discrete_map=SECTOR_COLORS, template=TEMPLATE, text="n",
                 labels={"n": "Nombre de titres", "subindustry": ""},
                 title="Nombre de titres par sous-industrie")
    fig.update_layout(height=560)
    show(fig)

    c1, c2 = st.columns(2)
    with c1:
        cnt = df.country.value_counts()
        top = cnt[cnt >= 3]
        pays = pd.concat([top, pd.Series({"Autres pays": cnt[cnt < 3].sum()})]).rename_axis("Pays").reset_index(name="n")
        fig = px.pie(pays, names="Pays", values="n", hole=0.45, title="Répartition par pays",
                     color_discrete_sequence=["#0062A3", "#5FB3E4", "#00A19A", "#A9CBEA", "#8A99AD",
                                              "#2E5E8C", "#002855", "#C5CED9"])
        fig.update_traces(textinfo="label+percent", marker_line_color="white", marker_line_width=2)
        show(fig, c1)
    with c2:
        zone = np.where(df.country == "UNITED STATES", "États-Unis", "Hors États-Unis")
        xt = pd.crosstab(df.sector, zone)
        fig = px.bar(xt.reset_index().melt(id_vars="sector", var_name="Zone", value_name="n"),
                     x="sector", y="n", color="Zone", barmode="group", text="n", template=TEMPLATE,
                     color_discrete_sequence=["#0062A3", "#5FB3E4"],
                     labels={"n": "Nombre de titres"}, title="Secteur × zone géographique")
        show(fig, c2)

    _top_sub = df.subindustry.value_counts()
    recap([f"{len(df)} titres : {(df.sector == 'Information Technology').mean():.0%} en technologie, "
           f"{(df.country == 'UNITED STATES').mean():.0%} américains.",
           f"Les 3 plus grosses sous-industries ({', '.join(_top_sub.index[:3])}) pèsent "
           f"{_top_sub.iloc[:3].sum() / len(df):.0%} de l'univers ; {(_top_sub <= 2).sum()} sous-industries ont 2 titres ou moins.",
           "Conséquence : un portefeuille équipondéré de l'univers est en réalité un pari « tech US ». La diversification "
           "devra se raisonner par sous-industrie, pas seulement par secteur."],
          ["Classification GICS (secteur → groupe → industrie → sous-industrie).",
           "Concentration : part des k plus gros groupes ; diversification sectorielle et géographique."])
    suite("Étape 2 : une synthèse chiffrée simple (moyennes, médianes) avant d'entrer dans le détail.")

# =========================================================================== #
# 2. Synthèse globale
# =========================================================================== #
with tabs[1]:
    intro("<b>Étape 2 – Vue globale.</b> Les grandes moyennes de l'univers : rendement, volatilité, "
          "Sharpe, valorisation, rentabilité" + (", ESG" if HAS_ESG else "") + ". La médiane est moins "
          "sensible aux valeurs extrêmes que la moyenne : un écart important entre les deux signale une "
          "distribution asymétrique.")
    synth_cols = ["ret_1m", "ret_ytd", "tr_prev_year", "vol", "sharpe_ante", "sharpe_post", "upside",
                  "pe_next", "peg", "fcf_yield", "ltg_eps", "sales_growth", "profit_margin", "rule_of_40",
                  "risk_prem"] + ESG_COLS
    s = df[synth_cols].agg(["mean", "median", "std", "min", "max"]).T
    s.columns = ["Moyenne", "Médiane", "Écart-type", "Min", "Max"]
    s.index = [lab(c) for c in s.index]
    st.dataframe(s.style.format("{:.2f}"), width="stretch", height=560)

    st.markdown("**Même synthèse, secteur par secteur** (moyenne / médiane)")
    g = df.groupby("sector")[synth_cols].agg(["mean", "median"]).T
    g.index = [f"{lab(v)} – {'moyenne' if s_ == 'mean' else 'médiane'}" for v, s_ in g.index]
    st.dataframe(g.style.format("{:.2f}"), width="stretch", height=560)

    st.markdown("**Le profil des secteurs en un coup d'œil**")
    med = df.groupby("sector")[synth_cols].median()
    zs = (med - df[synth_cols].median()) / df[synth_cols].std()
    fig = px.imshow(zs.T.rename(index=du.LABELS), aspect="auto", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, text_auto=".2f",
                    title="Écart de la médiane du secteur à celle de l'univers (en écarts-types)")
    fig.update_layout(height=600)
    show(fig)
    _gap = (df["ret_ytd"].mean() - df["ret_ytd"].median())
    recap([f"Perf. YTD : moyenne {df.ret_ytd.mean():.1f} % contre médiane {df.ret_ytd.median():.1f} % "
           f"({'quelques très fortes hausses tirent la moyenne' if _gap > 0 else 'quelques fortes baisses tirent la moyenne'}).",
           f"Volatilité estimée médiane : {df.vol.median():.0f} % par an ; prime de risque médiane : {df.risk_prem.median():.1f} %.",
           f"Upside médian des analystes : {df.upside.median():.1f} % ; marge nette médiane : {df.profit_margin.median():.1f} %."],
          ["Moyenne vs médiane : la médiane résiste aux valeurs extrêmes, l'écart entre les deux mesure l'asymétrie.",
           "Écart-type : dispersion autour de la moyenne. Winsorisation : écrêtage aux quantiles p et 1 − p.",
           "Carte de chaleur : (médiane du secteur − médiane univers) ÷ écart-type, soit un écart en « nombre d'écarts-types »."])
    suite("Étape 3 : on affine rendement et volatilité par secteur et sous-industrie.")

# =========================================================================== #
# 3. Rendement & risque par secteur
# =========================================================================== #
with tabs[2]:
    intro("<b>Étape 3 – Rendement et volatilité, du secteur au titre.</b> La volatilité est <i>estimée</i> "
          "(le fichier n'a pas d'historique de prix) : voir l'Annexe pour la méthode.")
    grp = st.radio("Niveau d'analyse", ["sector", "subindustry", "country"], horizontal=True,
                   format_func=GROUP_FR.get, key="rr_grp")
    agg = df.groupby(grp).agg(n=("ticker", "size"), ret_1m=("ret_1m", "mean"), ret_ytd=("ret_ytd", "mean"),
                              tr_prev_year=("tr_prev_year", "mean"), vol=("vol", "mean"),
                              vol_med=("vol", "median"), sharpe=("sharpe_ante", "median")).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        long = agg.melt(id_vars=grp, value_vars=["ret_1m", "ret_ytd", "tr_prev_year"],
                        var_name="Horizon", value_name="Rendement moyen (%)")
        long["Horizon"] = long["Horizon"].map(lab)
        fig = px.bar(long, x=grp, y="Rendement moyen (%)", color="Horizon", barmode="group",
                     template=TEMPLATE, color_discrete_sequence=["#0062A3", "#5FB3E4", "#00A19A"],
                     title="Rendement moyen par horizon")
        fig.update_layout(height=450)
        show(fig, c1)
    with c2:
        fig = px.box(df, x=grp, y="vol", color="sector" if grp != "sector" else grp, points="all",
                     hover_name="ticker", color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                     labels={"vol": lab("vol")}, title="Distribution de la volatilité estimée")
        fig.update_layout(height=450, showlegend=grp != "sector")
        show(fig, c2)

    fig = px.scatter(agg[agg.n >= 1], x="vol", y="tr_prev_year", size="n", text=grp, template=TEMPLATE,
                     size_max=45, color_discrete_sequence=["#0062A3"],
                     labels={"vol": "Volatilité moyenne (%)", "tr_prev_year": "Total return N-1 moyen (%)",
                             "n": "Titres"},
                     title="Couple rendement / risque par groupe (taille = nombre de titres)")
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(height=520)
    show(fig)

    st.markdown("**Au niveau des titres**")
    c1, c2 = st.columns(2)
    with c1:
        ret_var = st.selectbox("Rendement (axe Y)", ["tr_prev_year", "ret_ytd", "mu", "upside"], format_func=lab)
        fig = px.scatter(df, x="vol", y=ret_var, color="sector", hover_name="ticker",
                         hover_data={"name": True, "subindustry": True},
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                         labels={"vol": lab("vol"), ret_var: lab(ret_var)}, title="Rendement vs volatilité")
        show(fig, c1)
    with c2:
        fig = px.scatter(df, x="tr_prev_year", y="ret_ytd", color="sector", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                         labels={"tr_prev_year": lab("tr_prev_year"), "ret_ytd": lab("ret_ytd")},
                         title="Persistance : année N-1 vs YTD")
        fig.add_hline(y=0, line_color="#999", line_width=1)
        fig.add_vline(x=0, line_color="#999", line_width=1)
        show(fig, c2)

    st.markdown("**Dispersion entre titres d'un même groupe** (écart-type des rendements, en points de %)")
    disp = df.groupby(grp)[["ret_1m", "ret_ytd", "tr_prev_year"]].std()
    disp = disp[df.groupby(grp).size() >= 2].rename(columns=du.LABELS)
    fig = px.imshow(disp, text_auto=".1f", aspect="auto", color_continuous_scale="Blues")
    fig.update_layout(height=max(280, 28 * len(disp) + 120))
    show(fig)
    st.caption("Forte dispersion = groupe hétérogène, où le choix des titres compte plus que le choix du secteur.")
    _sv = df.groupby("sector").agg(tr=("tr_prev_year", "median"), v=("vol", "median"))
    _corr_rv = df["vol"].corr(df["tr_prev_year"], method="spearman")
    recap(["Total return N-1 médian : " + " ; ".join(f"{i} {r.tr:.1f} %" for i, r in _sv.iterrows()) + ".",
           "Volatilité médiane : " + " ; ".join(f"{i} {r.v:.0f} %" for i, r in _sv.iterrows()) + ".",
           f"Corrélation de rang volatilité / rendement N-1 : {_corr_rv:.2f} "
           f"({'plus de risque a été rémunéré' if _corr_rv > 0.2 else 'le risque supplémentaire n’a pas été clairement rémunéré'})."],
          ["Volatilité = écart-type annualisé des rendements, estimée ici sur 3 sous-périodes disjointes (log-rendements, σ√T).",
           "Rétrécissement (shrinkage) vers la médiane de la sous-industrie pour réduire le bruit d'estimation.",
           "Dispersion transversale : écart-type des rendements entre titres d'un même groupe."])
    suite("Étape 4 : ce que le marché paie (valorisation) et ce qu'attendent les analystes.")

# =========================================================================== #
# 8. Critère agrégé vs Rule of 40
# =========================================================================== #
with tabs[7]:
    intro("<b>Étape 8 – Un critère de comparaison agrégé.</b> Chaque facteur est transformé en <b>z-score</b> "
          "par rapport aux pairs (sous-industrie par défaut), les z-scores sont moyennés en 4 piliers, puis le "
          "<b>score final</b> est la moyenne des 4 piliers. On le compare ensuite à la Rule of 40.")
    st.latex(r"z_{i,f} = s_f\,\frac{x_{i,f} - \bar x_{g(i),f}}{\sigma_{g(i),f}}, \qquad "
             r"\text{Pilier}_{i,p} = \frac{1}{|p|}\sum_{f\in p} z_{i,f}, \qquad "
             r"\text{Score}_i = \frac{1}{4}\sum_{p=1}^{4}\text{Pilier}_{i,p}")
    st.caption("s_f = +1 si « plus haut = mieux », −1 sinon (PEG). Moyenne et écart-type du groupe g(i) rétrécis vers "
               "le secteur : (n·groupe + k·secteur) ÷ (n + k). Données écrêtées, z-scores bornés à ±3.")
    pil = pd.DataFrame([{"Pilier": p, "Facteur": lab(f), "Sens": "↑ mieux" if s_ > 0 else "↓ mieux"}
                        for p, fs in sc.PILLARS.items() for f, s_ in fs.items()])
    c1, c2 = st.columns([2, 3])
    c1.dataframe(pil, hide_index=True, width="stretch")
    with c2:
        zc = ZF.loc[df.index].rename(columns=du.LABELS)
        fig = px.box(zc.melt(var_name="Facteur", value_name="z"), x="Facteur", y="z", template=TEMPLATE,
                     color_discrete_sequence=["#0062A3"], title="Distribution des z-scores par facteur")
        fig.update_layout(height=380, xaxis_tickangle=-25)
        show(fig, c2)

    pillars = list(sc.PILLARS)
    rank = df[["ticker", "name", "sector", "subindustry"] + pillars + ["Score final", "rule_of_40"]] \
        .sort_values("Score final", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        top = rank.head(20).iloc[::-1]
        fig = go.Figure()
        cols_p = ["#0062A3", "#00A19A", "#A9CBEA", "#002855"]
        for p_, col_ in zip(pillars, cols_p):
            fig.add_trace(go.Bar(y=top.ticker, x=top[p_] / 4, name=p_, orientation="h", marker_color=col_))
        fig.add_trace(go.Scatter(y=top.ticker, x=top["Score final"], mode="markers", name="Score final",
                                 marker=dict(color="#0b0b0b", size=9, symbol="diamond")))
        fig.update_layout(barmode="relative", template=TEMPLATE, height=620,
                          title="Top 20 du score final (contribution de chaque pilier = pilier ÷ 4)")
        show(fig, c1)
    with c2:
        sm_ = df.groupby(["sector", "subindustry"])["Score final"].agg(["median", "count"]).reset_index() \
            .sort_values("median")
        fig = px.bar(sm_, x="median", y="subindustry", color="sector", orientation="h", text="count",
                     color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                     labels={"median": "Score final médian", "subindustry": ""},
                     title="Score médian par sous-industrie (≈ 0 par construction)")
        fig.update_layout(height=620)
        show(fig, c2)
    st.caption("Le score étant relatif aux pairs, la médiane de chaque sous-industrie est proche de 0 : le score "
               "classe les titres *au sein* de leur groupe, ce qui évite qu'un secteur entier domine le classement.")

    st.markdown("---")
    st.header("Comparaison avec la Rule of 40")
    r40 = df_raw_f.loc[df.index, "rule_of_40"]
    passe_rate = float((r40 >= 40).mean())
    st.markdown(f"""
**Les deux critères ne mesurent pas la même chose.**

| | Rule of 40 | Critère agrégé (score final) |
| --- | --- | --- |
| Définition | croissance du CA + marge nette ≥ 40 % | moyenne de 4 piliers de z-scores (8 facteurs) |
| Information utilisée | 2 colonnes (passé récent) | croissance, rentabilité, **valorisation** et **attentes du marché** |
| Référence | seuil absolu, identique pour tous | relative aux pairs de la sous-industrie |
| Résultat | binaire : passe / ne passe pas | continu : un classement complet |
| Prix payé | ignoré | intégré (PEG, earnings yield) |

**Méthode de comparaison, en quatre temps :** (1) les deux critères classent-ils les titres dans le même ordre ?
(2) sélectionnent-ils les mêmes titres ? (3) d'où viennent les écarts ? (4) quel profil de titres chacun retient-il ?
Ici **{int((r40 >= 40).sum())} titres sur {len(df)}** ({passe_rate:.0%}) passent la Rule of 40.
""")

    # ---- (1) Concordance des classements
    st.subheader("1. Les deux critères classent-ils les titres dans le même ordre ?")
    cmp_all = sc.rule40_comparison(df["Score final"], r40, passe_rate)
    k = st.columns(3)
    k[0].metric("Corrélation de Spearman", f"{cmp_all['spearman']:.2f}",
                help="Corrélation entre les rangs : 1 = même classement, 0 = aucun lien, −1 = classements inversés.")
    k[1].metric("Tau de Kendall", f"{cmp_all['kendall']:.2f}",
                help="Probabilité qu'une paire de titres soit rangée dans le même ordre, moins la probabilité inverse.")
    k[2].metric("Part de variance commune des rangs (ρ²)", f"{cmp_all['spearman'] ** 2:.0%}")
    d_ = df.assign(r40=r40.clip(-60, 120), statut=np.where(r40 >= 40, "Passe la Rule of 40", "Ne passe pas"))
    fig = px.scatter(d_, x="r40", y="Score final", color="sector", symbol="statut", hover_name="ticker",
                     hover_data={"name": True, "subindustry": True}, color_discrete_map=SECTOR_COLORS,
                     template=TEMPLATE, labels={"r40": "Rule of 40 (écrêtée à [−60 ; 120])"},
                     title="Score final vs Rule of 40 (ligne verticale = seuil de 40)")
    fig.add_vline(x=40, line_dash="dash", line_color="#666")
    fig.update_layout(height=520)
    show(fig)
    st.markdown(f"Avec ρ = {cmp_all['spearman']:.2f}, les deux critères vont **dans le même sens** mais sont "
                f"**loin d'être équivalents** : le classement par la Rule of 40 n'explique qu'environ "
                f"{cmp_all['spearman'] ** 2:.0%} de la variance du classement par le score.")

    # ---- (2) Concordance des sélections
    st.subheader("2. Sélectionnent-ils les mêmes titres ?")
    top_share = st.slider("Taille du « haut du classement » par le score (par défaut : autant de titres que ceux qui "
                          "passent la Rule of 40)", 0.1, 0.6, round(passe_rate, 2), 0.01, format="%.2f")
    cmp_ = sc.rule40_comparison(df["Score final"], r40, top_share)
    k = st.columns(4)
    k[0].metric("Taux d'accord", f"{cmp_['accord']:.0%}",
                help="Part des titres classés pareil (dans le haut du score ET passe la règle, ou ni l'un ni l'autre).")
    k[1].metric("Accord attendu par hasard", f"{cmp_['accord_hasard']:.0%}",
                help="Accord qu'on obtiendrait en tirant les deux sélections au hasard avec les mêmes proportions.")
    k[2].metric("κ de Cohen", f"{cmp_['kappa']:.2f}",
                help="Accord corrigé du hasard : 0 = pas mieux que le hasard, 1 = accord parfait. 0,2–0,4 = faible, "
                     "0,4–0,6 = modéré.")
    both = int((cmp_["top"] & cmp_["passe"]).sum())
    k[3].metric("Titres retenus par les deux", f"{both} / {int(cmp_['passe'].sum())}")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Tableau croisé**")
        st.dataframe(cmp_["crosstab"], width="stretch")
        st.caption("Le taux d'accord brut est flatteur, car la plupart des titres sont « ni l'un ni l'autre » ; "
                   "le κ de Cohen retire cette part d'accord mécanique.")
    with c2:
        sub_cmp = pd.DataFrame({"Passe la Rule of 40": cmp_["passe"].groupby(df.subindustry).mean() * 100,
                                "Haut du classement du score": cmp_["top"].groupby(df.subindustry).mean() * 100})
        sub_cmp = sub_cmp.loc[df.subindustry.value_counts().index]
        fig = px.bar(sub_cmp.reset_index().melt(id_vars="subindustry", var_name="Critère", value_name="pct"),
                     x="pct", y="subindustry", color="Critère", barmode="group", orientation="h", template=TEMPLATE,
                     color_discrete_sequence=[C_LIGHT, C_NAVY],
                     labels={"pct": "% des titres de la sous-industrie retenus", "subindustry": ""},
                     title="Qui est retenu, sous-industrie par sous-industrie ?")
        fig.update_layout(height=560, yaxis=dict(autorange="reversed"))
        show(fig, c2)

    # ---- (3) Origine des écarts
    st.subheader("3. D'où viennent les écarts ?")
    corr_p = pd.Series({p_: df[p_].corr(r40, method="spearman") for p_ in pillars + ["Score final"]})
    r40_rel = sc._shrunk_group_stats(r40, df["subindustry"], df["sector"], k_shrink)
    r40_z = (r40 - r40_rel[0]) / r40_rel[1]
    rho_rel = df["Score final"].corr(r40_z, method="spearman")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(corr_p.reset_index(name="rho").rename(columns={"index": "Pilier"}), x="rho", y="Pilier",
                     orientation="h", template=TEMPLATE, color_discrete_sequence=[C_BLUE], text_auto=".2f",
                     labels={"rho": "Spearman avec la Rule of 40"}, title="Corrélation de chaque pilier avec la Rule of 40")
        fig.update_layout(height=340)
        show(fig, c1)
    with c2:
        # robustesse : la conclusion dépend-elle du niveau de comparaison des z-scores ?
        rob = []
        for lvl, kk in [("univers", 0), ("sector", 0), ("subindustry", 0), ("subindustry", 5), ("subindustry", 10)]:
            z_ = sc.factor_zscores(base, level=lvl, k=kk)
            s_ = sc.composite_score(z_).loc[df.index, "Score final"]
            rob.append({"z-scores calculés par rapport à": {"univers": "l'univers", "sector": "le secteur",
                                                           "subindustry": f"la sous-industrie (k = {kk})"}[lvl],
                        "Spearman avec la Rule of 40": s_.corr(r40, method="spearman")})
        st.markdown("**Robustesse : le résultat dépend-il de la façon de calculer les z-scores ?**")
        st.dataframe(pd.DataFrame(rob).style.format({"Spearman avec la Rule of 40": "{:.2f}"}), hide_index=True,
                     width="stretch")
        st.markdown(f"Si l'on rend aussi la Rule of 40 **relative aux pairs** (z-score dans la sous-industrie), sa "
                    f"corrélation avec le score passe de {cmp_all['spearman']:.2f} à **{rho_rel:.2f}**.")
    best_p = corr_p.drop("Score final").idxmax()
    worst_p = corr_p.drop("Score final").idxmin()
    st.markdown(f"""
- Le pilier le plus proche de la Rule of 40 est **{best_p}** (ρ = {corr_p[best_p]:.2f}) : normal, la règle additionne
  croissance et marge. Le plus éloigné est **{worst_p}** (ρ = {corr_p[worst_p]:.2f}).
- Les écarts viennent donc surtout de ce que la Rule of 40 **ne regarde pas** : le **prix payé** pour la croissance
  (PEG, earnings yield) et ce qu'**attend le marché** (upside, prime de risque). Une société en forte croissance mais
  très chère peut passer la règle tout en ayant un score moyen, et inversement.
- Le second écart vient de la **référence** : la règle est un seuil absolu, plus facile à franchir pour un éditeur de
  logiciels que pour un opérateur télécom ou un fabricant de matériel. Le score, relatif aux pairs, laisse sa chance à
  chaque sous-industrie (voir le graphique « Qui est retenu »).
""")

    # ---- (4) Profil des titres retenus
    st.subheader("4. Quel profil de titres chaque critère retient-il ?")
    prof_vars = ["upside", "peg", "pe_next", "risk_prem", "mu", "vol", "sharpe_ante", "sales_growth", "profit_margin",
                 "tr_prev_year", "ret_ytd"] + ESG_COLS
    groups_ = {"Passent la Rule of 40": cmp_["passe"], "Haut du classement du score": cmp_["top"],
               "Retenus par les deux": cmp_["top"] & cmp_["passe"], "Univers": pd.Series(True, index=df.index)}
    prof = pd.DataFrame({g: df.loc[m_, prof_vars].median() for g, m_ in groups_.items()})
    prof.loc["Nombre de titres"] = [int(m_.sum()) for m_ in groups_.values()]
    prof.loc["Sous-industries représentées"] = [df.loc[m_, "subindustry"].nunique() for m_ in groups_.values()]
    prof.index = [lab(i) + (" (médiane)" if i in prof_vars else "") for i in prof.index]
    st.dataframe(prof.style.format("{:.2f}"), width="stretch")
    pr_ = df.loc[cmp_["passe"]]
    tp_ = df.loc[cmp_["top"]]
    st.markdown(f"""
- **Valorisation :** PEG médian {pr_.peg.median():.2f} pour les titres qui passent la Rule of 40, contre
  {tp_.peg.median():.2f} pour le haut du score. {"La Rule of 40 retient des titres plus chers par rapport à leur croissance." if pr_.peg.median() > tp_.peg.median() else "Les deux sélections paient la croissance à un prix comparable."}
- **Attentes :** upside médian {pr_.upside.median():.1f} % (Rule of 40) contre {tp_.upside.median():.1f} % (score) ;
  rendement attendu médian {pr_.mu.median():.1f} % contre {tp_.mu.median():.1f} %.
- **Risque :** volatilité médiane {pr_.vol.median():.0f} % contre {tp_.vol.median():.0f} % ; Sharpe ex ante médian
  {pr_.sharpe_ante.median():.2f} contre {tp_.sharpe_ante.median():.2f}.
- **Diversification :** {pr_.subindustry.nunique()} sous-industries représentées avec la Rule of 40, contre
  {tp_.subindustry.nunique()} avec le score.
""")
    dis = rank.assign(passe=cmp_["passe"], top=cmp_["top"])
    show_cols = ["ticker", "name", "subindustry", "Score final", "rule_of_40"] + pillars
    fmt = {c_: "{:.2f}" for c_ in ["Score final"] + pillars} | {"rule_of_40": "{:.1f}"}
    c1, c2 = st.columns(2)
    c1.markdown("**Bien classés par le score mais refusés par la Rule of 40**")
    c1.dataframe(dis[dis.top & ~dis.passe][show_cols].head(12).style.format(fmt), hide_index=True, width="stretch")
    c1.caption("Typiquement : croissance ou marge modestes, mais valorisation raisonnable et fortes attentes du marché.")
    c2.markdown("**Passent la Rule of 40 mais mal classés par le score**")
    c2.dataframe(dis[~dis.top & dis.passe][show_cols].head(12).style.format(fmt), hide_index=True, width="stretch")
    c2.caption("Typiquement : forte croissance ou forte marge, mais valorisation élevée (PEG haut) ou peu d'upside.")

    st.markdown("#### Conclusion de la comparaison")
    st.markdown(f"""
Le critère agrégé et la Rule of 40 sont **positivement liés** (Spearman {cmp_all['spearman']:.2f}, κ de Cohen
{cmp_['kappa']:.2f}) : une entreprise qui croît vite et rentablement tend aussi à bien scorer. Mais la Rule of 40 n'est
qu'**un filtre de qualité opérationnelle**, adapté aux éditeurs de logiciels, alors que le score est **un critère de
sélection d'investissement** : il tient compte du prix payé et des attentes, et compare chaque société à ses vrais
concurrents. C'est pourquoi le portefeuille final s'appuie sur le score ; la Rule of 40 reste un indicateur de contrôle
(part du portefeuille qui la respecte, voir l'onglet Portefeuille).
""")
    recap([f"Corrélation de rang score / Rule of 40 : {cmp_all['spearman']:.2f} ; κ de Cohen {cmp_['kappa']:.2f} "
           f"(accord {cmp_['accord']:.0%}, dont {cmp_['accord_hasard']:.0%} attendus par hasard).",
           f"Pilier le plus proche : « {best_p} » ; les écarts viennent de la valorisation, des attentes et de la "
           "référence relative aux pairs.",
           f"{both} titres sont retenus par les deux critères sur {int(cmp_['passe'].sum())}."],
          ["z-score : (valeur − moyenne des pairs) ÷ écart-type des pairs, unité commune « en écarts-types ».",
           "Estimateur bayésien empirique (rétrécissement vers le secteur) pour les petites sous-industries.",
           "Corrélations de rang de Spearman et de Kendall ; tableau de contingence ; κ de Cohen (accord corrigé du hasard)."])
    suite("Étape 9 : transformer ce score en rendement attendu, puis en Sharpe ex ante.")

# =========================================================================== #
# 9. Rendement attendu & Sharpe
# =========================================================================== #
with tabs[8]:
    intro("<b>Étape 9 – Rendement attendu et rendement par unité de risque.</b> Le rendement attendu réunit "
          "les trois informations prospectives du fichier : la <b>prime de risque</b> implicite (le niveau), "
          "l'<b>upside</b> des analystes et les <b>z-scores</b> du critère agrégé (l'écart de chaque titre à ce niveau). "
          "Le Sharpe ex ante le rapporte à la volatilité estimée.")
    st.latex(r"\mu_i = \underbrace{(1-\lambda)\,\text{Prime}_i + \lambda\,(\text{Upside}_i - r_f)}_{\text{niveau}} + "
             r"\underbrace{IC\cdot\sigma_i\cdot Z^{\alpha}_i}_{\text{alpha z-scores}}, \qquad "
             r"Z^{\alpha}_i = \tfrac14\big(Z^{croiss.}_i + Z^{rentab.}_i + Z^{valo.}_i + z^{upside}_i\big), \qquad "
             r"\text{Sharpe}^{ex\,ante}_i = \frac{\mu_i}{\sigma_i}")
    st.caption(f"Réglages actuels : λ = {lam:.2f}, IC = {ic:.2f}, r_f = {rf:.2f} %. Alpha = IC × σ × z (règle de "
               "Grinold) : un z-score de +1 ajoute IC × σ au rendement attendu. La prime de risque n'est pas dans "
               "Z^α car elle donne déjà le niveau : on ne la compte pas deux fois.")
    with st.expander("Pourquoi la prime de risque donne le niveau, et l'upside seulement l'écart relatif ?", expanded=True):
        st.markdown(f"""
- **L'upside brut est un objectif de cours à 12 mois, pas une espérance de rendement.** Les objectifs des analystes sont
  structurellement optimistes : ici **{(df_full.upside > 0).mean():.0%} des titres** ont un objectif au-dessus du cours et
  l'upside médian atteint **{df_full.upside.median():.1f} %**. Prendre 50 % d'upside brut et 50 % de prime de risque
  (première version) donnait un rendement attendu médian de **{MU_5050.median():.1f} %**, trop élevé pour être crédible.
- **La prime de risque implicite est déjà un rendement en excès du sans-risque**, déduit des cours et des estimations
  (médiane **{df_full.risk_prem.median():.1f} %**). C'est la « supposition » de marché la plus cohérente pour fixer le niveau.
- **L'information de l'upside est surtout relative** : un titre avec plus d'upside que ses pairs est plus apprécié des analystes.
  On la garde donc via son **z-score** (dans Z^α), ce qui retire le biais d'optimisme commun à tous les titres.
- Résultat avec les réglages actuels : rendement attendu médian **{base.mu.median():.1f} %** (contre {MU_5050.median():.1f} %
  avec 50/50). Le curseur λ de la barre latérale permet de revenir à l'ancienne pondération pour comparer.
""")
    dec = df.groupby("subindustry")[MU_COMP + ["mu"]].mean().sort_values("mu")
    fig = go.Figure()
    for p_, col_ in zip(MU_COMP, [C_NAVY, C_LIGHT, C_ACCENT]):
        if dec[p_].abs().sum() > 0:
            fig.add_trace(go.Bar(y=dec.index, x=dec[p_], name=p_, orientation="h", marker_color=col_))
    fig.add_trace(go.Scatter(y=dec.index, x=dec["mu"], mode="markers", name="Rendement attendu total",
                             marker=dict(color="#0b0b0b", size=9, symbol="diamond")))
    fig.update_layout(barmode="relative", template=TEMPLATE, height=560, xaxis_title="Contribution moyenne (%)",
                      title="Décomposition du rendement attendu, moyenne par sous-industrie")
    show(fig)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df, x="Score final", y="mu", color="sector", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE, labels={"mu": lab("mu")},
                         title="Score final vs rendement attendu")
        show(fig, c1)
    with c2:
        fig = px.histogram(df, x="mu", color="sector", nbins=30, barmode="overlay", opacity=0.7,
                           color_discrete_map=SECTOR_COLORS, template=TEMPLATE, labels={"mu": lab("mu")},
                           title="Distribution du rendement attendu")
        show(fig, c2)
    st.markdown("**Sharpe ex ante et ex post**")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="sector", y="sharpe_ante", color="sector", points="all", hover_name="ticker",
                     color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                     labels={"sharpe_ante": lab("sharpe_ante")}, title="Sharpe ex ante par secteur")
        fig.update_layout(showlegend=False)
        show(fig, c1)
    with c2:
        sm = df.groupby(["sector", "subindustry"]).sharpe_ante.median().reset_index().sort_values("sharpe_ante")
        fig = px.bar(sm, x="sharpe_ante", y="subindustry", color="sector", orientation="h",
                     color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                     labels={"sharpe_ante": "Sharpe ex ante médian", "subindustry": ""},
                     title="Sharpe ex ante médian par sous-industrie")
        fig.update_layout(height=520)
        show(fig, c2)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df, x="sharpe_post", y="sharpe_ante", color="sector", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                         labels={"sharpe_post": lab("sharpe_post"), "sharpe_ante": lab("sharpe_ante")},
                         title="Sharpe constaté (N-1) vs Sharpe attendu")
        show(fig, c1)
    with c2:
        n_top = 15
        top = pd.concat([df.nlargest(n_top, "sharpe_ante").assign(rang="Top 15"),
                         df.nsmallest(n_top, "sharpe_ante").assign(rang="Bas 15")])
        fig = px.bar(top.sort_values("sharpe_ante"), x="sharpe_ante", y="ticker", color="sector",
                     orientation="h", color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                     labels={"sharpe_ante": lab("sharpe_ante"), "ticker": ""},
                     title="Les 15 meilleurs et 15 moins bons Sharpe ex ante")
        fig.update_layout(height=650)
        show(fig, c2)
    _sh = df.groupby("sector")["sharpe_ante"].median()
    recap([f"Rendement attendu médian : {df.mu.median():.1f} % (niveau de marché médian "
           f"{(df['Ancrage prime de risque'] + df['Ancrage upside (niveau)']).median():.1f} %) ; l'alpha z-scores va de "
           f"{df['Alpha z-scores'].min():+.1f} à {df['Alpha z-scores'].max():+.1f} pt selon les titres.",
           "Sharpe ex ante médian : " + " ; ".join(f"{i} {v:.2f}" for i, v in _sh.items()) + ".",
           f"Corrélation de rang Sharpe ex ante / ex post : {df.sharpe_ante.corr(df.sharpe_post, method='spearman'):.2f} : "
           "ce qui a bien marché l'an dernier n'est pas forcément ce qui est attendu demain."],
          ["Ratio de Sharpe = (rendement − taux sans risque) ÷ volatilité, ex ante (attendu) ou ex post (constaté).",
           "Règle de Grinold : alpha = IC × volatilité × score ; IC = corrélation score / rendement futur.",
           "Combinaison linéaire de signaux (upside, prime, score) pour un rendement espéré « tout compris »."])
    suite("Étape 10 : combiner les titres. Quelle est la meilleure combinaison rendement / risque possible ?")

# =========================================================================== #
# 5. Frontière efficiente
# =========================================================================== #
with tabs[9]:
    intro("<b>Étape 10 – Frontière efficiente (Markowitz).</b> Pour chaque niveau de rendement attendu, le "
          "portefeuille de volatilité minimale, sans vente à découvert et avec 5 % maximum par titre. "
          "C'est une analyse de l'univers : la construction du portefeuille final viendra ensuite.")
    with st.expander("Hypothèses de corrélation (le fichier ne permet pas de les estimer)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        rho_sub = c1.slider("Même sous-industrie", 0.0, 0.95, 0.60, 0.05)
        rho_sec = c2.slider("Même secteur", 0.0, 0.95, 0.40, 0.05)
        rho_all = c3.slider("Secteurs différents", 0.0, 0.95, 0.25, 0.05)
        cap = c4.slider("Plafond par titre", 0.02, 0.20, 0.05, 0.01, format="%.2f")
    st.caption("Volatilités et rendements attendus : ceux de la barre latérale. Calcul sur la sélection filtrée.")

    R = an.block_correlation(df, rho_sub, max(min(rho_sec, rho_sub), rho_all), rho_all)
    cov = an.covariance(df["vol"], R)
    mu = df["mu"].to_numpy(float)
    if len(df) * cap < 1:
        st.warning(f"Avec {len(df)} titres et un plafond de {cap:.0%}, le portefeuille ne peut pas être investi à 100 %.")
    else:
        front, w_mv, w_ms = frontier_cached(tuple(mu), tuple(cov.ravel()), len(df), cap)
        ew = np.full(len(df), 1 / len(df))
        r_ew, v_ew = an.port_stats(ew, mu, cov)
        fig = go.Figure()
        for s_, c_ in SECTOR_COLORS.items():
            d_ = df[df.sector == s_]
            fig.add_trace(go.Scatter(x=d_["vol"], y=d_["mu"], mode="markers", name=s_, text=d_["ticker"],
                                     marker=dict(color=c_, size=7, opacity=0.6),
                                     hovertemplate="%{text}<br>vol %{x:.1f} %<br>rdt %{y:.1f} %<extra></extra>"))
        fig.add_trace(go.Scatter(x=front["vol"], y=front["ret"], mode="lines", name=f"Frontière (≤ {cap:.0%}/titre)",
                                 line=dict(color="#0b0b0b", width=2.5)))
        pts = [("Équipondéré univers", ew, "diamond", "#002855")]
        if w_mv is not None:
            pts.append(("Variance minimale", w_mv, "square", "#00A19A"))
        if w_ms is not None:
            pts.append(("Sharpe maximal", w_ms, "star", "#D6453D"))
        for name, w, sym, col in pts:
            r_, v_ = an.port_stats(w, mu, cov)
            fig.add_trace(go.Scatter(x=[v_], y=[r_], mode="markers", name=name,
                                     marker=dict(symbol=sym, size=15, color=col,
                                                                          line=dict(color="white", width=1.5))))
        for s_ in df.sector.unique():
            m_ = (df.sector == s_).to_numpy()
            w = m_ / m_.sum()
            r_, v_ = an.port_stats(w, mu, cov)
            fig.add_trace(go.Scatter(x=[v_], y=[r_], mode="markers", name=f"Équipondéré {s_}",
                                     marker=dict(symbol="circle-open", size=14, color=SECTOR_COLORS.get(s_),
                                                 line=dict(width=3))))
        fig.update_layout(template=TEMPLATE, height=620, xaxis_title=lab("vol"), yaxis_title=lab("mu"),
                          title="Titres, portefeuilles équipondérés et frontière efficiente",
                          xaxis_range=[0, min(df["vol"].max() * 1.05, 120)])
        show(fig)

        rows = []
        for name, w in [("Équipondéré univers", ew), ("Variance minimale", w_mv), ("Sharpe maximal", w_ms)]:
            if w is None:
                continue
            r_, v_ = an.port_stats(w, mu, cov)
            rows.append({"Portefeuille": name, "Rendement attendu (%)": r_, "Volatilité (%)": v_,
                         "Sharpe": r_ / v_, "Lignes": int((w > 1e-4).sum()),
                         "Nb effectif 1/Σw²": 1 / np.sum(w ** 2),
                         "Part tech (%)": 100 * w[(df.sector == "Information Technology").to_numpy()].sum()})
        st.dataframe(pd.DataFrame(rows).style.format(precision=2), hide_index=True, width="stretch")
        fig = px.line(front, x="vol", y="sharpe", markers=True, template=TEMPLATE,
                      labels={"vol": "Volatilité (%)", "sharpe": "Sharpe"},
                      title="Sharpe le long de la frontière", color_discrete_sequence=["#0062A3"])
        show(fig)
        st.caption("Limites : les corrélations sont des hypothèses ; les volatilités sont estimées sur 3 observations "
                   "par titre ; le rendement attendu dépend de la définition choisie. Faire varier ces paramètres "
                   "montre la sensibilité du résultat.")
        _ms = an.port_stats(w_ms, mu, cov) if w_ms is not None else (np.nan, np.nan)
        recap([f"Équipondéré : rendement attendu {r_ew:.1f} %, volatilité {v_ew:.1f} %, Sharpe {r_ew / v_ew:.2f}.",
               f"Point de Sharpe maximal de la frontière : {_ms[0]:.1f} % / {_ms[1]:.1f} %, Sharpe {_ms[0] / _ms[1]:.2f}, "
               f"avec au plus {cap:.0%} par titre.",
               "Le plafond par titre oblige à détenir au moins 1 ÷ plafond lignes : il impose la diversification."],
              ["Théorie moderne du portefeuille (Markowitz) : variance du portefeuille = wᵀΣw.",
               "Matrice de covariance Σ = D·R·D (volatilités × corrélations par blocs).",
               "Optimisation quadratique sous contraintes (Σw = 1, 0 ≤ w ≤ plafond), algorithme SLSQP."])
    suite("Étape 11 : zoom sur un titre.")

# =========================================================================== #
# 6. Valorisation & attentes
# =========================================================================== #
with tabs[3]:
    intro("<b>Étape 4 – Valorisation et attentes.</b> Le PEG (P/E ÷ croissance attendue) dit si la croissance "
          "est chère ; l'upside dit ce qu'en pensent les analystes. On les compare par secteur, sous-industrie et pays.")
    grp = st.radio("Comparer par", ["sector", "subindustry", "country"], horizontal=True,
                   format_func=GROUP_FR.get, key="val_grp")
    var = st.selectbox("Variable", ["peg", "upside", "pe_next", "fcf_yield", "earnings_yield", "ltg_eps"],
                       format_func=lab, key="val_var")
    gstat = df.groupby(grp)[var].agg(["median", "mean", "count"]).reset_index().sort_values("median")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(gstat, x="median", y=grp, orientation="h", text="count", template=TEMPLATE,
                     color_discrete_sequence=["#0062A3"],
                     labels={"median": f"Médiane – {lab(var)}", grp: ""},
                     title=f"{lab(var)} médian (étiquette = nombre de titres)")
        fig.add_vline(x=df[var].median(), line_dash="dash", line_color="#666",
                      annotation_text="médiane univers")
        fig.update_layout(height=max(380, 24 * len(gstat) + 120))
        show(fig, c1)
    with c2:
        fig = px.box(df, x="sector", y=var, color="sector", points="all", hover_name="ticker",
                     color_discrete_map=SECTOR_COLORS, template=TEMPLATE, labels={var: lab(var)},
                     title=f"Distribution de {lab(var)} par secteur")
        fig.update_layout(showlegend=False, height=max(380, 24 * len(gstat) + 120))
        show(fig, c2)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df, x="ltg_eps", y="pe_next", color="sector", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                         labels={"ltg_eps": lab("ltg_eps"), "pe_next": lab("pe_next")},
                         title="P/E N+1 vs croissance attendue (droites = PEG constant)")
        xs = np.linspace(max(df.ltg_eps.min(), 0.1), df.ltg_eps.max(), 50)
        for g_, dash in [(0.5, "dot"), (1, "dash"), (2, "dot")]:
            fig.add_trace(go.Scatter(x=xs, y=g_ * xs, mode="lines", name=f"PEG = {g_}",
                                     line=dict(color="#666", dash=dash, width=1)))
        fig.update_yaxes(range=[0, df.pe_next.max() * 1.05])
        show(fig, c1)
        st.caption("Sous la droite PEG = 1 : la croissance attendue est « payée » moins de 1x.")
    with c2:
        fig = px.scatter(df, x="peg", y="upside", color="sector", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                         labels={"peg": lab("peg"), "upside": lab("upside")},
                         title="PEG vs upside : les analystes préfèrent-ils la croissance bon marché ?")
        show(fig, c2)

    fig = px.scatter(df, x="earnings_yield", y="fcf_yield", color="sector", hover_name="ticker",
                     color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                     labels={"earnings_yield": lab("earnings_yield"), "fcf_yield": lab("fcf_yield")},
                     title="Earnings yield vs FCF yield (les bénéfices se transforment-ils en cash ?)")
    lim = [min(df.earnings_yield.min(), df.fcf_yield.min()), max(df.earnings_yield.max(), df.fcf_yield.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="FCF yield = EY",
                             line=dict(color="#666", dash="dash", width=1)))
    show(fig)
    _pg = df.groupby("sector")["peg"].median()
    recap(["PEG médian : " + " ; ".join(f"{i} {v:.2f}" for i, v in _pg.items()) + f" (univers {df.peg.median():.2f}).",
           f"Upside médian {df.upside.median():.1f} % ; {(df.upside > 0).mean():.0%} des titres ont un objectif de cours au-dessus du cours.",
           f"Corrélation de rang PEG / upside : {df.peg.corr(df.upside, method='spearman'):.2f} ; croissance LT / upside : "
           f"{df.ltg_eps.corr(df.upside, method='spearman'):.2f}."],
          ["P/E = cours ÷ BPA attendu ; earnings yield = 1/P/E ; PEG = P/E ÷ croissance attendue du BPA.",
           "FCF yield = free cash-flow ÷ capitalisation : rendement « cash » de l'action.",
           "Upside = (objectif de cours moyen − cours) ÷ cours : consensus des analystes."])
    suite("Étape 5 : la qualité de la croissance, avec la Rule of 40.")

# =========================================================================== #
# 7. Croissance & Rule of 40
# =========================================================================== #
with tabs[4]:
    intro("<b>Étape 5 – Croissance et rentabilité.</b> La Rule of 40 : croissance du CA (%) + marge (%) ≥ 40. "
          "C'est le critère auquel notre critère agrégé sera comparé.")
    d40 = df_raw_f.assign(passe=np.where(df_raw_f.rule_of_40 >= 40, "≥ 40 (passe)", "< 40"))
    k = st.columns(3)
    k[0].metric("Titres passant la Rule of 40", f"{(d40.passe != '< 40').sum()} / {len(d40)}")
    k[1].metric("Rule of 40 médiane", f"{d40.rule_of_40.median():.1f}")
    k[2].metric("Marge médiane", f"{d40.profit_margin.median():.1f} %")

    c1, c2 = st.columns(2)
    with c1:
        share = d40.groupby("sector").passe.apply(lambda s: (s != "< 40").mean() * 100).reset_index(name="pct")
        fig = px.bar(share, x="sector", y="pct", color="sector", color_discrete_map=SECTOR_COLORS,
                     template=TEMPLATE, text_auto=".0f", labels={"pct": "% de titres passant la règle"},
                     title="Part des titres passant la Rule of 40, par secteur")
        fig.update_layout(showlegend=False)
        show(fig, c1)
    with c2:
        fig = px.scatter(df, x="sales_growth", y="profit_margin", color="sector", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                         labels={"sales_growth": lab("sales_growth"), "profit_margin": lab("profit_margin")},
                         title="Croissance vs marge : frontière x + y = 40")
        xs = np.linspace(df.sales_growth.min(), df.sales_growth.max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=40 - xs, mode="lines", name="x + y = 40",
                                 line=dict(color="#666", dash="dash", width=1.5)))
        fig.update_yaxes(range=[df.profit_margin.min() - 5, df.profit_margin.max() + 5])
        show(fig, c2)

    c1, c2 = st.columns(2)
    with c1:
        top = df.sort_values("rule_of_40", ascending=False).head(25)
        fig = px.bar(top.sort_values("rule_of_40"), x="rule_of_40", y="ticker", color="sector",
                     orientation="h", color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                     labels={"rule_of_40": lab("rule_of_40"), "ticker": ""},
                     title="Top 25 Rule of 40 (valeurs écrêtées si l'option est active)")
        fig.add_vline(x=40, line_dash="dash", line_color="#666")
        fig.update_layout(height=620)
        show(fig, c1)
    with c2:
        fig = px.box(d40, x="passe", y="upside", color="passe", points="all", hover_name="ticker",
                     template=TEMPLATE, color_discrete_sequence=["#0062A3", "#5FB3E4"],
                     labels={"passe": "Rule of 40", "upside": lab("upside")},
                     title="Upside analystes selon la Rule of 40")
        fig.update_layout(showlegend=False, height=620)
        show(fig, c2)
    _ps = d40.groupby("sector").passe.apply(lambda s: (s != "< 40").mean())
    recap([f"{(d40.passe != '< 40').sum()} titres sur {len(d40)} passent la Rule of 40 (" +
           " ; ".join(f"{i} {v:.0%}" for i, v in _ps.items()) + ").",
           f"Rule of 40 médiane : {d40.rule_of_40.median():.1f} ; la règle pèse autant 1 point de croissance qu'1 point de marge.",
           "Limite : elle ignore la valorisation et les attentes, d'où l'intérêt d'un critère plus large (onglet 8)."],
          ["Rule of 40 = croissance du CA (%) + marge (%) ≥ 40 : arbitrage croissance / rentabilité des sociétés tech.",
           "Droite x + y = 40 dans le plan croissance / marge."])
    suite("Étape 6 : la dimension extra-financière (ESG).")

# =========================================================================== #
# 8. ESG
# =========================================================================== #
with tabs[5]:
    intro("<b>Étape 6 – ESG.</b> Les scores ESG ne figurent pas dans le fichier Excel : ils proviennent d'une "
          "source externe, citée titre par titre (fichier <code>data/esg_scores.csv</code>).")
    if not HAS_ESG:
        st.warning("Scores ESG en attente : déposer `data/esg_scores.csv` (même format que le modèle ci-dessous, "
                   "une ligne par ticker, avec la source). Tous les graphiques de cet onglet et la colonne ESG "
                   "de la synthèse s'activeront automatiquement.")
        tpl = df_full[["ticker", "name"]].assign(esg_score="", esg_scale="risk ou score", esg_provider="",
                                                 esg_date="", source_url="")
        st.download_button("Télécharger le modèle CSV", tpl.to_csv(index=False).encode(), "esg_scores.csv", "text/csv")
    else:
        prov = df_full["esg_provider"].dropna().unique()
        scale = df_full["esg_scale"].dropna().unique()
        lower_better = len(scale) and scale[0] == "risk"
        st.caption(f"Fournisseur(s) : {', '.join(map(str, prov))}. "
                   + ("Échelle de risque : plus le score est bas, meilleur est le profil ESG."
                      if lower_better else "Plus le score est élevé, meilleur est le profil ESG."))
        n_old = int(df_full["esg_provider"].fillna("").str.contains("2024").sum())
        n_missing = int(df_full["esg"].isna().sum())
        st.caption(f"Source principale : page publique S&P Global de chaque société (score {df_full['esg'].notna().sum() - n_old} "
                   f"titres, 2025–2026). {n_old} titres sans page publique : score CSA S&P Global du 22/11/2024 (document "
                   f"officiel S&P de rééquilibrage du DJSI). {n_missing} titres sans score public trouvé : ils restent dans "
                   "toutes les analyses mais ne sont pas éligibles quand le filtre ESG du portefeuille est activé.")
        de = df.dropna(subset=["esg"])
        k = st.columns(3)
        k[0].metric("Titres couverts", f"{len(de)} / {len(df)}")
        k[1].metric("Score moyen", f"{de.esg.mean():.1f}")
        k[2].metric("Score médian", f"{de.esg.median():.1f}")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(de, x="sector", y="esg", color="sector", points="all", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE, labels={"esg": lab("esg")},
                         title="Score ESG par secteur")
            fig.update_layout(showlegend=False)
            show(fig, c1)
        with c2:
            em = de.groupby(["sector", "subindustry"]).esg.median().reset_index().sort_values("esg")
            fig = px.bar(em, x="esg", y="subindustry", color="sector", orientation="h",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                         labels={"esg": "Score ESG médian", "subindustry": ""}, title="Score ESG médian par sous-industrie")
            show(fig, c2)
        st.markdown("**Lien ESG / rendement / risque**")
        yv = st.selectbox("Comparer l'ESG à", ["tr_prev_year", "ret_ytd", "vol", "sharpe_ante", "upside", "pe_next"],
                          format_func=lab, key="esg_y")
        f_ = du.ols_fit(de["esg"], de[yv])
        fig = px.scatter(de, x="esg", y=yv, color="sector", hover_name="ticker", trendline=None,
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE, labels={"esg": lab("esg"), yv: lab(yv)},
                         title=f"ESG vs {lab(yv)} (pente {f_['slope']:.2f}, Spearman {f_['spearman']:.2f}, R² {f_['r2']:.2f})")
        xs = np.linspace(de.esg.min(), de.esg.max(), 20)
        fig.add_trace(go.Scatter(x=xs, y=f_["slope"] * xs + f_["intercept"], mode="lines", name="Régression",
                                 line=dict(color="#333", dash="dash")))
        show(fig)
        de = de.assign(quintile=pd.qcut(de.esg.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]))
        qt = de.groupby("quintile", observed=True)[["tr_prev_year", "ret_ytd", "vol", "sharpe_ante", "upside"]].mean()
        st.markdown("**Rendement et risque moyens par quintile ESG** (Q1 = scores les plus bas)")
        st.dataframe(qt.rename(columns=du.LABELS).style.format("{:.2f}"), width="stretch")
        st.markdown("**Préparation de l'interrupteur ESG : au-dessus ou en dessous de la médiane de sa sous-industrie**")
        st.caption("Aucun titre n'est écarté ici. Ce repère servira seulement dans le portefeuille final, en mode ESG activé.")
        ab = df.assign(statut=np.select([df.esg_above == 1, df.esg_above == 0], ["Au-dessus de la médiane",
                                         "Égal ou en dessous"], "Non noté"))
        ct_ = ab.groupby(["subindustry", "statut"]).size().reset_index(name="n")
        fig = px.bar(ct_, x="n", y="subindustry", color="statut", orientation="h", template=TEMPLATE,
                     color_discrete_map={"Au-dessus de la médiane": "#00A19A", "Égal ou en dessous": "#5FB3E4",
                                         "Non noté": "#c8c8c8"},
                     labels={"n": "Nombre de titres", "subindustry": "", "statut": "Score ESG"},
                     title="Titres au-dessus de la médiane ESG de leur sous-industrie")
        fig.update_layout(height=560)
        show(fig)
        _sp = du.ols_fit(de["esg"], de["tr_prev_year"])["spearman"]
        _em = de.groupby("sector").esg.median()
        recap([f"{len(de)} titres sur {len(df)} ont un score ESG sourcé (S&P Global, échelle 0–100).",
               "Score médian : " + " ; ".join(f"{i} {v:.0f}" for i, v in _em.items()) + ".",
               f"Corrélation de rang ESG / total return N-1 : {_sp:.2f} : "
               + ("lien faible, l'ESG n'a pas « coûté » ni « rapporté » nettement sur la période." if abs(_sp) < 0.2
                  else ("les meilleurs scores ESG ont plutôt mieux performé." if _sp > 0
                        else "les meilleurs scores ESG ont plutôt moins bien performé.")),
               f"{int((df.esg_above == 1).sum())} titres sont au-dessus de la médiane de leur sous-industrie."],
              ["Score ESG S&P Global : évaluation relative à l'industrie (Corporate Sustainability Assessment).",
               "Médiane intra-sous-industrie : on compare un titre à ses pairs, pas un opérateur télécom à un éditeur de logiciels.",
               "Quintiles : 5 groupes de taille égale classés par score."])
        st.markdown("**Sources**")
        st.dataframe(df_full.loc[df_full.esg.notna(), ["ticker", "name", "esg", "esg_provider", "esg_date", "source_url"]],
                     hide_index=True, width="stretch",
                     column_config={"source_url": st.column_config.LinkColumn("Source")})
    suite("Étape 7 : vue d'ensemble des liens entre tous les facteurs.")

# =========================================================================== #
# 9. Relations entre facteurs
# =========================================================================== #
with tabs[6]:
    intro("<b>Étape 7 – Relations entre facteurs.</b> Quels facteurs vont ensemble ? Utile pour construire un "
          "critère agrégé sans compter deux fois la même information.")
    c1, c2 = st.columns([1, 3])
    with c1:
        method = st.radio("Corrélation", ["spearman", "pearson"],
                          format_func={"spearman": "Spearman (rangs, robuste)", "pearson": "Pearson (linéaire)"}.get)
        default_corr = du.NUMERIC + ["vol", "sharpe_ante"] + ESG_COLS
        corr_cols = st.multiselect("Variables", ALL_NUM, default=default_corr, format_func=lab, key="corr_cols")
    with c2:
        if len(corr_cols) >= 2:
            cm = df[corr_cols].corr(method=method).rename(index=du.LABELS, columns=du.LABELS)
            fig = px.imshow(cm, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                            color_continuous_scale="RdBu_r", title=f"Matrice de corrélation ({method})")
            fig.update_layout(height=700)
            show(fig, c2)
    if len(corr_cols) >= 2:
        cm_raw = df[corr_cols].corr(method=method)
        pairs = (cm_raw.where(np.triu(np.ones(cm_raw.shape, dtype=bool), 1)).stack().rename("corr").reset_index())
        pairs["|corr|"] = pairs["corr"].abs()
        pairs = pairs.sort_values("|corr|", ascending=False).head(10)
        pairs["Paire"] = pairs["level_0"].map(lab) + "  ×  " + pairs["level_1"].map(lab)
        st.markdown("**Les 10 relations les plus fortes**")
        st.dataframe(pairs[["Paire", "corr"]].style.format({"corr": "{:.2f}"}), hide_index=True, width="stretch")

    st.markdown("**Explorer une relation**")
    c1, c2, c3 = st.columns(3)
    xv = c1.selectbox("X", ALL_NUM, index=ALL_NUM.index("ltg_eps"), format_func=lab)
    yv = c2.selectbox("Y", ALL_NUM, index=ALL_NUM.index("upside"), format_func=lab)
    by_sector = c3.checkbox("Une régression par secteur", value=False)
    dd_all = df.dropna(subset=[xv, yv])
    fig = px.scatter(dd_all, x=xv, y=yv, color="sector", hover_name="ticker",
                     color_discrete_map=SECTOR_COLORS, template=TEMPLATE, labels={xv: lab(xv), yv: lab(yv)})
    groups = [(s_, dd_all[dd_all.sector == s_]) for s_ in dd_all.sector.unique()] if by_sector else [("Univers", dd_all)]
    stats = []
    for name, dd in groups:
        if len(dd) < 3:
            continue
        f_ = du.ols_fit(dd[xv], dd[yv])
        xs = np.linspace(dd[xv].min(), dd[xv].max(), 20)
        fig.add_trace(go.Scatter(x=xs, y=f_["slope"] * xs + f_["intercept"], mode="lines", name=f"MCO {name}",
                                 line=dict(width=2, dash="dash", color=SECTOR_COLORS.get(name, "#333"))))
        stats.append({"Groupe": name, "n": f_["n"], "Pente": f_["slope"], "R²": f_["r2"],
                      "Pearson": f_["pearson"], "Spearman": f_["spearman"]})
    fig.update_layout(height=550)
    show(fig)
    st.dataframe(pd.DataFrame(stats).style.format({"Pente": "{:.3f}", "R²": "{:.2f}", "Pearson": "{:.2f}",
                                                  "Spearman": "{:.2f}"}), hide_index=True, width="stretch")

    st.markdown("**Matrice de nuages de points**")
    spm = st.multiselect("Variables (3 à 5 conseillées)", ALL_NUM,
                         default=["upside", "ltg_eps", "pe_next", "vol"], format_func=lab, key="spm")
    if 2 <= len(spm) <= 6:
        fig = px.scatter_matrix(df, dimensions=spm, color="sector", hover_name="ticker",
                                color_discrete_map=SECTOR_COLORS, labels=du.LABELS, template=TEMPLATE)
        fig.update_traces(diagonal_visible=False, marker=dict(size=5))
        fig.update_layout(height=700)
        show(fig)
    if len(corr_cols) >= 2 and len(pairs):
        _p0 = pairs.iloc[0]
        recap([f"Relation la plus forte : {_p0['Paire']} ({_p0['corr']:.2f}).",
               "Des facteurs très corrélés portent la même information : dans le critère agrégé, ils sont regroupés "
               "dans un même pilier pour ne pas la compter deux fois."],
              ["Corrélation de Pearson (linéaire) et de Spearman (sur les rangs, robuste aux extrêmes).",
               "Régression linéaire par moindres carrés (MCO) : pente, R² = part de variance expliquée."])
    suite("Étape 8 : le critère agrégé, comparé à la Rule of 40.")

# =========================================================================== #
# 10. Fiche titre
# =========================================================================== #
with tabs[10]:
    intro("<b>Étape 11 – Un titre face à son secteur.</b> Position en percentiles sur les principaux critères.")
    choice = st.selectbox("Titre", df.ticker + " – " + df.name)
    tk = choice.split(" – ")[0]
    row = df[df.ticker == tk].iloc[0]
    st.markdown(f"**{row['name']}** · {row['sector']} · {row['subindustry']} · {row['country']}")
    prof_cols = ["upside", "ltg_eps", "sales_growth", "profit_margin", "fcf_yield", "earnings_yield",
                 "sharpe_ante", "tr_prev_year", "ret_ytd"] + ESG_COLS
    pct_all = df[prof_cols].rank(pct=True).loc[row.name] * 100
    peers = df[df.sector == row.sector]
    pct_sec = peers[prof_cols].rank(pct=True).loc[row.name] * 100
    theta = [lab(c) for c in prof_cols]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=pct_all.values, theta=theta, fill="toself", name="Percentile univers",
                                  line_color="#0062A3"))
    fig.add_trace(go.Scatterpolar(r=pct_sec.values, theta=theta, fill="toself", name="Percentile secteur",
                                  line_color="#5FB3E4", opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=520,
                      title="Profil en percentiles (100 = valeur la plus élevée de la sélection)")
    c1, c2 = st.columns([3, 2])
    show(fig, c1)
    vcols = du.NUMERIC + ["rule_of_40", "vol", "mu", "sharpe_ante", "sharpe_post"] + list(sc.PILLARS) + ["Score final"] + ESG_COLS
    vals = pd.DataFrame({"Valeur": row[vcols].astype(float), "Médiane secteur": peers[vcols].median(),
                         "Médiane univers": df[vcols].median()})
    vals.index = [lab(c) for c in vals.index]
    c2.dataframe(vals.style.format("{:.2f}"), width="stretch", height=520)
    suite("Étape 12 : le portefeuille final, avec ou sans filtre ESG.")

# =========================================================================== #
# Annexe : données & méthodes
# =========================================================================== #
with tabs[12]:
    st.subheader("Dictionnaire des colonnes")
    dico = pd.DataFrame(COLUMNS).rename(columns={
        "col": "Colonne Excel", "code": "Nom interne", "famille": "Famille", "unite": "Unité",
        "definition": "Définition", "lecture": "Comment le lire"})
    st.dataframe(dico, hide_index=True, width="stretch")

    st.subheader("Variables dérivées")
    st.dataframe(pd.DataFrame(DERIVED_DOC, columns=["Variable", "Formule", "Intérêt"]), hide_index=True, width="stretch")

    st.subheader("Méthode d'estimation de la volatilité")
    t1, t2, t3 = an.period_lengths(pd.Timestamp(data_date))
    st.markdown(f"""
Le fichier ne contient pas d'historique de prix, mais il donne **trois rendements sur des périodes qui ne se
chevauchent pas** :

| Période | Calcul à partir du fichier | Durée |
| --- | --- | --- |
| Année civile N-1 | colonne *Previous 1 Calendar Year Total Return* | {t1:.2f} an |
| Du 1er janvier à il y a un mois | (1 + YTD) ÷ (1 + perf 1M) − 1 | {t2:.2f} an |
| Dernier mois | colonne *%1M* | {t3:.3f} an |

On suppose que le log-rendement d'une période de durée T suit une loi normale de moyenne μT et de variance σ²T.
""")
    st.latex(r"\hat\mu = \frac{\sum_k \ell_k}{\sum_k T_k}, \qquad "
             r"\hat\sigma^2 = \frac{1}{3-1}\sum_{k=1}^{3} \frac{(\ell_k - \hat\mu T_k)^2}{T_k}")
    st.markdown("Trois observations, c'est peu : l'estimation est ensuite rapprochée de la médiane de la "
                "sous-industrie (σ = α·σ̂ titre + (1 − α)·médiane sous-industrie, α réglable). L'ancienne méthode, "
                "|perf 1M| × √12, n'utilise qu'une seule observation : un titre très volatil qui finit le mois à 0 % "
                "y paraîtrait sans risque. Elle reste disponible pour comparaison.")

    st.subheader("Statistiques descriptives (données brutes filtrées)")
    desc = df_raw_f[du.NUMERIC].describe(percentiles=[.05, .25, .5, .75, .95]).T
    desc.index = [lab(c) for c in desc.index]
    st.dataframe(desc.style.format("{:.2f}"), width="stretch")

    st.subheader("Valeurs extrêmes (règle de Tukey, 3 × IQR)")
    out = du.outlier_table(df_raw_f, du.NUMERIC, k=3)
    st.dataframe(out.style.format({"Valeur": "{:.2f}", "Borne basse": "{:.2f}", "Borne haute": "{:.2f}"}),
                 hide_index=True, width="stretch")

    st.subheader("Table complète (avec variables calculées)")
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Télécharger (CSV)", df.to_csv(index=False).encode(), "univers_enrichi.csv", "text/csv")

# =========================================================================== #
# 12. Portefeuille final
# =========================================================================== #
with tabs[11]:
    intro("<b>Étape 12 – Le portefeuille final.</b> Sélection des meilleurs scores de chaque sous-industrie (nombre de "
          "titres proportionnel à la taille de la sous-industrie), puis pondération qui <b>maximise le Sharpe ex ante</b> "
          "avec <b>5 % maximum par titre</b>. Comparaison avec le portefeuille <b>équipondéré de tout l'univers</b>. "
          "Le portefeuille est construit sur les 99 titres, indépendamment des filtres de la barre latérale.")
    st.latex(r"\max_{w}\ \frac{w^\top \mu}{\sqrt{w^\top \Sigma w}} \quad \text{s.c.} \quad \sum_i w_i = 1,"
             r"\quad 0 \le w_i \le 5\,\%")
    c1, c2, c3 = st.columns([2, 1, 1])
    esg_on = c1.toggle("🌱 Filtre ESG (score ESG > médiane de la sous-industrie)", value=False, disabled=not HAS_ESG,
                       help="Activé : seuls les titres dont le score ESG dépasse la médiane ESG de leur sous-industrie "
                            "sont éligibles. Désactivé : aucune considération ESG.")
    n_total = c2.slider("Nombre de titres sélectionnés", 20, 40, 30, 1,
                        help="Au moins 20 : avec 5 % maximum par titre, il faut 20 lignes pour investir 100 %.")
    cap_p = c3.slider("Plafond par titre", 0.03, 0.10, 0.05, 0.01, format="%.2f")
    if n_total * cap_p < 1:
        st.error("Pas assez de titres pour investir 100 % avec ce plafond.")
        st.stop()
    rhos = (rho_sub, max(min(rho_sec, rho_sub), rho_all), rho_all)
    st.caption(f"Corrélations supposées (onglet 10) : même sous-industrie {rhos[0]:.2f}, même secteur {rhos[1]:.2f}, "
               f"autres {rhos[2]:.2f}. Rendement attendu et volatilité : ceux des onglets 9 et 3.")

    full = base.copy()
    elig_all = pd.Series(True, index=full.index)
    elig_esg = (full["esg_above"] == 1) if HAS_ESG else elig_all
    ports = {"Sans ESG": build_portfolio(full, tuple(elig_all), n_total, cap_p, rhos)}
    if HAS_ESG:
        ports["Avec filtre ESG"] = build_portfolio(full, tuple(elig_esg), n_total, cap_p, rhos)
    cur_name = "Avec filtre ESG" if esg_on else "Sans ESG"
    P = ports[cur_name]

    R_full = an.block_correlation(full, *rhos)
    cov_full = an.covariance(full["vol"], R_full)
    ew = pd.Series(1 / len(full), index=full.index)
    rows = [pf.summary("Équipondéré univers (99 titres)", ew, full, cov_full, rf)]
    for nm, sel in ports.items():
        rows.append(pf.summary(f"Portefeuille {nm[0].lower() + nm[1:]}", sel["Poids"], full, cov_full, rf))
    comp = pd.DataFrame(rows).set_index("Portefeuille").T

    st.subheader(f"Rendu : portefeuille {cur_name[0].lower() + cur_name[1:]}")
    if esg_on:
        n_el = int(elig_esg.sum())
        st.info(f"Filtre ESG activé : {n_el} titres éligibles sur {len(full)} (score ESG strictement supérieur à la "
                f"médiane de leur sous-industrie). Les {int(full['esg'].isna().sum()) if 'esg' in full else 0} titres "
                "sans score ESG sourcé ne sont pas éligibles dans ce mode.")
    cur = comp[[c_ for c_ in comp.columns if cur_name.lower() in c_.lower()][0]]
    ewc = comp.iloc[:, 0]
    k = st.columns(5)
    k[0].metric("Rendement attendu", f"{cur['Rendement attendu (%)']:.1f} %",
                f"{cur['Rendement attendu (%)'] - ewc['Rendement attendu (%)']:+.1f} pt vs équipondéré")
    k[1].metric("Volatilité estimée", f"{cur['Volatilité estimée (%)']:.1f} %",
                f"{cur['Volatilité estimée (%)'] - ewc['Volatilité estimée (%)']:+.1f} pt", delta_color="inverse")
    k[2].metric("Sharpe ex ante", f"{cur['Sharpe ex ante']:.2f}", f"{cur['Sharpe ex ante'] - ewc['Sharpe ex ante']:+.2f}")
    k[3].metric("Lignes", f"{int(cur['Lignes'])}", f"nb effectif {cur['Nb effectif (1/Σw²)']:.1f}", delta_color="off")
    k[4].metric("Score final moyen", f"{cur['Score final moyen']:.2f}",
                f"{cur['Score final moyen'] - ewc['Score final moyen']:+.2f}")

    c1, c2 = st.columns([3, 2])
    with c1:
        Pw = P.sort_values("Poids")
        fig = px.bar(Pw, x=Pw["Poids"] * 100, y="ticker", color="sector", orientation="h",
                     color_discrete_map=SECTOR_COLORS, template=TEMPLATE, hover_data={"name": True, "subindustry": True},
                     labels={"x": "Poids (%)", "ticker": ""}, title="Poids des titres retenus")
        fig.add_vline(x=cap_p * 100, line_dash="dash", line_color=C_RED, annotation_text="plafond")
        fig.update_layout(height=max(500, 19 * len(Pw) + 120))
        show(fig, c1)
    with c2:
        sw = pd.DataFrame({"Portefeuille": P.groupby("subindustry")["Poids"].sum() * 100,
                           "Équipondéré univers": full.subindustry.value_counts(normalize=True) * 100}).fillna(0)
        sw = sw.sort_values("Équipondéré univers")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=sw.index, x=sw["Équipondéré univers"], name="Équipondéré univers", orientation="h",
                             marker_color=C_PALE))
        fig.add_trace(go.Bar(y=sw.index, x=sw["Portefeuille"], name="Portefeuille", orientation="h",
                             marker_color=C_NAVY))
        fig.update_layout(barmode="group", template=TEMPLATE, height=max(500, 19 * len(Pw) + 120),
                          xaxis_title="Poids (%)", title="Poids par sous-industrie", legend=dict(orientation="h", y=-0.1))
        show(fig, c2)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=full["vol"], y=full["mu"], mode="markers", name="Titres de l'univers",
                                 text=full["ticker"], marker=dict(color=C_PALE, size=7),
                                 hovertemplate="%{text}<br>vol %{x:.1f} %<br>rdt %{y:.1f} %<extra></extra>"))
        fig.add_trace(go.Scatter(x=P["vol"], y=P["mu"], mode="markers", name="Titres retenus", text=P["ticker"],
                                 marker=dict(color=C_BLUE, size=6 + 300 * P["Poids"]),
                                 hovertemplate="%{text}<br>vol %{x:.1f} %<br>rdt %{y:.1f} %<extra></extra>"))
        for col_name, sym, colr in [(comp.columns[0], "diamond", C_GREY)] + \
                [(c_, "star", C_RED if "esg" not in c_.lower() or "sans" in c_.lower() else C_TEAL)
                 for c_ in comp.columns[1:]]:
            fig.add_trace(go.Scatter(x=[comp.loc["Volatilité estimée (%)", col_name]],
                                     y=[comp.loc["Rendement attendu (%)", col_name]], mode="markers", name=col_name,
                                     marker=dict(symbol=sym, size=18, color=colr, line=dict(color="white", width=1.5))))
        fig.update_layout(template=TEMPLATE, height=520, xaxis_title=lab("vol"), yaxis_title=lab("mu"),
                          title="Où se situent les portefeuilles ?", xaxis_range=[0, min(full["vol"].max() * 1.05, 120)])
        show(fig, c1)
    with c2:
        rc = P.sort_values("Contribution au risque", ascending=False).head(15)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rc.ticker, y=rc["Poids"] * 100, name="Poids (%)", marker_color=C_PALE))
        fig.add_trace(go.Bar(x=rc.ticker, y=rc["Contribution au risque"] * 100, name="Contribution au risque (%)",
                             marker_color=C_NAVY))
        fig.update_layout(barmode="group", template=TEMPLATE, height=520,
                          title="Poids vs contribution au risque (15 premières contributions)")
        show(fig, c2)

    st.subheader("Comparaison des portefeuilles")
    st.dataframe(comp.style.format("{:.2f}"), width="stretch", height=740)
    st.caption("Les performances N-1 et YTD sont celles qu'auraient eues ces titres dans le passé, avec les poids "
               "d'aujourd'hui : c'est une description, pas un backtest (la sélection utilise des données de la même date).")

    st.subheader("Composition détaillée")
    det_cols = ["ticker", "name", "sector", "subindustry", "country", "Poids", "Score final", "mu", "vol",
                "sharpe_ante", "rule_of_40", "Raison", "Quota sous-industrie"] + (["esg", "esg_med_sub"] if HAS_ESG else [])
    det = P.sort_values("Poids", ascending=False)[det_cols].copy()
    det["Poids"] = det["Poids"] * 100
    det = det.rename(columns={"Poids": "Poids (%)", "mu": lab("mu"), "vol": lab("vol"), "sharpe_ante": lab("sharpe_ante"),
                              "rule_of_40": "Rule of 40", "esg": "Score ESG", "esg_med_sub": "Médiane ESG sous-ind."})
    zero = det[det["Poids (%)"] <= 0]
    det = det[det["Poids (%)"] > 0]
    st.dataframe(det.style.format(precision=2), hide_index=True, width="stretch")
    if len(zero):
        st.caption(f"Sélectionnés mais pondérés à 0 % par l'optimiseur (ils n'amélioraient pas le Sharpe du "
                   f"portefeuille) : {', '.join(zero.ticker)}.")
    st.download_button("Télécharger la composition (CSV)", det.to_csv(index=False).encode(),
                       f"portefeuille_{'esg' if esg_on else 'sans_esg'}.csv", "text/csv")

    if HAS_ESG:
        a_, b_ = set(ports["Sans ESG"].ticker), set(ports["Avec filtre ESG"].ticker)
        st.subheader("Ce que change le filtre ESG")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Communs aux deux ({len(a_ & b_)})**  \n" + ", ".join(sorted(a_ & b_)))
        c2.markdown(f"**Seulement sans ESG ({len(a_ - b_)})**  \n" + ", ".join(sorted(a_ - b_)))
        c3.markdown(f"**Seulement avec ESG ({len(b_ - a_)})**  \n" + ", ".join(sorted(b_ - a_)))
        s0, s1 = comp.iloc[:, 1], comp.iloc[:, 2]
        st.markdown(f"Coût (ou gain) du filtre ESG : Sharpe ex ante {s0['Sharpe ex ante']:.2f} → {s1['Sharpe ex ante']:.2f}, "
                    f"rendement attendu {s0['Rendement attendu (%)']:.1f} % → {s1['Rendement attendu (%)']:.1f} %, "
                    f"score ESG moyen {s0['Score ESG moyen']:.1f} → {s1['Score ESG moyen']:.1f}.")

    with st.expander("Méthode de construction, étape par étape", expanded=False):
        st.markdown(f"""
1. **Univers éligible** : les 99 titres, ou, filtre ESG activé, ceux dont le score ESG dépasse la médiane ESG de leur
   sous-industrie (médiane calculée sur les titres notés). Aucun titre n'est écarté en amont pour son ESG quand le filtre est coupé.
2. **Quotas** : {n_total} places réparties entre les 19 sous-industries proportionnellement à leur nombre de titres
   (méthode des plus forts restes). Une sous-industrie de 2 titres sur 99 n'obtient en général aucune place réservée.
3. **Sélection** : dans chaque sous-industrie, les meilleurs scores finaux, **seulement s'ils dépassent la médiane de
   leur sous-industrie** (on ne force pas un titre faible pour remplir un quota). Les places libres vont aux meilleurs
   scores restants de l'univers, toujours au-dessus de la médiane de leur groupe ; la colonne « Raison » l'indique.
4. **Pondération** : maximisation du Sharpe ex ante μ/σ du portefeuille, sans vente à découvert, {cap_p:.0%} maximum par
   titre (optimiseur SLSQP, plusieurs points de départ). Le rendement attendu μ combine prime de risque, upside et
   z-scores (onglet 9) ; la covariance combine les volatilités estimées et les corrélations supposées par blocs.
5. **Comparaison** avec le portefeuille équipondéré des 99 titres sur les mêmes indicateurs.
""")
    best = comp.loc["Sharpe ex ante"].idxmax()
    recap([f"Portefeuille {cur_name[0].lower() + cur_name[1:]} : {int(cur['Lignes'])} lignes, rendement attendu "
           f"{cur['Rendement attendu (%)']:.1f} %, volatilité {cur['Volatilité estimée (%)']:.1f} %, Sharpe ex ante "
           f"{cur['Sharpe ex ante']:.2f} contre {ewc['Sharpe ex ante']:.2f} pour l'équipondéré.",
           f"Score final moyen {cur['Score final moyen']:.2f} (équipondéré {ewc['Score final moyen']:.2f}) ; "
           f"{cur['Part passant la Rule of 40 (%)']:.0f} % du portefeuille passe la Rule of 40 "
           f"(équipondéré {ewc['Part passant la Rule of 40 (%)']:.0f} %).",
           f"Diversification : {int(cur['Sous-industries'])} sous-industries, poids max. d'une sous-industrie "
           f"{cur['Poids max. d\'une sous-industrie (%)']:.0f} %, part technologie {cur['Part technologie (%)']:.0f} %.",
           f"Meilleur Sharpe ex ante des trois : {best}."],
          ["Optimisation de Markowitz : maximisation du ratio de Sharpe (portefeuille tangent) sous contraintes.",
           "Nombre effectif de titres 1/Σw² : nombre de lignes équipondérées qui donnerait la même concentration.",
           "Contribution au risque w_i(Σw)_i / wᵀΣw : part de la variance du portefeuille due à chaque titre."])
