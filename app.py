"""Application Streamlit : analyse de l'univers « universe of stock 261113 » et construction de portefeuille.

Lancer en local :  streamlit run app.py
"""
from __future__ import annotations

import base64
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

st.set_page_config(page_title="Univers d'actions – Analyse et portefeuille", page_icon="📊", layout="wide")

# --------------------------------------------------------------------------- #
# Charte graphique (bleus Amundi) et logo
# --------------------------------------------------------------------------- #
C_NAVY, C_BLUE, C_LIGHT, C_TEAL, C_PALE, C_RED, C_GREY = (du.C_NAVY, du.C_BLUE, du.C_LIGHT, du.C_TEAL,
                                                            du.C_PALE, du.C_RED, du.C_GREY)
LOGO_FILE = Path(__file__).parent / "assets" / "logo_amundi.png"
if LOGO_FILE.exists():
    LOGO_URL = "data:image/png;base64," + base64.b64encode(LOGO_FILE.read_bytes()).decode()
    LOGO_SRC = str(LOGO_FILE)
else:
    LOGO_URL = LOGO_SRC = "https://static.amundi.com/web-assets/awf-assets/corporate/logo.png"
st.logo(LOGO_SRC, size="large", link="https://about.amundi.com/")
st.markdown(f"""
<style>
h1, h2, h3 {{ color: {C_NAVY}; }}
div[data-testid="stMetricValue"] {{ color: {C_NAVY}; }}
div[data-testid="stMetric"] {{ background: #F3F7FC; border-left: 4px solid {C_BLUE};
    padding: .5rem .8rem; border-radius: 6px; }}
/* barre d'onglets figée en haut de page */
div[data-testid="stTabs"] div[role="tablist"] {{ position: sticky; top: 3.6rem; z-index: 999;
    background: white; padding-top: .3rem; border-bottom: 1px solid #DCE6F2; }}
button[role="tab"] p {{ font-weight: 600; }}
.amundi-band {{ background: linear-gradient(90deg, {C_NAVY} 0%, {C_BLUE} 70%, {C_LIGHT} 100%);
    color: white; border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 1.2rem; }}
.amundi-band img {{ height: 46px; background: white; border-radius: 6px; padding: 6px 10px;
    color: {C_NAVY}; font-weight: 700; font-size: 1.3rem; }}
.amundi-band h1 {{ color: white; margin: 0; font-size: 1.9rem; }}
.amundi-band p {{ color: #DCE9F7; margin: .2rem 0 0 0; }}
.recap {{ background: #EEF4FB; border: 1px solid #C9DBF0; border-radius: 8px; padding: .8rem 1.1rem;
    margin: 1.2rem 0 .4rem 0; }}
.recap b.t {{ display: block; margin-bottom: .3rem; color: {C_NAVY}; }}
</style>""", unsafe_allow_html=True)

PLOT_CFG = {"displaylogo": False}
TEMPLATE = "plotly_white"
SECTOR_COLORS = du.SECTOR_COLORS
FR = {"sector": "Secteur", "subindustry": "Sous-industrie", "country": "Pays", "ticker": "Ticker",
      "variable": "Variable", "value": "Valeur"}
GROUP_FR = {"sector": "Secteur", "subindustry": "Sous-industrie", "country": "Pays"}
DATA_DATE = pd.Timestamp("2026-11-13")   # date des données (nom du fichier : 261113)
ALPHA_VOL = 0.5                          # poids de l'estimation individuelle de volatilité
K_SHRINK = 5.0                           # rétrécissement des z-scores vers le secteur
RHO = an.RHO_CENTRAL


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


def recap(points: list[str], methode: list[str] | None = None):
    """Encadré de fin d'onglet : constats chiffrés, et méthode de calcul quand elle est spécifique."""
    html = "<div class='recap'><b class='t'>Points clés</b><ul style='margin:0'>" + \
           "".join(f"<li>{x}</li>" for x in points) + "</ul>"
    if methode:
        html += "<b class='t' style='margin-top:.6rem'>Méthode</b><ul style='margin:0'>" + \
                "".join(f"<li>{x}</li>" for x in methode) + "</ul>"
    st.markdown(html + "</div>", unsafe_allow_html=True)


def ols_line(fig, x: pd.Series, y: pd.Series, name: str = "Régression linéaire", color: str = "#333"):
    f_ = du.ols_fit(x, y)
    xs = np.linspace(x.min(), x.max(), 30)
    fig.add_trace(go.Scatter(x=xs, y=f_["slope"] * xs + f_["intercept"], mode="lines", name=name,
                             line=dict(color=color, dash="dash", width=1.8)))
    return f_


@st.cache_data
def get_data():
    raw = du.load_raw()
    df = du.prepare(raw)
    esg = du.load_esg()
    if esg is not None:
        df = df.merge(esg[["ticker", "esg_score", "esg_scale", "esg_provider", "esg_date", "source_url"]],
                      on="ticker", how="left").rename(columns={"esg_score": "esg"})
    return raw, df, esg


@st.cache_data(show_spinner="Calcul de la frontière efficiente…")
def frontier_cached(mu: tuple, cov_flat: tuple, n: int, cap: float):
    cov = np.array(cov_flat).reshape(n, n)
    return an.efficient_frontier(np.array(mu), cov, cap, n_points=22)


@st.cache_data(show_spinner="Optimisation du portefeuille (Sharpe maximal)…")
def build_portfolio(d: pd.DataFrame, eligible: tuple, n_total: int, cap: float, rhos: tuple,
                    rescue: tuple | None = None):
    res = None if rescue is None else pd.Series(rescue, index=d.index)
    sel = pf.select(d, n_total, pd.Series(eligible, index=d.index), rescue=res)
    R = an.block_correlation(sel, *rhos)
    cov = an.covariance(sel["vol"], R)
    w = pf.max_sharpe(sel["mu"].to_numpy(float), cov, cap)
    sel["Poids"] = w
    sel["Contribution au risque"] = pf.risk_contributions(w, cov)
    return sel


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

st.sidebar.header("Paramètres")
wins = st.sidebar.checkbox("Écrêter les valeurs extrêmes (winsorisation)", value=True,
                           help="Chaque variable est ramenée dans ses quantiles [p ; 1 − p]. Aucun titre n'est supprimé.")
p = st.sidebar.slider("Quantile d'écrêtage p", 0.0, 0.10, 0.02, 0.01, disabled=not wins)
rf = st.sidebar.number_input("Taux sans risque (%)", 0.0, 10.0, an.RF_DEFAULT, 0.01,
                             help="T-bill américain 3 mois au 13/11/2026 : 4,32 % (Réserve fédérale, série FRED DGS3MO).")
lam = st.sidebar.slider("Part de l'upside brut dans le niveau de rendement (λ)", 0.0, 1.0, 0.0, 0.05,
                        help="0 : niveau = prime de risque implicite, l'upside n'intervient que relativement aux pairs "
                             "(z-score). 0,5 : moyenne 50/50 prime / upside brut.")
ic = st.sidebar.slider("Coefficient d'information (IC)", 0.0, 0.30, 0.10, 0.01,
                       help="Corrélation supposée entre le score et le rendement futur ; convertit le score en points de "
                            "rendement : alpha = IC × volatilité × score standardisé (règle de Grinold).")
n_total = st.sidebar.slider("Portefeuille : nombre de titres sélectionnés", 20, 40, 30, 1,
                            help="Au moins 20 : avec 5 % maximum par titre, 20 lignes sont nécessaires pour investir 100 %.")
CAP = 0.05

# --------------------------------------------------------------------------- #
# Calculs sur l'univers complet
# --------------------------------------------------------------------------- #
ALL_NUM = du.NUMERIC + du.DERIVED
base = du.winsorize(df_full, ALL_NUM, p, 1 - p) if wins else df_full.copy()
vol_obs = an.multi_period_vol(df_full, DATA_DATE)
base["vol_obs"] = vol_obs
base["vol"] = an.shrink_to_group(vol_obs, df_full["subindustry"], ALPHA_VOL)
ZF = sc.factor_zscores(base, level="subindustry", k=K_SHRINK)
SCORES = sc.composite_score(ZF)
PILLARS = list(sc.PILLARS)
for c_ in SCORES.columns:
    base[c_] = SCORES[c_]
MU_PARTS = sc.expected_return(base, SCORES["Score alpha"], base["vol"], rf, lam, ic)
MU_COMP = ["Ancrage prime de risque", "Ancrage upside (niveau)", "Alpha z-scores"]
for c_ in MU_COMP + ["mu"]:
    base[c_] = MU_PARTS[c_]
MU_5050 = sc.expected_return(base, SCORES["Score alpha"], base["vol"], rf, 0.5, ic)["mu"]
base["sharpe_ante"] = base["mu"] / base["vol"]
base["sharpe_post"] = (base["tr_prev_year"] - rf) / base["vol"]
RISK = ["vol", "mu", "sharpe_ante", "sharpe_post", "Score final"]
du.LABELS.setdefault("Score final", "Score final (critère agrégé)")
ESG_COLS = ["esg"] if HAS_ESG else []
if HAS_ESG:
    base["esg_med_sub"] = base.groupby("subindustry")["esg"].transform("median")
    base["esg_above"] = np.where(base["esg"].isna(), np.nan, (base["esg"] > base["esg_med_sub"]).astype(float))
ALL_NUM = ALL_NUM + RISK + ESG_COLS

# Portefeuilles (univers complet, indépendants des filtres)
full = base.copy()
elig_all = pd.Series(True, index=full.index)
ports = {"Sans ESG": build_portfolio(full, tuple(elig_all), n_total, CAP, RHO)}
if HAS_ESG:
    elig_esg = full["esg_above"] == 1
    rescue = full["esg"] >= full["esg"].median()
    ports["Avec filtre ESG"] = build_portfolio(full, tuple(elig_esg), n_total, CAP, RHO, tuple(rescue))
COV_FULL = an.covariance(full["vol"], an.block_correlation(full, *RHO))
EW = pd.Series(1 / len(full), index=full.index)

mask = df_full.sector.isin(sectors) & df_full.subindustry.isin(subs) & df_full.country.isin(countries)
df = base.loc[mask].copy()
df_raw_f = df_full.loc[mask].copy()
st.sidebar.caption(f"{len(df)} titres sélectionnés sur {len(df_full)}. Le portefeuille final porte toujours sur les "
                   f"{len(df_full)} titres.")
if len(df) < 3:
    st.warning("Sélection trop restreinte : élargir les filtres.")
    st.stop()

st.markdown(f"""<div class="amundi-band"><img src="{LOGO_URL}" alt="Amundi">
<div><h1>Univers d'actions : analyse et construction de portefeuille</h1>
<p>99 valeurs Technologie et Communication</p></div></div>""", unsafe_allow_html=True)
st.caption("Données : fichier « universe of stock 261113 » (99 titres, 17 colonnes), au 13/11/2026"
           + (" ; scores ESG S&P Global sourcés titre par titre." if HAS_ESG else "."))

tabs = st.tabs([
    "1 · Univers", "2 · Synthèse", "3 · Rendement & risque", "4 · Valorisation & attentes",
    "5 · Relations entre facteurs", "6 · Rule of 40 & critère agrégé", "7 · ESG",
    "8 · Rendement attendu & Sharpe", "9 · Frontière efficiente", "10 · Portefeuille final", "Annexe · Méthodes",
])

# =========================================================================== #
# 1. Univers
# =========================================================================== #
with tabs[0]:
    k = st.columns(4)
    k[0].metric("Titres", len(df))
    k[1].metric("Secteurs", df.sector.nunique())
    k[2].metric("Sous-industries", df.subindustry.nunique())
    k[3].metric("Pays", df.country.nunique())

    c1, c2 = st.columns(2)
    with c1:
        sec = df.sector.value_counts().rename_axis("sector").reset_index(name="n")
        fig = px.pie(sec, names="sector", values="n", color="sector", color_discrete_map=SECTOR_COLORS,
                     hole=0.45, title="Répartition par secteur")
        fig.update_traces(textinfo="label+percent+value", marker_line_color="white", marker_line_width=2)
        fig.update_layout(showlegend=False)
        show(fig, c1)
    with c2:
        fig = px.sunburst(df.assign(n=1), path=["sector", "subindustry"], values="n", color="sector",
                          color_discrete_map=SECTOR_COLORS, title="Secteur → sous-industrie")
        fig.update_traces(insidetextorientation="radial", marker_line_color="white")
        show(fig, c2)

    subc = df.groupby(["sector", "subindustry"]).size().reset_index(name="n").sort_values("n")
    fig = px.bar(subc, x="n", y="subindustry", color="sector", orientation="h",
                 color_discrete_map=SECTOR_COLORS, template=TEMPLATE, text="n",
                 labels={"n": "Nombre de titres", "subindustry": ""}, title="Nombre de titres par sous-industrie")
    fig.update_layout(height=560)
    show(fig)

    c1, c2 = st.columns(2)
    with c1:
        cnt = df.country.value_counts()
        top = cnt[cnt >= 3]
        pays = pd.concat([top, pd.Series({"Autres pays": cnt[cnt < 3].sum()})]).rename_axis("Pays").reset_index(name="n")
        fig = px.pie(pays, names="Pays", values="n", hole=0.45, title="Répartition par pays",
                     color_discrete_sequence=[C_NAVY, C_BLUE, C_LIGHT, C_TEAL, C_PALE, C_GREY, "#2E5E8C", "#C5CED9"])
        fig.update_traces(textinfo="label+percent", marker_line_color="white", marker_line_width=2)
        show(fig, c1)
    with c2:
        zone = np.where(df.country == "UNITED STATES", "États-Unis", "Hors États-Unis")
        xt = pd.crosstab(df.sector, zone)
        fig = px.bar(xt.reset_index().melt(id_vars="sector", var_name="Zone", value_name="n"),
                     x="sector", y="n", color="Zone", barmode="group", text="n", template=TEMPLATE,
                     color_discrete_sequence=[C_BLUE, C_LIGHT], labels={"n": "Nombre de titres"},
                     title="Secteur × zone géographique")
        show(fig, c2)
    _top_sub = df.subindustry.value_counts()
    recap([f"{len(df)} titres : {(df.sector == 'Information Technology').mean():.0%} en technologie, "
           f"{(df.country == 'UNITED STATES').mean():.0%} américains.",
           f"Les 3 premières sous-industries ({', '.join(_top_sub.index[:3])}) représentent "
           f"{_top_sub.iloc[:3].sum() / len(df):.0%} de l'univers ; {(_top_sub <= 2).sum()} sous-industries comptent "
           "2 titres ou moins.",
           "L'équipondéré de l'univers est donc avant tout une exposition technologie américaine : la diversification "
           "du portefeuille se pilote au niveau des sous-industries."])

# =========================================================================== #
# 2. Synthèse
# =========================================================================== #
with tabs[1]:
    synth_cols = ["ret_1m", "ret_ytd", "tr_prev_year", "vol", "sharpe_ante", "sharpe_post", "upside",
                  "pe_next", "peg", "fcf_yield", "ltg_eps", "sales_growth", "profit_margin", "rule_of_40",
                  "risk_prem"] + ESG_COLS
    s = df[synth_cols].agg(["mean", "median", "std", "min", "max"]).T
    s.columns = ["Moyenne", "Médiane", "Écart-type", "Min", "Max"]
    s.index = [lab(c) for c in s.index]
    st.subheader("Statistiques de l'univers")
    st.dataframe(s.style.format("{:.2f}"), width="stretch", height=560)

    st.subheader("Par secteur (moyenne / médiane)")
    g = df.groupby("sector")[synth_cols].agg(["mean", "median"]).T
    g.index = [f"{lab(v)} – {'moyenne' if s_ == 'mean' else 'médiane'}" for v, s_ in g.index]
    st.dataframe(g.style.format("{:.2f}"), width="stretch", height=560)

    med = df.groupby("sector")[synth_cols].median()
    zs = (med - df[synth_cols].median()) / df[synth_cols].std()
    fig = px.imshow(zs.T.rename(index=du.LABELS), aspect="auto", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, text_auto=".2f",
                    title="Profil des secteurs : écart de la médiane du secteur à celle de l'univers (en écarts-types)")
    fig.update_layout(height=600)
    show(fig)
    _gap = df["ret_ytd"].mean() - df["ret_ytd"].median()
    recap([f"Perf. YTD : moyenne {df.ret_ytd.mean():.1f} % contre médiane {df.ret_ytd.median():.1f} % "
           f"({'distribution tirée par quelques très fortes hausses' if _gap > 0 else 'distribution tirée par quelques fortes baisses'}).",
           f"Volatilité estimée médiane : {df.vol.median():.0f} % ; prime de risque médiane : {df.risk_prem.median():.1f} %.",
           f"Upside médian des analystes : {df.upside.median():.1f} % ; marge nette médiane : {df.profit_margin.median():.1f} %."],
          ["Valeurs écrêtées aux quantiles 2 % / 98 % (winsorisation) avant calcul ; le tableau brut est conservé."])

# =========================================================================== #
# 3. Rendement & risque
# =========================================================================== #
with tabs[2]:
    t1, t2, t3 = an.period_lengths(DATA_DATE)
    st.subheader("Volatilité : méthode d'estimation")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"""
Le fichier fournit trois rendements sur des périodes **disjointes** : année civile N-1 ({t1:.2f} an), du 1er janvier
à il y a un mois ({t2:.2f} an, déduit de YTD et de la perf. 1 mois), dernier mois ({t3:.3f} an). Les log-rendements
ℓ<sub>k</sub> d'une période de durée T<sub>k</sub> sont supposés suivre une loi N(μT<sub>k</sub> ; σ²T<sub>k</sub>) :
""", unsafe_allow_html=True)
        st.latex(r"\hat\mu = \frac{\sum_k \ell_k}{\sum_k T_k}, \qquad "
                 r"\hat\sigma^2 = \frac{1}{3-1}\sum_{k=1}^{3} \frac{(\ell_k - \hat\mu T_k)^2}{T_k}, \qquad "
                 r"\sigma_i = 0{,}5\,\hat\sigma_i + 0{,}5\,\text{médiane}_{\text{sous-industrie}}(\hat\sigma)")
        st.markdown("Le rapprochement de la médiane de la sous-industrie réduit le bruit d'une estimation fondée sur "
                    "trois observations.")
    with c2:
        fig = px.scatter(base, x="vol_obs", y="vol", color="sector", hover_name="ticker", template=TEMPLATE,
                         color_discrete_map=SECTOR_COLORS,
                         labels={"vol_obs": "σ̂ du titre seul (%)", "vol": "σ retenue (%)"},
                         title="Effet du rapprochement vers la sous-industrie")
        fig.update_layout(height=330, showlegend=False)
        show(fig, c2)

    grp = st.radio("Niveau d'analyse", ["sector", "subindustry", "country"], horizontal=True,
                   format_func=GROUP_FR.get, key="rr_grp")
    agg = df.groupby(grp).agg(n=("ticker", "size"), ret_1m=("ret_1m", "mean"), ret_ytd=("ret_ytd", "mean"),
                              tr_prev_year=("tr_prev_year", "mean"), vol=("vol", "mean")).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        long = agg.melt(id_vars=grp, value_vars=["ret_1m", "ret_ytd", "tr_prev_year"],
                        var_name="Horizon", value_name="Rendement moyen (%)")
        long["Horizon"] = long["Horizon"].map(lab)
        fig = px.bar(long, x=grp, y="Rendement moyen (%)", color="Horizon", barmode="group", template=TEMPLATE,
                     color_discrete_sequence=[C_PALE, C_LIGHT, C_NAVY], title="Rendement moyen par horizon")
        fig.update_layout(height=450)
        show(fig, c1)
    with c2:
        fig = px.box(df, x=grp, y="vol", color="sector" if grp != "sector" else grp, points="all",
                     hover_name="ticker", color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                     labels={"vol": lab("vol")}, title="Distribution de la volatilité estimée")
        fig.update_layout(height=450, showlegend=grp != "sector")
        show(fig, c2)
        st.caption("Boîte : 1er à 3e quartile, trait = médiane. Moustaches (« fences ») : Q1 − 1,5 × IQR et "
                   "Q3 + 1,5 × IQR, avec IQR = Q3 − Q1 ; les points au-delà sont atypiques.")

    c1, c2 = st.columns(2)
    with c1:
        ret_opts = {"tr_prev_year": lab("tr_prev_year"), "ret_ytd": lab("ret_ytd"),
                    "mu": "Rendement attendu ex ante (%, en excès de r_f, onglet 8)", "upside": lab("upside")}
        ret_var = st.selectbox("Rendement (axe vertical)", list(ret_opts), format_func=ret_opts.get)
        fig = px.scatter(df, x="vol", y=ret_var, color="sector", hover_name="ticker",
                         hover_data={"name": True, "subindustry": True}, color_discrete_map=SECTOR_COLORS,
                         template=TEMPLATE, labels={"vol": lab("vol"), ret_var: ret_opts[ret_var]},
                         title="Rendement vs volatilité")
        f_rv = ols_line(fig, df["vol"], df[ret_var])
        fig.update_layout(height=480)
        show(fig, c1)
        st.caption(f"Droite des moindres carrés : pente {f_rv['slope']:.2f} pt de rendement par point de volatilité, "
                   f"R² {f_rv['r2']:.2f}.")
    with c2:
        fig = px.scatter(df, x="tr_prev_year", y="ret_ytd", color="sector", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                         labels={"tr_prev_year": lab("tr_prev_year"), "ret_ytd": lab("ret_ytd")},
                         title="Persistance : année N-1 vs YTD")
        lo_ = float(min(df.tr_prev_year.min(), df.ret_ytd.min()))
        hi_ = float(max(df.tr_prev_year.max(), df.ret_ytd.max()))
        fig.add_trace(go.Scatter(x=[lo_, hi_], y=[lo_, hi_], mode="lines", name="y = x (même performance)",
                                 line=dict(color=C_GREY, width=1.5)))
        f_pp = ols_line(fig, df["tr_prev_year"], df["ret_ytd"])
        fig.update_layout(height=550)
        show(fig, c2)
        st.caption(f"Pente de la régression {f_pp['slope']:.2f} (1 = persistance parfaite), Spearman "
                   f"{f_pp['spearman']:.2f}.")

    disp = df.groupby(grp)[["ret_1m", "ret_ytd", "tr_prev_year"]].std()
    disp = disp[df.groupby(grp).size() >= 2].rename(columns=du.LABELS)
    fig = px.imshow(disp, text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
                    title="Dispersion des rendements au sein de chaque groupe (écart-type entre titres, en points)")
    fig.update_layout(height=max(280, 28 * len(disp) + 120))
    show(fig)
    _sv = df.groupby("sector").agg(tr=("tr_prev_year", "median"), v=("vol", "median"))
    _corr_rv = df["vol"].corr(df["tr_prev_year"], method="spearman")
    recap(["Total return N-1 médian : " + " ; ".join(f"{i} {r.tr:.1f} %" for i, r in _sv.iterrows()) + ".",
           "Volatilité médiane : " + " ; ".join(f"{i} {r.v:.0f} %" for i, r in _sv.iterrows()) + ".",
           f"Corrélation de rang volatilité / rendement N-1 : {_corr_rv:.2f}"
           + (" : le risque a été rémunéré sur l'année écoulée." if _corr_rv > 0.2 else "."),
           f"Persistance N-1 → YTD : pente {f_pp['slope']:.2f}, Spearman {f_pp['spearman']:.2f}."],
          ["Volatilité annualisée estimée sur 3 log-rendements disjoints, rapprochée à 50 % de la médiane de la "
           "sous-industrie (voir ci-dessus)."])

# =========================================================================== #
# 4. Valorisation & attentes
# =========================================================================== #
with tabs[3]:
    grp = st.radio("Comparer par", ["sector", "subindustry", "country"], horizontal=True,
                   format_func=GROUP_FR.get, key="val_grp")
    var = st.selectbox("Variable", ["peg", "upside", "pe_next", "fcf_yield", "earnings_yield", "ltg_eps"],
                       format_func=lab, key="val_var")
    gstat = df.groupby(grp)[var].agg(["median", "mean", "count"]).reset_index().sort_values("median")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(gstat, x="median", y=grp, orientation="h", text="count", template=TEMPLATE,
                     color_discrete_sequence=[C_BLUE], labels={"median": f"Médiane – {lab(var)}", grp: ""},
                     title=f"{lab(var)} médian (étiquette = nombre de titres)")
        fig.add_vline(x=df[var].median(), line_dash="dash", line_color="#666", annotation_text="médiane univers")
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
    with c2:
        fig = px.scatter(df, x="peg", y="upside", color="sector", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                         labels={"peg": lab("peg"), "upside": lab("upside")}, title="PEG vs upside des analystes")
        ols_line(fig, df["peg"], df["upside"])
        show(fig, c2)

    fig = px.scatter(df, x="earnings_yield", y="fcf_yield", color="sector", hover_name="ticker",
                     color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                     labels={"earnings_yield": lab("earnings_yield"), "fcf_yield": lab("fcf_yield")},
                     title="Earnings yield vs FCF yield : conversion des bénéfices en cash")
    lim = [min(df.earnings_yield.min(), df.fcf_yield.min()), max(df.earnings_yield.max(), df.fcf_yield.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="FCF yield = earnings yield",
                             line=dict(color="#666", dash="dash", width=1)))
    show(fig)
    _pg = df.groupby("sector")["peg"].median()
    recap(["PEG médian : " + " ; ".join(f"{i} {v:.2f}" for i, v in _pg.items()) + f" (univers {df.peg.median():.2f}).",
           f"Upside médian {df.upside.median():.1f} % ; {(df.upside > 0).mean():.0%} des titres ont un objectif de cours "
           "supérieur au cours : biais d'optimisme des analystes.",
           f"Corrélation de rang PEG / upside : {df.peg.corr(df.upside, method='spearman'):.2f} ; croissance LT / upside : "
           f"{df.ltg_eps.corr(df.upside, method='spearman'):.2f}."],
          ["Earnings yield N+1 = 100 ÷ P/E N+1, comparable au FCF yield."])

# =========================================================================== #
# 5. Relations entre facteurs
# =========================================================================== #
with tabs[4]:
    c1, c2 = st.columns([1, 3])
    with c1:
        method = st.radio("Corrélation", ["spearman", "pearson"],
                          format_func={"spearman": "Spearman (rangs)", "pearson": "Pearson (linéaire)"}.get)
        default_corr = du.NUMERIC + ["vol"] + ESG_COLS
        corr_cols = st.multiselect("Variables", ALL_NUM, default=default_corr, format_func=lab, key="corr_cols")
    with c2:
        if len(corr_cols) >= 2:
            cm = df[corr_cols].corr(method=method).rename(index=du.LABELS, columns=du.LABELS)
            fig = px.imshow(cm, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                            color_continuous_scale="RdBu_r", title=f"Matrice de corrélation ({method})")
            fig.update_layout(height=680)
            show(fig, c2)
    pairs = pd.DataFrame()
    if len(corr_cols) >= 2:
        cm_raw = df[corr_cols].corr(method=method)
        pairs = (cm_raw.where(np.triu(np.ones(cm_raw.shape, dtype=bool), 1)).stack().rename("corr").reset_index())
        pairs["|corr|"] = pairs["corr"].abs()
        pairs = pairs.sort_values("|corr|", ascending=False).head(8)
        pairs["Paire"] = pairs["level_0"].map(lab) + "  ×  " + pairs["level_1"].map(lab)

    st.subheader("Utilisation pour le critère agrégé : pondération sans redondance")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Les relations les plus fortes**")
        if len(pairs):
            st.dataframe(pairs[["Paire", "corr"]].style.format({"corr": "{:.2f}"}), hide_index=True, width="stretch")
        wt = pd.DataFrame([{"Pilier": p_, "Facteur": lab(f), "Sens": "↑" if s_ > 0 else "↓ (inversé)",
                            "Poids dans le score": f"{1 / 4 / len(fs):.1%}"}
                           for p_, fs in sc.PILLARS.items() for f, s_ in fs.items()])
        st.markdown("**Pondération retenue**")
        st.dataframe(wt, hide_index=True, width="stretch")
    with c2:
        pc = SCORES.loc[df.index, PILLARS].corr(method="spearman")
        fig = px.imshow(pc, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto",
                        title="Corrélation entre les 4 piliers (après z-scores)")
        fig.update_layout(height=380)
        show(fig, c2)
        ev = np.linalg.eigvalsh(pc.to_numpy())
        n_eff = ev.sum() ** 2 / (ev ** 2).sum()
        st.metric("Nombre effectif de piliers indépendants", f"{n_eff:.2f} / 4",
                  help="(Σλ)² / Σλ², λ = valeurs propres de la matrice de corrélation des piliers. 4 = aucune redondance.")
    off = pc.where(~np.eye(4, dtype=bool)).abs().stack()
    st.markdown(f"""
- Les facteurs qui portent la même information sont **regroupés dans un même pilier**, puis moyennés : un pilier pèse
  25 % quel que soit son nombre de facteurs, ce qui évite de surpondérer une information répétée.
- Une fois les z-scores calculés, la corrélation absolue moyenne entre piliers est de **{off.mean():.2f}**
  (maximum {off.max():.2f}, entre {off.idxmax()[0]} et {off.idxmax()[1]}) ; le nombre effectif de piliers
  indépendants est de **{n_eff:.1f} sur 4**. La redondance résiduelle est faible : la pondération égale des piliers est conservée.
- Les variables de momentum (RSI, perf. 1 mois, YTD, N-1) ne sont pas intégrées au score : elles servent à estimer la
  volatilité, et leur forte corrélation entre elles (RSI × perf. 1 mois) en ferait un signal redondant de court terme.
""")

    st.subheader("Analyse d'une relation")
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
        f_ = ols_line(fig, dd[xv], dd[yv], f"MCO {name}", SECTOR_COLORS.get(name, "#333"))
        stats.append({"Groupe": name, "n": f_["n"], "Pente": f_["slope"], "R²": f_["r2"],
                      "Pearson": f_["pearson"], "Spearman": f_["spearman"]})
    fig.update_layout(height=500)
    show(fig)
    st.dataframe(pd.DataFrame(stats).style.format({"Pente": "{:.3f}", "R²": "{:.2f}", "Pearson": "{:.2f}",
                                                  "Spearman": "{:.2f}"}), hide_index=True, width="stretch")
    if len(pairs):
        _p0 = pairs.iloc[0]
        recap([f"Relation la plus forte : {_p0['Paire']} ({_p0['corr']:.2f}).",
               f"Après regroupement en piliers : {n_eff:.1f} piliers effectivement indépendants sur 4."],
              ["Nombre effectif de piliers = (Σλ)² / Σλ², λ valeurs propres de la matrice de corrélation des piliers."])

# =========================================================================== #
# 6. Rule of 40 & critère agrégé
# =========================================================================== #
with tabs[5]:
    r40 = df_raw_f.loc[df.index, "rule_of_40"]
    passe = r40 >= 40
    passe_rate = float(passe.mean())

    st.subheader("Rule of 40")
    k = st.columns(4)
    k[0].metric("Titres passant la règle", f"{int(passe.sum())} / {len(df)}")
    k[1].metric("Rule of 40 médiane", f"{r40.median():.1f}")
    for i_, (s_, v_) in enumerate(list(passe.groupby(df.sector).mean().items())[:2]):
        k[2 + i_].metric(f"Taux de passage – {s_}", f"{v_:.0%}")
    fig = px.scatter(df.assign(statut=np.where(passe, "Passe", "Ne passe pas")), x="sales_growth", y="profit_margin",
                     color="sector", symbol="statut", hover_name="ticker", color_discrete_map=SECTOR_COLORS,
                     template=TEMPLATE, labels={"sales_growth": lab("sales_growth"), "profit_margin": lab("profit_margin")},
                     title="Croissance du CA vs marge nette : la Rule of 40 est la droite x + y = 40")
    xs = np.linspace(df.sales_growth.min(), df.sales_growth.max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=40 - xs, mode="lines", name="x + y = 40", line=dict(color="#666", dash="dash")))
    fig.update_yaxes(range=[df.profit_margin.min() - 5, df.profit_margin.max() + 5])
    fig.update_layout(height=460)
    show(fig)

    st.subheader("Critère agrégé")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.latex(r"z_{i,f} = s_f\,\frac{x_{i,f} - \bar x_{g(i),f}}{\sigma_{g(i),f}}")
        st.latex(r"\text{Score}_i = \tfrac14\sum_{p=1}^{4}\tfrac{1}{|p|}\sum_{f\in p} z_{i,f}")
        st.markdown("""
- z-score de chaque facteur **relatif à la sous-industrie** g(i) ; s<sub>f</sub> = −1 pour le PEG.
- Moyenne et variance d'une petite sous-industrie rapprochées de celles du secteur :
  (n·groupe + 5·secteur) ÷ (n + 5) ; z-scores bornés à ±3.
- 4 piliers à poids égal : **Croissance** (croissance LT du BPA, croissance du CA), **Rentabilité** (marge, FCF yield),
  **Valorisation** (PEG, earnings yield), **Attentes** (upside, prime de risque).
""", unsafe_allow_html=True)
    with c2:
        rank = df[["ticker", "name", "sector", "subindustry"] + PILLARS + ["Score final", "rule_of_40"]] \
            .sort_values("Score final", ascending=False)
        top = rank.head(15).iloc[::-1]
        fig = go.Figure()
        for p_, col_ in zip(PILLARS, [C_NAVY, C_BLUE, C_LIGHT, C_TEAL]):
            fig.add_trace(go.Bar(y=top.ticker, x=top[p_] / 4, name=p_, orientation="h", marker_color=col_))
        fig.add_trace(go.Scatter(y=top.ticker, x=top["Score final"], mode="markers", name="Score final",
                                 marker=dict(color="#0b0b0b", size=9, symbol="diamond")))
        fig.update_layout(barmode="relative", template=TEMPLATE, height=470,
                          title="Top 15 du score final et contribution de chaque pilier",
                          legend=dict(orientation="h", y=-0.12))
        show(fig, c2)

    st.subheader("Comparaison du critère agrégé avec la Rule of 40")
    cmp_ = sc.rule40_comparison(df["Score final"], r40, passe_rate)
    both = int((cmp_["top"] & cmp_["passe"]).sum())
    k = st.columns(4)
    k[0].metric("Corrélation de rang (Spearman)", f"{cmp_['spearman']:.2f}")
    k[1].metric("Tau de Kendall", f"{cmp_['kendall']:.2f}")
    k[2].metric("κ de Cohen", f"{cmp_['kappa']:.2f}",
                help=f"Accord observé {cmp_['accord']:.0%}, accord attendu par hasard {cmp_['accord_hasard']:.0%} ; "
                     "κ = (observé − hasard) / (1 − hasard).")
    k[3].metric("Titres retenus par les deux", f"{both} / {int(passe.sum())}",
                help=f"Comparaison à taille égale : top {passe_rate:.0%} du score vs titres passant la règle.")
    c1, c2 = st.columns(2)
    with c1:
        d_ = df.assign(r40=r40.clip(-60, 120), statut=np.where(passe, "Passe la Rule of 40", "Ne passe pas"))
        fig = px.scatter(d_, x="r40", y="Score final", color="sector", symbol="statut", hover_name="ticker",
                         hover_data={"name": True, "subindustry": True}, color_discrete_map=SECTOR_COLORS,
                         template=TEMPLATE, labels={"r40": "Rule of 40 (axe borné à [−60 ; 120])"},
                         title="Score final vs Rule of 40")
        fig.add_vline(x=40, line_dash="dash", line_color="#666")
        fig.add_hline(y=float(df["Score final"].quantile(1 - passe_rate)), line_dash="dot", line_color="#666")
        fig.update_layout(height=500, legend=dict(orientation="h", y=-0.2))
        show(fig, c1)
    with c2:
        sub_cmp = pd.DataFrame({"Rule of 40": cmp_["passe"].groupby(df.subindustry).mean() * 100,
                                "Score (top)": cmp_["top"].groupby(df.subindustry).mean() * 100})
        sub_cmp = sub_cmp.loc[df.subindustry.value_counts().index]
        fig = px.bar(sub_cmp.reset_index().melt(id_vars="subindustry", var_name="Critère", value_name="pct"),
                     x="pct", y="subindustry", color="Critère", barmode="group", orientation="h", template=TEMPLATE,
                     color_discrete_sequence=[C_LIGHT, C_NAVY],
                     labels={"pct": "% des titres de la sous-industrie retenus", "subindustry": ""},
                     title="Titres retenus par sous-industrie")
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.2))
        show(fig, c2)

    corr_p = pd.Series({p_: df[p_].corr(r40, method="spearman") for p_ in PILLARS})
    prof_vars = ["peg", "upside", "mu", "vol", "sharpe_ante", "tr_prev_year"] + ESG_COLS
    groups_ = {"Passent la Rule of 40": cmp_["passe"], "Top du score": cmp_["top"],
               "Univers": pd.Series(True, index=df.index)}
    prof = pd.DataFrame({g_: df.loc[m_, prof_vars].median() for g_, m_ in groups_.items()})
    prof.loc["Sous-industries représentées"] = [df.loc[m_, "subindustry"].nunique() for m_ in groups_.values()]
    prof.index = [lab(i) + (" – médiane" if i in prof_vars else "") for i in prof.index]
    st.markdown("**Profil des titres retenus par chaque critère**")
    st.dataframe(prof.style.format("{:.2f}"), width="stretch")
    st.markdown("**Corrélation de rang de chaque pilier avec la Rule of 40** : " +
                " · ".join(f"{p_} {v_:.2f}" for p_, v_ in corr_p.items()))
    dis = rank.assign(Statut=np.select([cmp_["top"] & ~cmp_["passe"], ~cmp_["top"] & cmp_["passe"]],
                                       ["Score élevé, Rule of 40 non atteinte", "Rule of 40 atteinte, score faible"], ""))
    dis = dis[dis.Statut != ""][["Statut", "ticker", "name", "subindustry", "Score final", "rule_of_40"] + PILLARS]
    with st.expander(f"Titres sur lesquels les deux critères divergent ({len(dis)})"):
        st.dataframe(dis.style.format({c_: "{:.2f}" for c_ in ["Score final"] + PILLARS} | {"rule_of_40": "{:.1f}"}),
                     hide_index=True, width="stretch")
    pr_, tp_ = df.loc[cmp_["passe"]], df.loc[cmp_["top"]]
    best_p, worst_p = corr_p.idxmax(), corr_p.idxmin()
    recap([f"Classements liés mais distincts : Spearman {cmp_['spearman']:.2f} (≈ {cmp_['spearman'] ** 2:.0%} de variance "
           f"de rang commune), κ de Cohen {cmp_['kappa']:.2f} (accord modéré) ; {both} titres sur {int(passe.sum())} "
           "retenus par les deux critères.",
           f"Pilier le plus proche de la Rule of 40 : {best_p} (ρ {corr_p[best_p]:.2f}) ; le plus éloigné : {worst_p} "
           f"(ρ {corr_p[worst_p]:.2f}).",
           f"La Rule of 40 retient des titres plus chers (PEG médian {pr_.peg.median():.2f} contre {tp_.peg.median():.2f}) "
           f"et plus concentrés ({pr_.subindustry.nunique()} sous-industries contre {tp_.subindustry.nunique()}).",
           "Conclusion : la Rule of 40 est un filtre de qualité opérationnelle, absolu et favorable aux éditeurs de "
           "logiciels ; le score intègre le prix payé et les attentes, relativement aux pairs. Le portefeuille repose "
           "sur le score ; la Rule of 40 sert d'indicateur de contrôle."],
          ["Comparaison à taille égale : top du score de même effectif que les titres passant la règle.",
           "κ de Cohen = (accord observé − accord attendu par hasard) ÷ (1 − accord attendu par hasard)."])

# =========================================================================== #
# 7. ESG
# =========================================================================== #
with tabs[6]:
    if not HAS_ESG:
        st.warning("Fichier `data/esg_scores.csv` absent.")
    else:
        n_old = int(df_full["esg_provider"].fillna("").str.contains("2024").sum())
        n_missing = int(df_full["esg"].isna().sum())
        st.caption(f"S&P Global ESG Score (0–100, plus haut = meilleur, relatif à l'industrie). "
                   f"{int(df_full['esg'].notna().sum()) - n_old} titres : page S&P Global de la société (2025–2026) ; "
                   f"{n_old} titres : score CSA S&P Global du 22/11/2024 ; {n_missing} titres sans score public.")
        de = df.dropna(subset=["esg"])
        k = st.columns(4)
        k[0].metric("Titres notés", f"{len(de)} / {len(df)}")
        k[1].metric("Score moyen", f"{de.esg.mean():.1f}")
        k[2].metric("Score médian", f"{de.esg.median():.1f}")
        k[3].metric("Au-dessus de la médiane de leur sous-industrie", f"{int((df.esg_above == 1).sum())}")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(de, x="sector", y="esg", color="sector", points="all", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE, labels={"esg": lab("esg")},
                         title="Score ESG par secteur")
            fig.update_layout(showlegend=False, height=520)
            show(fig, c1)
        with c2:
            ab = df.assign(statut=np.select([df.esg_above == 1, df.esg_above == 0],
                                            ["Au-dessus de la médiane", "Égal ou en dessous"], "Non noté"))
            ct_ = ab.groupby(["subindustry", "statut"]).size().reset_index(name="n")
            fig = px.bar(ct_, x="n", y="subindustry", color="statut", orientation="h", template=TEMPLATE,
                         color_discrete_map={"Au-dessus de la médiane": C_TEAL, "Égal ou en dessous": C_LIGHT,
                                             "Non noté": "#D5DCE6"},
                         labels={"n": "Nombre de titres", "subindustry": "", "statut": "Score ESG"},
                         title="Position vs médiane ESG de la sous-industrie")
            fig.update_layout(height=520, legend=dict(orientation="h", y=-0.15))
            show(fig, c2)
        yv = st.selectbox("Comparer l'ESG à", ["tr_prev_year", "ret_ytd", "vol", "sharpe_ante", "upside", "Score final"],
                          format_func=lab, key="esg_y")
        fig = px.scatter(de, x="esg", y=yv, color="sector", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE, labels={"esg": lab("esg"), yv: lab(yv)})
        f_ = ols_line(fig, de["esg"], de[yv])
        fig.update_layout(title=f"ESG vs {lab(yv)}  ·  pente {f_['slope']:.2f}  ·  Spearman {f_['spearman']:.2f}  ·  "
                                f"R² {f_['r2']:.2f}", height=480)
        show(fig)
        st.caption("Droite des moindres carrés. Pente : variation moyenne de la variable pour +1 point d'ESG. "
                   "Spearman : corrélation des rangs (−1 à 1). R² : part de la variance expliquée par l'ESG.")
        de = de.assign(quintile=pd.qcut(de.esg.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]))
        qt = de.groupby("quintile", observed=True)[["esg", "tr_prev_year", "ret_ytd", "vol", "sharpe_ante", "upside"]].mean()
        st.markdown("**Rendement et risque moyens par quintile ESG** (Q1 = scores les plus bas)")
        st.dataframe(qt.rename(columns=du.LABELS).style.format("{:.2f}"), width="stretch")
        with st.expander("Sources des scores ESG"):
            st.dataframe(df_full.loc[df_full.esg.notna(), ["ticker", "name", "esg", "esg_provider", "esg_date", "source_url"]],
                         hide_index=True, width="stretch",
                         column_config={"source_url": st.column_config.LinkColumn("Source")})
        _sp = du.ols_fit(de["esg"], de["tr_prev_year"])["spearman"]
        _em = de.groupby("sector").esg.median()
        recap([f"{len(de)} titres notés sur {len(df)}. Score médian : "
               + " ; ".join(f"{i} {v:.0f}" for i, v in _em.items()) + ".",
               f"Corrélation de rang ESG / total return N-1 : {_sp:.2f}"
               + (" : pas de lien net entre ESG et performance sur la période." if abs(_sp) < 0.2
                  else (" : les meilleurs scores ESG ont plutôt surperformé." if _sp > 0
                        else " : les meilleurs scores ESG ont plutôt sous-performé.")),
               "Le filtre ESG du portefeuille compare chaque titre à la médiane ESG de sa sous-industrie (titres notés)."])

# =========================================================================== #
# 8. Rendement attendu & Sharpe
# =========================================================================== #
with tabs[7]:
    st.latex(r"\mu_i = \underbrace{(1-\lambda)\,\text{Prime}_i + \lambda\,(\text{Upside}_i - r_f)}_{\text{niveau}} + "
             r"\underbrace{IC\cdot\sigma_i\cdot \tilde Z^{\alpha}_i}_{\text{alpha z-scores}}, \qquad "
             r"\text{Sharpe}^{ex\,ante}_i = \frac{\mu_i}{\sigma_i}")
    st.markdown(f"""
- **Niveau** : prime de risque implicite, rendement en excès du sans-risque déjà intégré au fichier (λ = {lam:.2f}).
  L'upside brut n'est pas utilisé en niveau : {(df_full.upside > 0).mean():.0%} des objectifs de cours sont au-dessus du
  cours (médiane {df_full.upside.median():.1f} %), et une moyenne 50/50 prime / upside donnerait un rendement attendu
  médian de {MU_5050.median():.1f} %.
- **Alpha** : Z̃<sup>α</sup> = moyenne des piliers Croissance, Rentabilité, Valorisation et du z-score de l'upside,
  re-standardisée (écart-type 1) ; IC = {ic:.2f}. L'upside apporte ainsi son information relative aux pairs, sans son biais de niveau.
""", unsafe_allow_html=True)
    k = st.columns(4)
    k[0].metric("Rendement attendu médian", f"{df.mu.median():.1f} %")
    k[1].metric("Écart-type entre titres : prime", f"{df['Ancrage prime de risque'].std():.1f} pt")
    k[2].metric("Écart-type entre titres : alpha z-scores", f"{df['Alpha z-scores'].std():.1f} pt")
    k[3].metric("Corrélation de rang μ / score final", f"{df.mu.corr(df['Score final'], method='spearman'):.2f}")
    st.caption("Le niveau moyen vient de la prime ; la sélection et la pondération dépendent des écarts entre titres, "
               "auxquels l'alpha z-scores contribue dans une proportion comparable à la prime.")
    dec = df.groupby("subindustry")[MU_COMP + ["mu"]].mean().sort_values("mu")
    fig = go.Figure()
    for p_, col_ in zip(MU_COMP, [C_NAVY, C_LIGHT, C_TEAL]):
        if dec[p_].abs().sum() > 0:
            fig.add_trace(go.Bar(y=dec.index, x=dec[p_], name=p_, orientation="h", marker_color=col_))
    fig.add_trace(go.Scatter(y=dec.index, x=dec["mu"], mode="markers", name="Rendement attendu",
                             marker=dict(color="#0b0b0b", size=9, symbol="diamond")))
    fig.update_layout(barmode="relative", template=TEMPLATE, height=520, xaxis_title="Contribution moyenne (%)",
                      title="Décomposition du rendement attendu par sous-industrie (moyenne)")
    show(fig)

    st.subheader("Pertinence du score : régressions sur les piliers fondamentaux")
    targets = {"risk_prem": lab("risk_prem"), "upside": lab("upside"), "tr_prev_year": lab("tr_prev_year"),
               "ret_ytd": lab("ret_ytd")}
    regs = []
    for y_, nm in targets.items():
        r_ = du.ols_multi(base[y_], base[["Croissance", "Rentabilité", "Valorisation"]])
        row = {"Variable expliquée": nm, "R²": r_["r2"]}
        for p_ in ["Croissance", "Rentabilité", "Valorisation"]:
            row[f"{p_} (coef.)"] = r_["coef"][p_]
            row[f"{p_} (t)"] = r_["t"][p_]
        regs.append(row)
    regs = pd.DataFrame(regs)
    st.dataframe(regs.style.format(precision=2), hide_index=True, width="stretch")
    st.caption("Régression linéaire en coupe sur les 99 titres. |t| > 2 : coefficient significatif au seuil de 5 %. "
               "Les données étant toutes à la même date, il s'agit d'un test de cohérence et non d'un backtest.")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="sector", y="sharpe_ante", color="sector", points="all", hover_name="ticker",
                     color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                     labels={"sharpe_ante": lab("sharpe_ante")}, title="Sharpe ex ante par secteur")
        fig.update_layout(showlegend=False, height=520)
        show(fig, c1)
    with c2:
        top = pd.concat([df.nlargest(12, "sharpe_ante"), df.nsmallest(12, "sharpe_ante")])
        fig = px.bar(top.sort_values("sharpe_ante"), x="sharpe_ante", y="ticker", color="sector", orientation="h",
                     color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                     labels={"sharpe_ante": lab("sharpe_ante"), "ticker": ""},
                     title="12 meilleurs et 12 moins bons Sharpe ex ante")
        fig.update_layout(height=520)
        show(fig, c2)
    _sh = df.groupby("sector")["sharpe_ante"].median()
    _r = regs.set_index("Variable expliquée")
    recap([f"Rendement attendu médian : {df.mu.median():.1f} % ; alpha z-scores de {df['Alpha z-scores'].min():+.1f} à "
           f"{df['Alpha z-scores'].max():+.1f} pt.",
           "Sharpe ex ante médian : " + " ; ".join(f"{i} {v:.2f}" for i, v in _sh.items()) + ".",
           f"Les piliers fondamentaux expliquent {_r.loc[lab('upside'), 'R²']:.0%} de la variance de l'upside et "
           f"{_r.loc[lab('tr_prev_year'), 'R²']:.0%} de celle du total return N-1 : le score capte une information "
           "reconnue par le marché."],
          ["Alpha = IC × σ × score standardisé (règle de Grinold).",
           f"Sharpe ex post = (total return N-1 − {rf:.2f} %) ÷ volatilité estimée."])

# =========================================================================== #
# 9. Frontière efficiente
# =========================================================================== #
with tabs[8]:
    st.markdown(f"Portefeuilles long-only de variance minimale pour chaque niveau de rendement attendu, "
                f"**{CAP:.0%} maximum par titre**, sur la sélection filtrée.")
    R = an.block_correlation(df, *RHO)
    cov = an.covariance(df["vol"], R)
    mu = df["mu"].to_numpy(float)
    if len(df) * CAP < 1:
        st.warning(f"Avec {len(df)} titres et {CAP:.0%} maximum par titre, le portefeuille ne peut pas être investi à 100 %.")
    else:
        front, w_mv, w_ms = frontier_cached(tuple(mu), tuple(cov.ravel()), len(df), CAP)
        ew = np.full(len(df), 1 / len(df))
        fig = go.Figure()
        for s_, c_ in SECTOR_COLORS.items():
            d_ = df[df.sector == s_]
            fig.add_trace(go.Scatter(x=d_["vol"], y=d_["mu"], mode="markers", name=s_, text=d_["ticker"],
                                     marker=dict(color=c_, size=7, opacity=0.6),
                                     hovertemplate="%{text}<br>vol %{x:.1f} %<br>rdt %{y:.1f} %<extra></extra>"))
        fig.add_trace(go.Scatter(x=front["vol"], y=front["ret"], mode="lines", name=f"Frontière (≤ {CAP:.0%} par titre)",
                                 line=dict(color="#0b0b0b", width=2.5)))
        pts = [("Équipondéré", ew, "diamond", C_GREY)]
        if w_mv is not None:
            pts.append(("Variance minimale", w_mv, "square", C_TEAL))
        if w_ms is not None:
            pts.append(("Sharpe maximal", w_ms, "star", C_RED))
        for name, w, sym, col in pts:
            r_, v_ = an.port_stats(w, mu, cov)
            fig.add_trace(go.Scatter(x=[v_], y=[r_], mode="markers", name=name,
                                     marker=dict(symbol=sym, size=15, color=col, line=dict(color="white", width=1.5))))
        fig.update_layout(template=TEMPLATE, height=600, xaxis_title=lab("vol"), yaxis_title=lab("mu"),
                          title="Titres, portefeuille équipondéré et frontière efficiente",
                          xaxis_range=[0, min(df["vol"].max() * 1.05, 120)])
        show(fig)
        rows = []
        for name, w in [("Équipondéré", ew), ("Variance minimale", w_mv), ("Sharpe maximal", w_ms)]:
            if w is None:
                continue
            r_, v_ = an.port_stats(w, mu, cov)
            rows.append({"Portefeuille": name, "Rendement attendu (%)": r_, "Volatilité (%)": v_, "Sharpe": r_ / v_,
                         "Lignes": int((w > 1e-4).sum()), "Nb effectif 1/Σw²": 1 / np.sum(w ** 2)})
        st.dataframe(pd.DataFrame(rows).style.format(precision=2), hide_index=True, width="stretch")

    st.subheader("Corrélations : hypothèses et sensibilité")
    st.markdown(f"""
Le fichier ne contient qu'un rendement par titre et par période (3 périodes disjointes) : une matrice de corrélation
historique ne peut pas être estimée. Un test sur ces 3 sous-périodes donne une corrélation moyenne de 0,22 au sein
d'une sous-industrie, 0,19 au sein d'un secteur et −0,19 entre secteurs : la valeur négative traduit le bruit d'un
échantillon de 3 points, pas une réalité économique. Le modèle retient donc une **structure par blocs** : corrélation
**{RHO[0]:.2f}** entre titres de la même sous-industrie, **{RHO[1]:.2f}** au sein d'un secteur, **{RHO[2]:.2f}**
entre secteurs. Le tableau ci-dessous mesure la sensibilité des conclusions à ces hypothèses.
""")
    sens = []
    for nm, rh in an.RHO_SETS.items():
        cov_s = an.covariance(full["vol"], an.block_correlation(full, *rh))
        P_s = build_portfolio(full, tuple(elig_all), n_total, CAP, rh)
        w_s = P_s["Poids"].reindex(full.index).fillna(0).to_numpy()
        r_e, v_e = an.port_stats(EW.to_numpy(), full["mu"].to_numpy(float), cov_s)
        r_p, v_p = an.port_stats(w_s, full["mu"].to_numpy(float), cov_s)
        sens.append({"Hypothèse (sous-ind. / secteur / autres)": f"{nm} ({rh[0]:.2f} / {rh[1]:.2f} / {rh[2]:.2f})",
                     "Équipondéré : volatilité (%)": v_e, "Équipondéré : Sharpe": r_e / v_e,
                     "Portefeuille sans ESG : volatilité (%)": v_p, "Portefeuille sans ESG : Sharpe": r_p / v_p,
                     "Écart de Sharpe": r_p / v_p - r_e / v_e})
    st.dataframe(pd.DataFrame(sens).style.format(precision=2), hide_index=True, width="stretch")
    _sens = pd.DataFrame(sens)
    recap([f"Avec l'hypothèse centrale, le portefeuille sans ESG améliore le Sharpe ex ante de "
           f"{_sens.loc[1, 'Écart de Sharpe']:+.2f} par rapport à l'équipondéré ; l'écart reste positif sous les "
           f"hypothèses basse ({_sens.loc[0, 'Écart de Sharpe']:+.2f}) et haute ({_sens.loc[2, 'Écart de Sharpe']:+.2f})."
           if (_sens["Écart de Sharpe"] > 0).all() else
           "Le classement des portefeuilles dépend des hypothèses de corrélation : voir le tableau de sensibilité."],
          ["Covariance Σ = D·R·D : D = volatilités estimées, R = corrélations par blocs.",
           "Frontière : minimisation de wᵀΣw pour chaque rendement cible, 0 ≤ w ≤ 5 %, Σw = 1 (SLSQP)."])

# =========================================================================== #
# 10. Portefeuille final
# =========================================================================== #
with tabs[9]:
    st.latex(r"\max_{w}\ \frac{w^\top \mu}{\sqrt{w^\top \Sigma w}} \quad \text{s.c.} \quad \sum_i w_i = 1,"
             r"\quad 0 \le w_i \le 5\,\%")
    esg_on = st.toggle("Filtre ESG : score ESG supérieur à la médiane de la sous-industrie", value=False,
                       disabled=not HAS_ESG)
    cur_name = "Avec filtre ESG" if esg_on else "Sans ESG"
    cur_label = cur_name[0].lower() + cur_name[1:]
    P = ports[cur_name]
    rows = [pf.summary("Équipondéré univers (99 titres)", EW, full, COV_FULL, rf)]
    for nm, sel in ports.items():
        rows.append(pf.summary(f"Portefeuille {nm[0].lower() + nm[1:]}", sel["Poids"], full, COV_FULL, rf))
    comp = pd.DataFrame(rows).set_index("Portefeuille").T
    cur = comp[f"Portefeuille {cur_label}"]
    ewc = comp.iloc[:, 0]

    if esg_on:
        n_res = int((P["Raison"] == "Repêché (diversification sectorielle)").sum())
        st.caption(f"{int(elig_esg.sum())} titres éligibles sur {len(full)} (score ESG > médiane de la sous-industrie ; "
                   f"titres non notés exclus). {n_res} titre(s) repêché(s) pour la diversification : score final > médiane "
                   "de la sous-industrie et score ESG ≥ médiane ESG de l'univers, priorité aux sous-industries non "
                   "représentées.")
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
    Pw = P[P["Poids"] > 0].sort_values("Poids")
    with c1:
        fig = px.bar(Pw, x=Pw["Poids"] * 100, y="ticker", color="sector", orientation="h",
                     color_discrete_map=SECTOR_COLORS, template=TEMPLATE, hover_data={"name": True, "subindustry": True},
                     labels={"x": "Poids (%)", "ticker": ""}, title="Poids des titres")
        fig.add_vline(x=CAP * 100, line_dash="dash", line_color=C_RED, annotation_text="5 %")
        fig.update_layout(height=max(500, 20 * len(Pw) + 120), legend=dict(orientation="h", y=-0.08))
        show(fig, c1)
    with c2:
        sw = pd.DataFrame({"Portefeuille": P.groupby("subindustry")["Poids"].sum() * 100,
                           "Équipondéré univers": full.subindustry.value_counts(normalize=True) * 100}).fillna(0)
        sw = sw.sort_values("Équipondéré univers")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=sw.index, x=sw["Équipondéré univers"], name="Équipondéré univers", orientation="h",
                             marker_color=C_PALE))
        fig.add_trace(go.Bar(y=sw.index, x=sw["Portefeuille"], name="Portefeuille", orientation="h", marker_color=C_NAVY))
        fig.update_layout(barmode="group", template=TEMPLATE, height=max(500, 20 * len(Pw) + 120),
                          xaxis_title="Poids (%)", title="Poids par sous-industrie", legend=dict(orientation="h", y=-0.08))
        show(fig, c2)

    st.subheader("Comparaison avec le portefeuille équipondéré")
    st.dataframe(comp.style.format("{:.2f}"), width="stretch", height=740)
    st.caption("Performances N-1 et YTD : poids actuels appliqués aux rendements passés (description, pas un backtest).")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=full["vol"], y=full["mu"], mode="markers", name="Univers", text=full["ticker"],
                                 marker=dict(color=C_PALE, size=7),
                                 hovertemplate="%{text}<br>vol %{x:.1f} %<br>rdt %{y:.1f} %<extra></extra>"))
        fig.add_trace(go.Scatter(x=Pw["vol"], y=Pw["mu"], mode="markers", name="Titres retenus", text=Pw["ticker"],
                                 marker=dict(color=C_BLUE, size=6 + 300 * Pw["Poids"]),
                                 hovertemplate="%{text}<br>vol %{x:.1f} %<br>rdt %{y:.1f} %<extra></extra>"))
        for col_name, sym, colr in [(comp.columns[0], "diamond", C_GREY), (comp.columns[1], "star", C_RED)] + \
                ([(comp.columns[2], "star", C_TEAL)] if comp.shape[1] > 2 else []):
            fig.add_trace(go.Scatter(x=[comp.loc["Volatilité estimée (%)", col_name]],
                                     y=[comp.loc["Rendement attendu (%)", col_name]], mode="markers", name=col_name,
                                     marker=dict(symbol=sym, size=18, color=colr, line=dict(color="white", width=1.5))))
        fig.update_layout(template=TEMPLATE, height=500, xaxis_title=lab("vol"), yaxis_title=lab("mu"),
                          title="Positionnement rendement / risque", xaxis_range=[0, min(full["vol"].max() * 1.05, 120)],
                          legend=dict(orientation="h", y=-0.2))
        show(fig, c1)
    with c2:
        rc = Pw.sort_values("Contribution au risque", ascending=False).head(15)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rc.ticker, y=rc["Poids"] * 100, name="Poids (%)", marker_color=C_PALE))
        fig.add_trace(go.Bar(x=rc.ticker, y=rc["Contribution au risque"] * 100, name="Contribution au risque (%)",
                             marker_color=C_NAVY))
        fig.update_layout(barmode="group", template=TEMPLATE, height=500, legend=dict(orientation="h", y=-0.2),
                          title="Poids et contribution au risque (15 premières contributions)")
        show(fig, c2)

    st.subheader("Composition")
    det_cols = ["ticker", "name", "sector", "subindustry", "country", "Poids", "Score final", "mu", "vol",
                "sharpe_ante", "rule_of_40", "Raison"] + (["esg", "esg_med_sub"] if HAS_ESG else [])
    det = P.sort_values("Poids", ascending=False)[det_cols].copy()
    det["Poids"] = det["Poids"] * 100
    det = det.rename(columns={"ticker": "Ticker", "name": "Société", "sector": "Secteur", "subindustry": "Sous-industrie",
                              "country": "Pays", "Poids": "Poids (%)", "mu": "Rendement attendu (%)",
                              "vol": "Volatilité (%)", "sharpe_ante": lab("sharpe_ante"),
                              "rule_of_40": "Rule of 40", "esg": "Score ESG", "esg_med_sub": "Médiane ESG sous-ind."})
    zero = det[det["Poids (%)"] <= 0]
    det = det[det["Poids (%)"] > 0]
    st.dataframe(det.style.format(precision=2), hide_index=True, width="stretch")
    if len(zero):
        st.caption(f"Sélectionnés mais pondérés à 0 % par l'optimiseur : {', '.join(zero['Ticker'])}.")
    st.download_button("Télécharger la composition (CSV)", det.to_csv(index=False).encode(),
                       f"portefeuille_{'esg' if esg_on else 'sans_esg'}.csv", "text/csv")

    if HAS_ESG:
        a_, b_ = set(ports["Sans ESG"].loc[ports["Sans ESG"].Poids > 0, "ticker"]), \
            set(ports["Avec filtre ESG"].loc[ports["Avec filtre ESG"].Poids > 0, "ticker"])
        st.subheader("Effet du filtre ESG")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Communs ({len(a_ & b_)})**  \n" + ", ".join(sorted(a_ & b_)))
        c2.markdown(f"**Uniquement sans ESG ({len(a_ - b_)})**  \n" + ", ".join(sorted(a_ - b_)))
        c3.markdown(f"**Uniquement avec ESG ({len(b_ - a_)})**  \n" + ", ".join(sorted(b_ - a_)))

    st.subheader("Fiche titre : profil d'une ligne du portefeuille")
    hold = Pw.sort_values("Poids", ascending=False)
    choice = st.selectbox("Titre", hold.ticker + " – " + hold.name + "  (" + (hold["Poids"] * 100).round(1).astype(str) + " %)")
    tk = choice.split(" – ")[0]
    row = full[full.ticker == tk].iloc[0]
    st.markdown(f"**{row['name']}** · {row['sector']} · {row['subindustry']} · {row['country']}")
    prof_cols = ["upside", "ltg_eps", "sales_growth", "profit_margin", "fcf_yield", "earnings_yield",
                 "sharpe_ante", "tr_prev_year", "Score final"] + ESG_COLS
    pct_all = full[prof_cols].rank(pct=True).loc[row.name] * 100
    peers = full[full.subindustry == row.subindustry]
    pct_sub = peers[prof_cols].rank(pct=True).loc[row.name] * 100
    theta = [lab(c) for c in prof_cols]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=pct_all.values, theta=theta, fill="toself", name="Percentile univers",
                                  line_color=C_BLUE))
    fig.add_trace(go.Scatterpolar(r=pct_sub.values, theta=theta, fill="toself", name="Percentile sous-industrie",
                                  line_color=C_TEAL, opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=500,
                      title="Profil en percentiles (100 = valeur la plus élevée)")
    c1, c2 = st.columns([3, 2])
    show(fig, c1)
    vcols = du.NUMERIC + ["rule_of_40", "vol", "mu", "sharpe_ante"] + PILLARS + ["Score final"] + ESG_COLS
    vals = pd.DataFrame({"Valeur": row[vcols].astype(float), "Médiane sous-industrie": peers[vcols].median(),
                         "Médiane univers": full[vcols].median()})
    vals.index = [lab(c) for c in vals.index]
    c2.dataframe(vals.style.format("{:.2f}"), width="stretch", height=500)

    best = comp.loc["Sharpe ex ante"].idxmax()
    recap([f"Portefeuille {cur_label} : {int(cur['Lignes'])} lignes, rendement attendu {cur['Rendement attendu (%)']:.1f} %, "
           f"volatilité {cur['Volatilité estimée (%)']:.1f} %, Sharpe ex ante {cur['Sharpe ex ante']:.2f} contre "
           f"{ewc['Sharpe ex ante']:.2f} pour l'équipondéré.",
           f"Score final moyen {cur['Score final moyen']:.2f} (équipondéré {ewc['Score final moyen']:.2f}) ; "
           f"{cur['Part passant la Rule of 40 (%)']:.0f} % des poids passent la Rule of 40 "
           f"(équipondéré {ewc['Part passant la Rule of 40 (%)']:.0f} %).",
           f"{int(cur['Sous-industries'])} sous-industries ; poids maximal d'une sous-industrie "
           f"{cur['Poids max. d\'une sous-industrie (%)']:.0f} % ; technologie {cur['Part technologie (%)']:.0f} %.",
           f"Meilleur Sharpe ex ante des trois portefeuilles : {best}."],
          [f"Quotas : {n_total} places réparties entre sous-industries au prorata de leur taille (plus forts restes) ; "
           "un titre n'est retenu que si son score dépasse la médiane de sa sous-industrie ; places non pourvues "
           "attribuées aux meilleurs scores restants.",
           "Pondération : maximisation du Sharpe ex ante, 0 ≤ w ≤ 5 %, Σw = 1 (SLSQP, plusieurs points de départ).",
           "Nombre effectif de titres = 1 / Σw² ; contribution au risque = w_i (Σw)_i / wᵀΣw."])

# =========================================================================== #
# Annexe : méthodes
# =========================================================================== #
with tabs[10]:
    st.subheader("Synthèse des méthodes")
    st.markdown(f"""
| Étape | Méthode |
| --- | --- |
| Valeurs extrêmes | Winsorisation : chaque variable est bornée à ses quantiles {p:.0%} et {1 - p:.0%} (interpolation linéaire) |
| Volatilité | Écart-type annualisé de 3 log-rendements disjoints, 50 % titre / 50 % médiane de la sous-industrie |
| z-scores | Relatifs à la sous-industrie, moyenne et variance rétrécies vers le secteur (k = 5), bornés à ±3 |
| Score final | Moyenne de 4 piliers à poids égal (8 facteurs) |
| Rendement attendu | Prime de risque implicite + IC × σ × score standardisé (IC = {ic:.2f}) |
| Taux sans risque | {rf:.2f} % : T-bill américain 3 mois au 13/11/2026 ([FRED DGS3MO]({an.RF_SOURCE})) |
| Corrélations | Structure par blocs {RHO[0]:.2f} / {RHO[1]:.2f} / {RHO[2]:.2f}, sensibilité dans l'onglet 9 |
| Portefeuille | Quotas par sous-industrie, score > médiane, Sharpe ex ante maximal, 5 % maximum par titre |
| Filtre ESG | Score ESG > médiane de la sous-industrie ; repêchage des bons scores (> médiane) dont l'ESG ≥ médiane de l'univers, priorité aux sous-industries absentes |
""")
    st.subheader("Table complète")
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Télécharger (CSV)", df.to_csv(index=False).encode(), "univers_enrichi.csv", "text/csv")
