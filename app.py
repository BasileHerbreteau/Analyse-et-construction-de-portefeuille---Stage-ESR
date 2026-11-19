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
[role="tab"] p {{ font-weight: 600; }}
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
K_SHRINK = 3.0                           # rétrécissement des z-scores vers le secteur (≈ médiane des k estimés)
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
                           help="Chaque variable, sauf l'upside des analystes (conservé tel quel), est ramenée dans "
                                "ses quantiles [p ; 1 − p]. Aucun titre n'est supprimé.")
p = st.sidebar.slider("Quantile d'écrêtage p", 0.0, 0.10, 0.02, 0.01, disabled=not wins)
rf = st.sidebar.number_input("Taux sans risque (%)", 0.0, 10.0, an.RF_DEFAULT, 0.01,
                             help="T-bill américain 3 mois au 13/11/2026 : 4,32 % (Réserve fédérale, série FRED DGS3MO).")
lam = st.sidebar.slider("Part de l'upside brut dans le niveau de rendement (λ)", 0.0, 1.0, 0.0, 0.05,
                        help="0 : niveau = prime de risque implicite, l'upside n'intervient que relativement aux pairs "
                             "(z-score). 0,5 : moyenne 50/50 prime / upside brut.")
ic = st.sidebar.slider("Coefficient d'information (IC)", 0.0, 0.30, 0.10, 0.01,
                       help="Corrélation supposée entre le score et le rendement futur ; convertit le score en points de "
                            "rendement : alpha = IC × volatilité × score standardisé (règle de Grinold). Repères "
                            "(Grinold et Kahn) : 0,05 bon, 0,10 très bon, au-delà de 0,15 exceptionnel.")
CAP = 0.05
N_TOTAL = 30                              # titres sélectionnés dans le portefeuille

# --------------------------------------------------------------------------- #
# Calculs sur l'univers complet
# --------------------------------------------------------------------------- #
ALL_NUM = du.NUMERIC + du.DERIVED
WINS_COLS = [c for c in ALL_NUM if c != "upside"]   # l'upside des analystes est conservé tel quel
base = du.winsorize(df_full, WINS_COLS, p, 1 - p) if wins else df_full.copy()
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
ports = {"Sans ESG": build_portfolio(full, tuple(elig_all), N_TOTAL, CAP, RHO)}
if HAS_ESG:
    elig_esg = full["esg_above"] == 1
    rescue = full["esg"] >= full["esg"].median()
    ports["Avec filtre ESG"] = build_portfolio(full, tuple(elig_esg), N_TOTAL, CAP, RHO, tuple(rescue))
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
    "1 · Univers", "2 · Rendement & risque", "3 · Valorisation & attentes", "4 · Critère agrégé",
    "5 · Comparaison avec la Rule of 40", "6 · ESG", "7 · Rendement attendu & frontière",
    "8 · Portefeuille final", "Annexe",
])
# Au clic sur un onglet, la page revient au début de l'onglet (juste sous la barre figée).
st.iframe("""<script>
const doc = window.parent.document;
if (!doc.__tabScrollTop) {
  doc.__tabScrollTop = true;
  doc.addEventListener("click", (e) => {
    const btn = e.target.closest('[role="tab"]');
    if (!btn) return;
    const tabs = btn.closest('div[data-testid="stTabs"]');
    if (!tabs || tabs.parentElement.closest('div[data-testid="stTabs"]')) return;
    const header = doc.querySelector('header[data-testid="stHeader"]');
    const offset = header ? header.getBoundingClientRect().bottom : 0;
    const go = () => {
      const top = tabs.getBoundingClientRect().top - offset;
      if (top >= 0) return;
      let el = tabs.parentElement;
      while (el && el !== doc.body && !(el.scrollHeight > el.clientHeight &&
             /(auto|scroll)/.test(window.parent.getComputedStyle(el).overflowY))) el = el.parentElement;
      (el && el !== doc.body ? el : window.parent).scrollBy({top: top, behavior: "instant"});
    };
    requestAnimationFrame(go); setTimeout(go, 120);
  }, true);
}
</script>""", height="content")

# =========================================================================== #
# 1. Univers
# =========================================================================== #
with tabs[0]:
    k = st.columns(4)
    k[0].metric("Titres", len(df))
    k[1].metric("Information Technology", int((df.sector == "Information Technology").sum()))
    k[2].metric("Communication Services", int((df.sector == "Communication Services").sum()))
    k[3].metric("Titres américains", f"{(df.country == 'UNITED STATES').mean():.0%}")

    c1, c2 = st.columns([3, 2])
    with c1:
        subc = df.groupby(["sector", "subindustry"]).size().reset_index(name="n").sort_values("n")
        fig = px.bar(subc, x="n", y="subindustry", color="sector", orientation="h",
                     color_discrete_map=SECTOR_COLORS, template=TEMPLATE, text="n",
                     labels={"n": "Nombre de titres", "subindustry": ""}, title="Nombre de titres par sous-industrie")
        fig.update_layout(height=560, legend=dict(orientation="h", y=-0.12))
        show(fig, c1)
    with c2:
        cnt = df.country.value_counts()
        top = cnt[cnt >= 3]
        pays = pd.concat([top, pd.Series({"Autres pays": cnt[cnt < 3].sum()})]).rename_axis("Pays").reset_index(name="n")
        fig = px.pie(pays, names="Pays", values="n", hole=0.45, title="Répartition par pays",
                     color_discrete_sequence=[C_NAVY, C_BLUE, C_LIGHT, C_TEAL, C_PALE, C_GREY, "#2E5E8C", "#C5CED9"])
        fig.update_traces(textinfo="label+percent", marker_line_color="white", marker_line_width=2)
        fig.update_layout(height=560, showlegend=False)
        show(fig, c2)

    st.subheader("Profil des deux secteurs")
    synth_cols = ["ret_1m", "ret_ytd", "tr_prev_year", "vol", "sharpe_ante", "upside", "pe_next", "peg",
                  "fcf_yield", "ltg_eps", "sales_growth", "profit_margin", "rule_of_40", "risk_prem"] + ESG_COLS
    # sens favorable (+1 : plus haut = mieux, −1 : plus bas = mieux)
    FAV = {"ret_1m": 1, "ret_ytd": 1, "tr_prev_year": 1, "vol": -1, "sharpe_ante": 1, "upside": 1, "pe_next": -1,
           "peg": -1, "fcf_yield": 1, "ltg_eps": 1, "sales_growth": 1, "profit_margin": 1, "rule_of_40": 1,
           "risk_prem": 1, "esg": 1}
    med = df.groupby("sector")[synth_cols].median()
    prof_s = med.T
    prof_s["Univers"] = df[synth_cols].median()
    prof_s["Min"] = df[synth_cols].min()
    prof_s["Max"] = df[synth_cols].max()
    secs_ = [c_ for c_ in prof_s.columns if c_ in SECTOR_COLORS]

    inv_ = {lab(c_): c_ for c_ in prof_s.index}
    prof_s.index = [lab(c_) for c_ in prof_s.index]

    def _fav(row):
        out = [""] * len(row)
        sgn = FAV.get(inv_.get(row.name, row.name), 0)
        if len(secs_) == 2:
            a_, b_ = row[secs_[0]], row[secs_[1]]
            win = secs_[0] if (a_ - b_) * sgn > 0 else secs_[1] if (b_ - a_) * sgn > 0 else None
            if win:
                out[list(row.index).index(win)] = f"background-color: rgba(0, 161, 154, .22); font-weight: 600"
        return out
    prof_disp = prof_s.style.apply(_fav, axis=1).format("{:.2f}")
    st.dataframe(prof_disp, width="stretch", height=38 * (len(prof_s) + 1))
    st.caption("Médianes par secteur. Surligné : secteur le plus favorable sur l'indicateur (volatilité, P/E et PEG : "
               "le plus bas). Min / Max : univers.")

    _top_sub = df.subindustry.value_counts()
    _gap = df["ret_ytd"].mean() - df["ret_ytd"].median()
    pts_ = [f"<b>Composition</b> : {len(df)} titres, {(df.sector == 'Information Technology').mean():.0%} en technologie, "
            f"{(df.country == 'UNITED STATES').mean():.0%} américains. Les 3 premières sous-industries "
            f"({', '.join(_top_sub.index[:3])}) représentent {_top_sub.iloc[:3].sum() / len(df):.0%} de l'univers ; "
            f"{(_top_sub <= 2).sum()} sous-industries comptent 2 titres ou moins. L'équipondéré est donc avant tout une "
            "exposition technologie américaine : la diversification se pilote au niveau des sous-industries."]
    IT_, CS_ = "Information Technology", "Communication Services"
    if {IT_, CS_} <= set(df.sector):
        it, cs = med.loc[IT_], med.loc[CS_]

        def lead(v, hi_good=True):
            return "Communication Services" if (cs[v] > it[v]) == hi_good else "Information Technology"
        pts_ += [
            f"<b>Performance</b> : Communication Services devant sur le dernier mois (médiane {cs.ret_1m:+.1f} % contre "
            f"{it.ret_1m:+.1f} %) et sur l'année N-1 ({cs.tr_prev_year:.1f} % contre {it.tr_prev_year:.1f} %) ; "
            f"Information Technology nettement devant depuis le début de l'année (YTD {it.ret_ytd:+.1f} % contre "
            f"{cs.ret_ytd:+.1f} %). Le repli récent de la technologie suit donc un fort parcours annuel."
            if cs.ret_1m > it.ret_1m and it.ret_ytd > cs.ret_ytd else
            f"<b>Performance</b> médiane YTD : Information Technology {it.ret_ytd:+.1f} %, Communication Services "
            f"{cs.ret_ytd:+.1f} % ; sur 1 mois : {it.ret_1m:+.1f} % contre {cs.ret_1m:+.1f} %.",
            f"<b>Risque</b> : volatilité médiane {it.vol:.0f} % en technologie contre {cs.vol:.0f} % en communication "
            f"(× {it.vol / cs.vol:.1f}) ; {lead('sharpe_ante')} offre le meilleur Sharpe ex ante médian "
            f"({max(it.sharpe_ante, cs.sharpe_ante):.2f} contre {min(it.sharpe_ante, cs.sharpe_ante):.2f}).",
            f"<b>Valorisation</b> : la technologie est plus chère (P/E N+1 {it.pe_next:.1f}x contre {cs.pe_next:.1f}x) "
            f"mais sa croissance attendue est bien supérieure (BPA long terme {it.ltg_eps:.0f} % contre {cs.ltg_eps:.0f} %) : "
            f"rapportés à la croissance, les deux secteurs sont proches (PEG {it.peg:.2f} contre {cs.peg:.2f}).",
            f"<b>Rentabilité et cash</b> : marge nette plus élevée en technologie ({it.profit_margin:.1f} % contre "
            f"{cs.profit_margin:.1f} %), mais rendement FCF bien supérieur en communication ({cs.fcf_yield:.1f} % contre "
            f"{it.fcf_yield:.1f} %) : des titres moins chers qui génèrent davantage de cash par euro investi.",
            f"<b>Attentes</b> : upside médian {it.upside:.0f} % en technologie contre {cs.upside:.0f} %, prime de risque "
            f"{it.risk_prem:.1f} % contre {cs.risk_prem:.1f} % : le marché et les analystes attendent davantage de la "
            "technologie, en contrepartie d'un risque plus élevé.",
        ]
    pts_ += [f"<b>Univers</b> : perf. YTD moyenne {df.ret_ytd.mean():.1f} % contre médiane {df.ret_ytd.median():.1f} % "
             f"({'quelques très fortes hausses tirent la moyenne' if _gap > 0 else 'quelques fortes baisses tirent la moyenne'}) ; "
             f"upside des analystes de {df.upside.min():.0f} % à {df.upside.max():.0f} %."]
    recap(pts_, ["Valeurs écrêtées aux quantiles 2 % / 98 % (winsorisation) avant calcul, sauf l'upside des analystes, "
                 "conservé tel quel."])

# =========================================================================== #
# 2. Rendement & risque
# =========================================================================== #
with tabs[1]:
    t1, t2, t3 = an.period_lengths(DATA_DATE)
    exp_ = st.expander("Méthode d'estimation de la volatilité")
    c1, c2 = exp_.columns([3, 2])
    with c1:
        st.markdown(f"""
Le fichier fournit trois rendements sur des périodes **disjointes** : année civile N-1 ({t1:.2f} an), du 1er janvier
à il y a un mois ({t2:.2f} an, déduit de YTD et de la perf. 1 mois), dernier mois ({t3:.3f} an).

- **ℓ<sub>k</sub> = ln(1 + R<sub>k</sub>)** : log-rendement de la période k, R<sub>k</sub> étant son rendement simple ;
  les log-rendements s'additionnent d'une période à l'autre.
- **T<sub>k</sub>** : durée de la période k, en années.
- **Hypothèse N(μT<sub>k</sub> ; σ²T<sub>k</sub>)** : le log-rendement d'une période suit une loi normale dont la
  moyenne (μT<sub>k</sub>) et la variance (σ²T<sub>k</sub>) sont proportionnelles à sa durée. μ est le rendement
  annuel moyen, σ la volatilité annualisée recherchée. Chaque écart est donc rapporté à la durée de sa période, ce qui
  permet de combiner un mois et une année dans une même estimation.
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
        _hi = float(max(base["vol_obs"].max(), base["vol"].max()))
        fig.add_trace(go.Scatter(x=[0, _hi], y=[0, _hi], mode="lines", name="y = x (aucun ajustement)",
                                 line=dict(color=C_GREY, dash="dash", width=1.5)))
        fig.add_annotation(x=_hi * 0.85, y=_hi * 0.85, text="y = x", showarrow=False, yshift=12, font_color=C_GREY)
        fig.update_layout(height=380, showlegend=False)
        show(fig, c2)
        st.caption("Sous la diagonale : volatilité individuelle élevée, ramenée vers celle des pairs ; au-dessus : "
                   f"volatilité faible, relevée. Dispersion entre titres : {base['vol_obs'].std():.0f} pt avant, "
                   f"{base['vol'].std():.0f} pt après.")

    hz = df.groupby("sector")[["ret_1m", "ret_ytd", "tr_prev_year"]].median().reset_index() \
        .melt(id_vars="sector", var_name="Horizon", value_name="Rendement médian (%)")
    hz["Horizon"] = hz["Horizon"].map(lab)
    fig = px.bar(hz, x="Horizon", y="Rendement médian (%)", color="sector", barmode="group", text_auto=".1f",
                 color_discrete_map=SECTOR_COLORS, template=TEMPLATE, title="Rendement médian par horizon et par secteur")
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.2))
    show(fig)

    c1, c2 = st.columns(2)
    with c1:
        ret_var = "tr_prev_year"
        ret_opts = {"tr_prev_year": lab("tr_prev_year")}
        fig = px.scatter(df, x="vol", y=ret_var, color="sector", hover_name="ticker",
                         hover_data={"name": True, "subindustry": True}, color_discrete_map=SECTOR_COLORS,
                         template=TEMPLATE, labels={"vol": lab("vol"), ret_var: ret_opts[ret_var]},
                         title="Rendement vs volatilité")
        f_rv = ols_line(fig, df["vol"], df[ret_var])
        fig.update_layout(height=480)
        show(fig, c1)
        st.caption(f"Droite des moindres carrés : en moyenne, +10 pt de volatilité vont de pair avec "
                   f"{10 * f_rv['slope']:+.1f} pt de rendement (pente {f_rv['slope']:.2f}). R² {f_rv['r2']:.2f} : la "
                   f"volatilité explique {f_rv['r2']:.0%} des écarts de rendement entre titres.")
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
        st.caption(f"Trait plein y = x : même performance les deux années. Droite pointillée (régression) : pente "
                   f"{f_pp['slope']:.2f}, soit {10 * f_pp['slope']:+.1f} pt de YTD en moyenne pour +10 pt en N-1 "
                   f"(0 = aucune persistance, 1 = persistance parfaite) ; Spearman {f_pp['spearman']:.2f}.")

    _sv = df.groupby("sector").agg(tr=("tr_prev_year", "median"), ytd=("ret_ytd", "median"), v=("vol", "median"))
    _corr_rv = df["vol"].corr(df["tr_prev_year"], method="spearman")
    _f_n1 = du.ols_fit(df["vol"], df["tr_prev_year"])
    _f_ytd = du.ols_fit(df["vol"], df["ret_ytd"])
    _vs = df.groupby("subindustry")["vol"].median().sort_values()
    _pers = abs(f_pp["spearman"])
    recap(["<b>Rendements</b> : total return N-1 médian " + " ; ".join(f"{i} {r.tr:.1f} %" for i, r in _sv.iterrows())
           + " ; YTD médian " + " ; ".join(f"{i} {r.ytd:+.1f} %" for i, r in _sv.iterrows()) + ".",
           "<b>Risque</b> : volatilité médiane " + " ; ".join(f"{i} {r.v:.0f} %" for i, r in _sv.iterrows())
           + f". Sous-industries les plus risquées : {', '.join(_vs.index[-3:][::-1])} ; les moins risquées : "
           f"{', '.join(_vs.index[:3])}.",
           f"<b>Risque et rendement</b> : +10 pt de volatilité correspondent en moyenne à {10 * _f_n1['slope']:+.1f} pt de "
           f"rendement N-1 (R² {_f_n1['r2']:.0%}) et {10 * _f_ytd['slope']:+.1f} pt de YTD (R² {_f_ytd['r2']:.0%}). "
           + ("Le risque a été rémunéré, surtout cette année, mais il n'explique qu'une partie des écarts de "
              "performance : la sélection titre par titre reste déterminante."
              if _f_ytd["slope"] > 0 and _f_n1["slope"] > 0 else
              "Le lien entre risque et performance est faible ou instable sur ces deux périodes."),
           f"<b>Persistance</b> : pente {f_pp['slope']:.2f} et Spearman {f_pp['spearman']:.2f} entre N-1 et YTD. "
           + ("Les gagnants de l'an dernier ne sont pas ceux de cette année : la performance passée ne prédit pas la "
              "performance récente. Le momentum n'est donc pas retenu comme facteur du score."
              if _pers < 0.2 else
              "La performance passée se prolonge en partie : un effet momentum est présent sur la période."),
           f"<b>Estimation</b> : le rapprochement vers la sous-industrie réduit la dispersion des volatilités de "
           f"{base['vol_obs'].std():.0f} à {base['vol'].std():.0f} pt et corrige les estimations extrêmes dues à un "
           "seul mois atypique."],
          ["Volatilité annualisée estimée sur 3 log-rendements disjoints, rapprochée à 50 % de la médiane de la "
           "sous-industrie (encart « Méthode » en haut de l'onglet)."])

# =========================================================================== #
# 3. Valorisation & attentes
# =========================================================================== #
with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df, x="ltg_eps", y="pe_next", color="sector", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                         labels={"ltg_eps": lab("ltg_eps"), "pe_next": lab("pe_next")},
                         title="P/E N+1 vs croissance attendue (droites = PEG constant)")
        xs = np.linspace(max(df.ltg_eps.min(), 0.1), df.ltg_eps.max(), 50)
        y_max = float(df.pe_next.max() * 1.05)
        for g_, dash, col_, w_ in [(0.5, "dot", C_TEAL, 1.6), (1, "dash", C_GREY, 1.6), (2, "solid", C_RED, 2.2)]:
            fig.add_trace(go.Scatter(x=xs, y=g_ * xs, mode="lines", name=f"PEG = {g_:g}",
                                     line=dict(color=col_, dash=dash, width=w_)))
            x_lab = min(float(xs[-1]), y_max / g_ * 0.92)
            fig.add_annotation(x=x_lab, y=g_ * x_lab, text=f"PEG {g_:g}", showarrow=False, xshift=-4, yshift=10,
                               font=dict(color=col_, size=11))
        fig.update_yaxes(range=[0, y_max])
        show(fig, c1)
        st.caption("Au-dessus de la droite PEG = 2 : P/E élevé au regard de la croissance attendue ; sous PEG = 1 : "
                   "croissance obtenue à prix modéré.")
    with c2:
        fig = px.scatter(df, x="peg", y="upside", color="sector", hover_name="ticker",
                         color_discrete_map=SECTOR_COLORS, template=TEMPLATE,
                         labels={"peg": lab("peg"), "upside": lab("upside")}, title="PEG vs upside des analystes")
        ols_line(fig, df["peg"], df["upside"])
        show(fig, c2)

    _pg = df.groupby("sector")[["peg", "pe_next", "ltg_eps", "fcf_yield", "earnings_yield"]].median()
    _sp_pu = df.peg.corr(df.upside, method="spearman")
    _sp_gu = df.ltg_eps.corr(df.upside, method="spearman")
    _conv = (df.fcf_yield >= df.earnings_yield).mean()
    recap(["<b>PEG</b> médian : " + " ; ".join(f"{i} {v:.2f}" for i, v in _pg["peg"].items())
           + f" (univers {df.peg.median():.2f}). Répartition : {(df.peg < 1).mean():.0%} des titres sous 1 (croissance "
           f"payée à prix modéré), {((df.peg >= 1) & (df.peg <= 2)).mean():.0%} entre 1 et 2, "
           f"{(df.peg > 2).mean():.0%} au-dessus de 2 (valorisation exigeante au regard de la croissance).",
           "<b>Prix et croissance</b> : P/E N+1 médian " + " ; ".join(f"{i} {v:.1f}x" for i, v in _pg["pe_next"].items())
           + ", croissance LT du BPA " + " ; ".join(f"{i} {v:.0f} %" for i, v in _pg["ltg_eps"].items())
           + ". Les multiples élevés de la technologie sont en grande partie justifiés par la croissance attendue.",
           f"<b>Attentes des analystes</b> : upside médian {df.upside.median():.1f} %, {(df.upside > 0).mean():.0%} des "
           "objectifs de cours au-dessus du cours. Ce biais d'optimisme justifie de ne pas utiliser l'upside brut comme "
           "niveau de rendement attendu.",
           f"<b>Cohérence</b> : corrélation de rang PEG / upside {_sp_pu:.2f}, croissance LT / upside {_sp_gu:.2f}"
           + (" : les analystes voient davantage de potentiel sur les titres de forte croissance plutôt que sur les "
              "titres bon marché." if _sp_gu > abs(_sp_pu) else " : l'upside est davantage lié au prix payé qu'à la croissance."),
          ],
          ["Earnings yield N+1 = 100 ÷ P/E N+1, comparable au FCF yield."])

# =========================================================================== #
# 4. Critère agrégé : corrélations, piliers, score
# =========================================================================== #
with tabs[3]:
    st.subheader("Corrélations entre les variables du fichier")
    method = "spearman"
    corr_cols = du.NUMERIC + ["vol"] + ESG_COLS
    cm = df[corr_cols].corr(method=method).rename(index=du.LABELS, columns=du.LABELS)
    fig = px.imshow(cm, text_auto=".2f", aspect="auto", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    title="Matrice de corrélation de rang (Spearman)")
    fig.update_layout(height=640)
    show(fig)
    st.caption("Spearman : corrélation entre les classements des titres (rangs), de −1 à 1, insensible aux valeurs "
               "extrêmes.")
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

    st.subheader("Construction du score")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.latex(r"z_{i,f} = s_f\,\frac{x_{i,f} - \bar x_{g(i),f}}{\sigma_{g(i),f}}")
        st.latex(r"\text{Score}_i = \tfrac14\sum_{p=1}^{4}\tfrac{1}{|p|}\sum_{f\in p} z_{i,f}")
        st.markdown("""
- z-score de chaque facteur **relatif à la sous-industrie** g(i) ; s<sub>f</sub> = −1 pour le PEG.
- Moyenne et variance d'une sous-industrie de n titres rapprochées de celles du secteur :
  (n·groupe + 3·secteur) ÷ (n + 3), soit 50 % de poids propre à 3 titres, 85 % à 17 titres ; z-scores bornés à ±3.
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

    if len(pairs):
        _p0 = pairs.iloc[0]
        _neg = pc.where(~np.eye(4, dtype=bool)).stack().idxmin()
        _cf = df[["ltg_eps", "sales_growth", "profit_margin", "fcf_yield", "peg", "earnings_yield", "upside",
                  "risk_prem"]].corr(method="spearman")
        recap([f"<b>Relations les plus fortes</b> : {', '.join(f'{r.Paire} ({r.corr:+.2f})' for r in pairs.head(3).itertuples())}. "
               "Les variables très corrélées portent la même information : les additionner sans précaution reviendrait "
               "à la compter deux fois.",
               f"<b>Au sein des piliers</b> : croissance LT / croissance du CA {_cf.loc['ltg_eps', 'sales_growth']:+.2f}, "
               f"marge / FCF yield {_cf.loc['profit_margin', 'fcf_yield']:+.2f}, upside / prime de risque "
               f"{_cf.loc['upside', 'risk_prem']:+.2f} : chaque pilier regroupe des facteurs de même nature, moyennés "
               "avant d'entrer dans le score.",
               f"<b>Entre piliers</b> : corrélation absolue moyenne {off.mean():.2f} ; relation la plus négative entre "
               f"{_neg[0]} et {_neg[1]} ({pc.loc[_neg]:+.2f})"
               + (" : les sociétés en forte croissance sacrifient une partie de leur marge. "
                  if set(_neg) == {"Croissance", "Rentabilité"} else
                  " : la qualité et la croissance se paient. " if "Valorisation" in _neg else ". ")
               + f"Avec {n_eff:.1f} piliers effectivement indépendants sur 4, chaque pilier apporte une information propre : "
               "la pondération égale est justifiée."],
              ["Nombre effectif de piliers = (Σλ)² / Σλ², λ valeurs propres de la matrice de corrélation des piliers."])

# =========================================================================== #
# 5. Comparaison avec la Rule of 40
# =========================================================================== #
with tabs[4]:
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

    st.subheader("Comparaison du critère agrégé avec la Rule of 40")
    cmp_ = sc.rule40_comparison(df["Score final"], r40, passe_rate)
    both = int((cmp_["top"] & cmp_["passe"]).sum())
    n_pass = int(passe.sum())
    hasard = cmp_["top"].mean() * n_pass
    k = st.columns(4)
    k[0].metric("Corrélation de rang (Spearman)", f"{cmp_['spearman']:.2f}",
                help="Corrélation entre les deux classements (rangs) : 1 = classements identiques, 0 = aucun lien.")
    k[1].metric("Tau de Kendall", f"{cmp_['kendall']:.2f}",
                help="Pour deux titres pris au hasard : P(classés dans le même ordre) − P(ordre inverse).")
    k[2].metric("Titres retenus par les deux", f"{both} / {n_pass}",
                help=f"Comparaison à taille égale : top {passe_rate:.0%} du score vs titres passant la règle.")
    k[3].metric("Recouvrement attendu sans lien", f"{hasard:.0f} / {n_pass}",
                help="Nombre de titres communs si les deux critères étaient indépendants : "
                     f"{n_pass} × {cmp_['top'].mean():.0%}.")
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
    recap([f"<b>Classements liés mais distincts</b> : Spearman {cmp_['spearman']:.2f}, Kendall {cmp_['kendall']:.2f} "
           f"(deux titres pris au hasard sont classés dans le même ordre par les deux critères dans "
           f"{(1 + cmp_['kendall']) / 2:.0%} des cas).",
           f"<b>Recouvrement</b> : {both} titres sur {n_pass} retenus par les deux critères ({both / n_pass:.0%}), contre "
           f"{hasard:.0f} attendus si les critères étaient indépendants : l'accord est réel mais partiel, "
           f"{n_pass - both} titres sur {n_pass} diffèrent.",
           f"<b>Piliers</b> : {best_p} est le plus proche de la Rule of 40 (ρ {corr_p[best_p]:.2f}), {worst_p} le plus "
           f"éloigné (ρ {corr_p[worst_p]:.2f})" + (" : la Rule of 40 mesure la performance opérationnelle, pas le prix "
                                                  "payé pour l'obtenir." if worst_p == "Valorisation" else "."),
           f"<b>Profil</b> : la Rule of 40 retient des titres plus chers (PEG médian {pr_.peg.median():.2f} contre {tp_.peg.median():.2f}) "
           f"et plus concentrés ({pr_.subindustry.nunique()} sous-industries contre {tp_.subindustry.nunique()}).",
           "<b>Conclusion</b> : la Rule of 40 est un filtre de qualité opérationnelle, absolu et favorable aux éditeurs de "
           "logiciels ; le score intègre le prix payé et les attentes, relativement aux pairs. Le portefeuille repose "
           "sur le score ; la Rule of 40 sert d'indicateur de contrôle."],
          ["Comparaison à taille égale : top du score de même effectif que les titres passant la règle.",
           "Kendall = (paires concordantes − paires discordantes) ÷ nombre de paires."])

# =========================================================================== #
# 6. ESG
# =========================================================================== #
with tabs[5]:
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
        de = de.assign(quintile=pd.qcut(de.esg.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]))
        qt = de.groupby("quintile", observed=True)[["esg", "tr_prev_year", "ret_ytd", "vol", "sharpe_ante", "upside"]].mean()
        with st.expander("Sources des scores ESG"):
            st.dataframe(df_full.loc[df_full.esg.notna(), ["ticker", "name", "esg", "esg_provider", "esg_date", "source_url"]],
                         hide_index=True, width="stretch",
                         column_config={"source_url": st.column_config.LinkColumn("Source")})
        _sp = du.ols_fit(de["esg"], de["tr_prev_year"])["spearman"]
        _sv_ = du.ols_fit(de["esg"], de["vol"])["spearman"]
        _ss_ = du.ols_fit(de["esg"], de["Score final"])["spearman"]
        _em = de.groupby("sector").esg.median()
        _us = de.groupby(de.country == "UNITED STATES").esg.median()
        recap([f"<b>Couverture</b> : {len(de)} titres notés sur {len(df)}. Score médian "
               + " ; ".join(f"{i} {v:.0f}" for i, v in _em.items())
               + (f" ; États-Unis {_us.get(True, np.nan):.0f} contre {_us.get(False, np.nan):.0f} hors États-Unis"
                  + (" : les sociétés non américaines, majoritairement européennes, sont mieux notées."
                     if _us.get(False, 0) > _us.get(True, 0) else ".") if len(_us) == 2 else "."),
               f"<b>Performance</b> : corrélation de rang ESG / total return N-1 {_sp:.2f}"
               + (" : pas de lien net entre ESG et performance sur la période." if abs(_sp) < 0.2
                  else (" : les meilleurs scores ESG ont plutôt surperformé." if _sp > 0
                        else " : les meilleurs scores ESG ont plutôt sous-performé."))
               + f" Quintile ESG le plus haut : total return N-1 moyen {qt['tr_prev_year'].iloc[-1]:.1f} % contre "
               f"{qt['tr_prev_year'].iloc[0]:.1f} % pour le plus bas.",
               f"<b>Risque</b> : corrélation ESG / volatilité {_sv_:.2f}"
               + (" : les titres les mieux notés sont en moyenne moins volatils."
                  if _sv_ < -0.2 else " : pas de lien marqué entre ESG et risque."),
               f"<b>Score final</b> : corrélation ESG / score {_ss_:.2f}"
               + (" : l'ESG est une information indépendante du critère agrégé ; le filtre ESG modifie donc la "
                  "sélection sans la dégrader mécaniquement." if abs(_ss_) < 0.2 else "."),
               "Le filtre ESG du portefeuille compare chaque titre à la médiane ESG de sa sous-industrie (titres notés), "
               "ce qui neutralise les écarts structurels de notation entre industries."])

# =========================================================================== #
# 7. Rendement attendu & frontière efficiente
# =========================================================================== #
with tabs[6]:
    st.subheader("Rendement attendu ex ante")
    st.latex(r"\mu_i = \underbrace{(1-\lambda)\,\text{Prime}_i + \lambda\,(\text{Upside}_i - r_f)}_{\text{niveau}} + "
             r"\underbrace{IC\cdot\sigma_i\cdot \tilde Z^{\alpha}_i}_{\text{alpha z-scores}}, \qquad "
             r"\text{Sharpe}^{ex\,ante}_i = \frac{\mu_i}{\sigma_i}")
    st.markdown(f"""
- **Niveau** : prime de risque implicite, rendement en excès du sans-risque déjà intégré au fichier (λ = {lam:.2f}).
  L'upside brut n'est pas utilisé en niveau : {(df_full.upside > 0).mean():.0%} des objectifs de cours sont au-dessus du
  cours (médiane {df_full.upside.median():.1f} %), et une moyenne 50/50 prime / upside donnerait un rendement attendu
  médian de {MU_5050.median():.1f} %.
- **Alpha** : Z<sup>α</sup> = moyenne des piliers Croissance, Rentabilité, Valorisation et du z-score de l'upside.
  Une moyenne de z-scores imparfaitement corrélés est moins dispersée qu'un z-score (écart-type
  {SCORES['Score alpha'].std():.2f}) : elle est re-standardisée, Z̃<sup>α</sup> = (Z<sup>α</sup> − moyenne) ÷ écart-type
  sur les {len(full)} titres, pour retrouver un écart-type de 1 ; IC = {ic:.2f}. L'upside apporte ainsi son information
  relative aux pairs, sans son biais de niveau.
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
    t_cols = [c_ for c_ in regs.columns if c_.endswith("(t)")]
    st.dataframe(regs.style.format(precision=2)
                 .map(lambda v: f"background-color: {C_PALE}; font-weight: 600" if abs(v) > 2 else "", subset=t_cols),
                 hide_index=True, width="stretch")
    st.caption("Régression linéaire en coupe sur les 99 titres : variable = a + b₁·Croissance + b₂·Rentabilité + "
               "b₃·Valorisation. Coefficient : variation de la variable (en points) pour +1 point de pilier, les deux "
               "autres étant fixés. t = coefficient ÷ son erreur-type ; |t| > 2 (surligné) : effet significatif au seuil "
               "de 5 %. R² : part des écarts entre titres expliquée par les trois piliers ensemble. Données à une même "
               "date : test de cohérence, pas un backtest.")

    _sh = df.groupby("sector")["sharpe_ante"].median()
    _r = regs.set_index("Variable expliquée")
    def _sig(nm):
        s_ = [p_ for p_ in ["Croissance", "Rentabilité", "Valorisation"] if abs(_r.loc[nm, f"{p_} (t)"]) > 2]
        return ", ".join(f"{p_} ({'+' if _r.loc[nm, f'{p_} (coef.)'] > 0 else '−'})" for p_ in s_) or "aucun pilier"
    _mu_s = df.groupby("sector")["mu"].median()
    _recap7 = ([f"<b>Niveau</b> : rendement attendu médian {df.mu.median():.1f} % en excès du taux sans risque ("
           + " ; ".join(f"{i} {v:.1f} %" for i, v in _mu_s.items()) + ").",
           f"<b>Alpha</b> : de {df['Alpha z-scores'].min():+.1f} à {df['Alpha z-scores'].max():+.1f} pt selon les titres ; "
           f"corrélation de rang μ / score final {df.mu.corr(df['Score final'], method='spearman'):.2f} : le rendement "
           "attendu reflète le score sans s'y réduire, la prime de risque de chaque titre restant le socle.",
           "<b>Sharpe ex ante</b> médian : " + " ; ".join(f"{i} {v:.2f}" for i, v in _sh.items())
           + " : la technologie offre plus de rendement attendu mais aussi plus de volatilité.",
           f"<b>Pertinence</b> : les trois piliers expliquent {_r.loc[lab('upside'), 'R²']:.0%} des écarts d'upside "
           f"(significatifs : {_sig(lab('upside'))}) et {_r.loc[lab('risk_prem'), 'R²']:.0%} des écarts de prime "
           f"de risque ({_sig(lab('risk_prem'))}) : analystes et marché valorisent des caractéristiques mesurées par le "
           f"score. Sur les performances passées, R² {_r.loc[lab('tr_prev_year'), 'R²']:.0%} en N-1 et "
           f"{_r.loc[lab('ret_ytd'), 'R²']:.0%} en YTD."],
          ["Alpha = IC × σ × score standardisé (règle de Grinold).",
           f"Sharpe ex post = (total return N-1 − {rf:.2f} %) ÷ volatilité estimée."])

    st.subheader("Frontière efficiente")
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
        P_s = build_portfolio(full, tuple(elig_all), N_TOTAL, CAP, rh)
        w_s = P_s["Poids"].reindex(full.index).fillna(0).to_numpy()
        r_e, v_e = an.port_stats(EW.to_numpy(), full["mu"].to_numpy(float), cov_s)
        r_p, v_p = an.port_stats(w_s, full["mu"].to_numpy(float), cov_s)
        sens.append({"Hypothèse (sous-ind. / secteur / autres)": f"{nm} ({rh[0]:.2f} / {rh[1]:.2f} / {rh[2]:.2f})",
                     "Équipondéré : volatilité (%)": v_e, "Équipondéré : Sharpe": r_e / v_e,
                     "Portefeuille sans ESG : volatilité (%)": v_p, "Portefeuille sans ESG : Sharpe": r_p / v_p,
                     "Écart de Sharpe": r_p / v_p - r_e / v_e})
    st.dataframe(pd.DataFrame(sens).style.format(precision=2), hide_index=True, width="stretch")
    _sens = pd.DataFrame(sens)
    _fr = []
    if len(df) * CAP >= 1:
        _rw = pd.DataFrame(rows).set_index("Portefeuille")
        if {"Équipondéré", "Sharpe maximal"} <= set(_rw.index):
            e_, m_ = _rw.loc["Équipondéré"], _rw.loc["Sharpe maximal"]
            _fr.append(f"<b>Frontière</b> : l'équipondéré ({e_['Rendement attendu (%)']:.1f} % pour "
                       f"{e_['Volatilité (%)']:.1f} % de volatilité) se situe nettement sous la frontière. Le portefeuille "
                       f"de Sharpe maximal atteint {m_['Rendement attendu (%)']:.1f} % pour {m_['Volatilité (%)']:.1f} % "
                       f"(Sharpe {m_['Sharpe']:.2f} contre {e_['Sharpe']:.2f}) avec {int(m_['Lignes'])} lignes.")
        if "Variance minimale" in _rw.index:
            v_ = _rw.loc["Variance minimale"]
            _fr.append(f"<b>Variance minimale</b> : {v_['Volatilité (%)']:.1f} % de volatilité, obtenue en concentrant les "
                       "poids sur les titres les moins volatils et en diversifiant entre secteurs.")
    recap(_recap7[0] + _fr + [f"Avec l'hypothèse centrale, le portefeuille sans ESG améliore le Sharpe ex ante de "
           f"{_sens.loc[1, 'Écart de Sharpe']:+.2f} par rapport à l'équipondéré ; l'écart reste positif sous les "
           f"hypothèses basse ({_sens.loc[0, 'Écart de Sharpe']:+.2f}) et haute ({_sens.loc[2, 'Écart de Sharpe']:+.2f})."
           if (_sens["Écart de Sharpe"] > 0).all() else
           "Le classement des portefeuilles dépend des hypothèses de corrélation : voir le tableau de sensibilité."],
          _recap7[1] + ["Covariance Σ = D·R·D : D = volatilités estimées, R = corrélations par blocs.",
           "Frontière : minimisation de wᵀΣw pour chaque rendement cible, 0 ≤ w ≤ 5 %, Σw = 1 (SLSQP)."])

# =========================================================================== #
# 8. Portefeuille final
# =========================================================================== #
with tabs[7]:
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
        fig.update_layout(height=max(500, 20 * len(Pw) + 120), legend=dict(orientation="h", y=-0.16))
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
                          xaxis_title="Poids (%)", title="Poids par sous-industrie", legend=dict(orientation="h", y=-0.16))
        show(fig, c2)

    st.subheader("Comparaison avec le portefeuille équipondéré")
    # sens favorable de chaque indicateur (+1 : plus haut = mieux, −1 : plus bas = mieux, 0 : descriptif)
    SENS = {"Rendement attendu (%)": 1, "Volatilité estimée (%)": -1, "Sharpe ex ante": 1, "Score final moyen": 1,
            "Score ESG moyen": 1, "Total return N-1 (%)": 1, "Sharpe ex post N-1": 1,
            "Perf. YTD (%)": 1, "Upside moyen (%)": 1, "Prime de risque moy. (%)": 1, "PEG moyen (x)": -1,
            "Part passant la Rule of 40 (%)": 1, "Sous-industries": 1, "Poids max. d'une sous-industrie (%)": -1}
    KEY_ROWS = ["Rendement attendu (%)", "Volatilité estimée (%)", "Sharpe ex ante", "Score final moyen",
                "Score ESG moyen"]

    def comp_table(rows_):
        num = comp.loc[[r_ for r_ in rows_ if r_ in comp.index]]
        ew_ = num.iloc[:, 0]
        txt = num.copy().astype(object)
        css = pd.DataFrame("", index=num.index, columns=num.columns)
        for r_ in num.index:
            dec_ = 0 if r_ in ("Lignes", "Sous-industries") else 2
            txt.loc[r_, num.columns[0]] = f"{ew_[r_]:.{dec_}f}"
            css.loc[r_, num.columns[0]] = "background-color: #F2F4F7; color: #4A5A70"
            scale = 0.5 * max(num.loc[r_].abs().max(), 1e-9)
            for c_ in num.columns[1:]:
                v_, d_ = num.loc[r_, c_], num.loc[r_, c_] - ew_[r_]
                txt.loc[r_, c_] = f"{v_:.{dec_}f}   ({d_:+.{dec_}f})"
                s_ = SENS.get(r_, 0)
                inten = min(1.0, abs(d_) / scale)
                if s_ and inten >= 0.05:
                    rgb = "0, 161, 154" if s_ * d_ > 0 else "214, 69, 61"
                    css.loc[r_, c_] = (f"background-color: rgba({rgb}, {0.12 + 0.55 * inten:.2f});"
                                       + (" font-weight: 700" if inten > 0.5 else ""))
        return txt.style.apply(lambda _: css, axis=None)

    st.markdown("**Indicateurs clés**")
    st.dataframe(comp_table(KEY_ROWS), width="stretch")
    st.markdown("**Détail**")
    st.dataframe(comp_table([r_ for r_ in comp.index if r_ not in KEY_ROWS]), width="stretch", height=560)
    st.caption("Entre parenthèses : écart avec l'équipondéré. Vert : écart favorable ; rouge : défavorable ; intensité "
               "proportionnelle à l'ampleur de l'écart. Lignes non colorées : indicateurs descriptifs. Performances N-1 "
               "et YTD : poids actuels appliqués aux rendements passés (description, pas un backtest).")

    c2 = st.container()
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

    fiche = st.expander("Fiche titre : profil d'une ligne du portefeuille")
    hold = Pw.sort_values("Poids", ascending=False)
    choice = fiche.selectbox("Titre", hold.ticker + " – " + hold.name + "  (" + (hold["Poids"] * 100).round(1).astype(str) + " %)")
    tk = choice.split(" – ")[0]
    row = full[full.ticker == tk].iloc[0]
    fiche.markdown(f"**{row['name']}** · {row['sector']} · {row['subindustry']} · {row['country']}")
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
    c1, c2 = fiche.columns([3, 2])
    show(fig, c1)
    vcols = du.NUMERIC + ["rule_of_40", "vol", "mu", "sharpe_ante"] + PILLARS + ["Score final"] + ESG_COLS
    vals = pd.DataFrame({"Valeur": row[vcols].astype(float), "Médiane sous-industrie": peers[vcols].median(),
                         "Médiane univers": full[vcols].median()})
    vals.index = [lab(c) for c in vals.index]
    c2.dataframe(vals.style.format("{:.2f}"), width="stretch", height=500)

    best = comp.loc["Sharpe ex ante"].idxmax()
    recap([f"<b>Portefeuille {cur_label}</b> : {int(cur['Lignes'])} lignes, rendement attendu {cur['Rendement attendu (%)']:.1f} %, "
           f"volatilité {cur['Volatilité estimée (%)']:.1f} %, Sharpe ex ante {cur['Sharpe ex ante']:.2f} contre "
           f"{ewc['Sharpe ex ante']:.2f} pour l'équipondéré.",
           f"<b>Qualité</b> : score final moyen {cur['Score final moyen']:.2f} (équipondéré {ewc['Score final moyen']:.2f}) ; "
           f"{cur['Part passant la Rule of 40 (%)']:.0f} % des poids passent la Rule of 40 "
           f"(équipondéré {ewc['Part passant la Rule of 40 (%)']:.0f} %).",
           f"<b>Diversification</b> : {int(cur['Sous-industries'])} sous-industries ; poids maximal d'une sous-industrie "
           f"{cur['Poids max. d\'une sous-industrie (%)']:.0f} % ; technologie {cur['Part technologie (%)']:.0f} %.",
           f"<b>Risque</b> : les 5 premières contributions représentent "
           f"{Pw['Contribution au risque'].nlargest(5).sum():.0%} du risque pour {Pw['Poids'].nlargest(5).sum():.0%} des "
           f"poids ; {int((Pw['Poids'] >= CAP - 1e-6).sum())} titres au plafond de 5 %.",
           (f"<b>Filtre ESG</b> : score ESG moyen {comp.loc['Score ESG moyen'].iloc[2]:.1f} contre "
            f"{comp.loc['Score ESG moyen'].iloc[1]:.1f} sans filtre, pour un Sharpe ex ante de "
            f"{comp.loc['Sharpe ex ante'].iloc[2]:.2f} contre {comp.loc['Sharpe ex ante'].iloc[1]:.2f} : le gain "
            f"extra-financier coûte {comp.loc['Sharpe ex ante'].iloc[1] - comp.loc['Sharpe ex ante'].iloc[2]:.2f} de "
            f"Sharpe" + (", et les deux portefeuilles restent nettement au-dessus de l'équipondéré."
                         if comp.loc["Sharpe ex ante"].iloc[1:].min() > comp.loc["Sharpe ex ante"].iloc[0] + 0.1 else ".")
            if comp.shape[1] > 2 else f"Meilleur Sharpe ex ante : {best}.")],
          [f"Quotas : {N_TOTAL} places réparties entre sous-industries au prorata de leur taille (plus forts restes) ; "
           "un titre n'est retenu que si son score dépasse la médiane de sa sous-industrie ; places non pourvues "
           "attribuées aux meilleurs scores restants.",
           "Pondération : maximisation du Sharpe ex ante, 0 ≤ w ≤ 5 %, Σw = 1 (SLSQP, plusieurs points de départ).",
           "Nombre effectif de titres = 1 / Σw² ; contribution au risque = w_i (Σw)_i / wᵀΣw."])

# =========================================================================== #
# Annexe : méthodes et limites
# =========================================================================== #
with tabs[8]:
    st.subheader("Synthèse des méthodes")
    st.markdown(f"""
| Étape | Méthode |
| --- | --- |
| Valeurs extrêmes | Winsorisation : chaque variable, sauf l'upside des analystes, est bornée à ses quantiles {p:.0%} et {1 - p:.0%} |
| Volatilité | Écart-type annualisé de 3 log-rendements disjoints, 50 % titre / 50 % médiane de la sous-industrie |
| z-scores | Relatifs à la sous-industrie, moyenne et variance rapprochées du secteur (poids de 3 titres), bornés à ±3 |
| Score final | Moyenne de 4 piliers à poids égal (8 facteurs) |
| Rendement attendu | Prime de risque implicite + IC × σ × score standardisé (IC = {ic:.2f}) |
| Taux sans risque | {rf:.2f} % : T-bill américain 3 mois au 13/11/2026 ([FRED DGS3MO]({an.RF_SOURCE})) |
| Corrélations | Structure par blocs {RHO[0]:.2f} / {RHO[1]:.2f} / {RHO[2]:.2f}, sensibilité dans l'onglet 7 |
| Portefeuille | 30 titres : quotas par sous-industrie, score > médiane, Sharpe ex ante maximal, 5 % maximum par titre |
| Filtre ESG | Score ESG > médiane de la sous-industrie ; repêchage des bons scores (> médiane) dont l'ESG ≥ médiane de l'univers, priorité aux sous-industries absentes |
""")
    st.subheader("Limites et angles morts")
    c1, c2 = st.columns(2)
    c1.markdown("""
**Données**
- Une seule date d'observation, sans historique de prix : volatilités estimées sur 3 rendements seulement,
  corrélations supposées (structure par blocs) et non estimées.
- Ni capitalisation ni liquidité dans le fichier : une méga-capitalisation et une petite valeur sont traitées de la
  même façon ; capacité, coûts de transaction et liquidité ne sont pas mesurés.
- Devise des rendements non précisée : le risque de change d'un investisseur en euros n'est pas modélisé.
- Upside, croissance LT du BPA et PEG reposent sur le consensus des analystes : prévisions structurellement
  optimistes, parfois anciennes ; le PEG perd son sens quand la croissance attendue est faible ou négative.
- Prime de risque implicite calculée par Bloomberg selon son propre modèle : non vérifiable.
- Sous-industries GICS de 1 ou 2 titres : comparaisons entre pairs fragiles, malgré le rapprochement vers le secteur.

**ESG**
- Scores relatifs à l'industrie, de dates différentes ; titres non notés exclus du filtre ESG.
- Un autre fournisseur (MSCI, Sustainalytics) peut classer différemment : les notations ESG divergent fortement.
- Médiane ESG d'une sous-industrie de 2 ou 3 titres peu robuste.
""")
    c2.markdown("""
**Modèle**
- Pondération égale des piliers : choix de simplicité et de robustesse, non optimisé ; d'autres poids changeraient
  le classement.
- Score relatif aux pairs : une sous-industrie entièrement chère peut contenir des titres bien notés.
- IC = 0,10 : hypothèse standard, non estimée faute de rendements futurs ; l'alpha en dépend directement.
- Aucun backtest : les régressions de pertinence sont en coupe, à une même date ; le pouvoir prédictif du score
  n'est pas démontré.
- Hypothèse de loi normale pour la volatilité : les risques extrêmes (queues épaisses) sont sous-estimés.
- Rule of 40 conçue pour les éditeurs de logiciels : son seuil de 40 a moins de sens pour les télécoms ou le matériel.

**Portefeuille**
- La maximisation du Sharpe amplifie les erreurs d'estimation : l'optimiseur surpondère les titres dont le
  rendement attendu est surestimé ; le plafond de 5 % et les quotas limitent cet effet sans le supprimer.
- Expositions factorielles non contrôlées : bêta, taille, style croissance / value, momentum.
- Portefeuille statique : ni rééquilibrage, ni rotation, ni coûts de transaction.
- Performances N-1 et YTD du portefeuille : poids actuels appliqués au passé (biais rétrospectif), à lire comme
  une description et non comme une performance réalisable.
""")
    st.subheader("Table complète")
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Télécharger (CSV)", df.to_csv(index=False).encode(), "univers_enrichi.csv", "text/csv")
