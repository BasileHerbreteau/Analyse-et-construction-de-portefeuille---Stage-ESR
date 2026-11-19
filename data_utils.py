"""Chargement, nettoyage et variables dérivées de l'univers « universe of stock 261113 ».

Toutes les grandeurs proviennent exclusivement du fichier Excel : aucune donnée
extérieure n'est utilisée. Les variables dérivées sont de simples calculs à
partir des colonnes d'origine.
"""
from __future__ import annotations

from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", module="openpyxl")

DATA_PATH = Path(__file__).parent / "data" / "universe_of_stock_261113.xlsx"

# Renommage : noms courts et lisibles pour le code (colonne Excel -> nom interne)
RENAME = {
    "Upside from Analyst Aggregate view": "upside",
    "Ticker": "ticker",
    "Short Name": "name",
    "GICS Sector": "sector",
    "GICS SubInd Name": "subindustry",
    "Cntry Terrtry Fl Name": "country",
    "RSI 14D": "rsi14",
    "%1M": "ret_1m",
    "%YTD": "ret_ytd",
    "Previous 1 Calendar Year - Historical Total Return": "tr_prev_year",
    "FCF Yld": "fcf_yield",
    "Est P/E Nxt Yr": "pe_next",
    "BEst Est LTG EPS": "ltg_eps",
    "BEst PEG Ratio": "peg",
    "Rsk Prem": "risk_prem",
    "SALES_GROWTH": "sales_growth",
    "PROF_MARGIN": "profit_margin",
}

# Libellés français affichés dans l'application (unités incluses)
LABELS = {
    "upside": "Upside analystes (%)",
    "rsi14": "RSI 14 jours",
    "ret_1m": "Perf. 1 mois (%)",
    "ret_ytd": "Perf. YTD (%)",
    "tr_prev_year": "Total return année N-1 (%)",
    "fcf_yield": "FCF yield (%)",
    "pe_next": "P/E estimé N+1 (x)",
    "ltg_eps": "Croissance LT BPA estimée (%)",
    "peg": "PEG (x)",
    "risk_prem": "Prime de risque (%)",
    "sales_growth": "Croissance du CA (%)",
    "profit_margin": "Marge nette (%)",
    # dérivées
    "earnings_yield": "Earnings yield N+1 (%)",
    "rule_of_40": "Rule of 40 (croissance CA + marge, %)",
    "vol_proxy_1m": "Proxy vol. : |perf 1M| × √12 (%)",
    "rsi_dist": "Distance au RSI neutre |RSI − 50|",
    "momentum_gap": "Écart YTD − 1 mois (pts)",
    "fcf_minus_ey": "FCF yield − earnings yield (pts)",
    # risque / rendement (calculés dans analytics.py)
    "vol": "Volatilité estimée (% annualisé)",
    "mu": "Rendement attendu ex ante (%, en excès de r_f)",
    "sharpe_ante": "Sharpe ex ante",
    "sharpe_post": "Sharpe ex post (année N-1)",
    # ESG (fichier externe sourcé, optionnel)
    "esg": "Score ESG",
}

NUMERIC = [
    "upside", "rsi14", "ret_1m", "ret_ytd", "tr_prev_year", "fcf_yield",
    "pe_next", "ltg_eps", "peg", "risk_prem", "sales_growth", "profit_margin",
]
DERIVED = ["earnings_yield", "rule_of_40", "vol_proxy_1m",
           "rsi_dist", "momentum_gap", "fcf_minus_ey"]
CATEGORICAL = ["sector", "subindustry", "country"]

# Sens économique : +1 = « plus c'est haut, mieux c'est », -1 = l'inverse.
# Sert aux z-scores orientés (scoring futur) ; 0 = pas d'orientation évidente.
DIRECTION = {
    "upside": +1, "rsi14": 0, "ret_1m": 0, "ret_ytd": 0, "tr_prev_year": 0,
    "fcf_yield": +1, "pe_next": -1, "ltg_eps": +1, "peg": -1, "risk_prem": +1,
    "sales_growth": +1, "profit_margin": +1, "earnings_yield": +1,
    "rule_of_40": +1,
}

# Palette inspirée de la charte Amundi (bleus) : bleu marine, bleu, bleu clair, sarcelle
C_NAVY, C_BLUE, C_LIGHT, C_TEAL, C_PALE, C_RED, C_GREY = (
    "#002855", "#0062A3", "#5FB3E4", "#00A19A", "#A9CBEA", "#D6453D", "#8A99AD")
SECTOR_COLORS = {
    "Information Technology": C_BLUE,
    "Communication Services": C_LIGHT,
}


ESG_PATH = Path(__file__).parent / "data" / "esg_scores.csv"
ESG_COLUMNS = ["ticker", "name", "esg_score", "esg_scale", "esg_provider", "esg_date", "source_url"]


def load_esg(path: Path | str = ESG_PATH) -> pd.DataFrame | None:
    """Scores ESG sourcés (fichier optionnel, une ligne par ticker, avec l'URL de la source).

    esg_scale = "risk" si un score bas est meilleur (ex. Sustainalytics ESG Risk Rating),
    "score" si un score haut est meilleur (ex. Bloomberg ESG Disclosure, LSEG, S&P CSA).
    """
    path = Path(path)
    if not path.exists():
        return None
    esg = pd.read_csv(path)
    esg = esg.dropna(subset=["esg_score"])
    return esg if len(esg) else None


CAPS_PATH = Path(__file__).parent / "data" / "market_caps.csv"


def load_caps(path: Path | str = CAPS_PATH) -> pd.DataFrame | None:
    """Capitalisations boursières en Md$ (source externe, information complémentaire).

    lignes_univers = 2 quand une même société occupe deux lignes de l'univers (deux cotations ou deux
    catégories d'actions) : sa capitalisation est alors partagée entre les deux lignes pour la pondération.
    """
    path = Path(path)
    if not path.exists():
        return None
    caps = pd.read_csv(path).dropna(subset=["mcap_usd_bn"])
    return caps if len(caps) else None


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    df = df.dropna(how="all")
    return df


def prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.rename(columns=RENAME).copy()
    for c in CATEGORICAL + ["ticker", "name"]:
        df[c] = df[c].astype(str).str.strip()
    # L'upside est stocké en fraction (0.25 = 25 %) : on le passe en % comme le reste
    df["upside"] = df["upside"] * 100
    # ---- variables dérivées (calculs simples sur les colonnes du fichier) ----
    df["earnings_yield"] = 100 / df["pe_next"]
    df["rule_of_40"] = df["sales_growth"] + df["profit_margin"]
    df["vol_proxy_1m"] = df["ret_1m"].abs() * np.sqrt(12)
    df["rsi_dist"] = (df["rsi14"] - 50).abs()
    df["momentum_gap"] = df["ret_ytd"] - df["ret_1m"]
    df["fcf_minus_ey"] = df["fcf_yield"] - df["earnings_yield"]
    return df.reset_index(drop=True)


def winsorize(df: pd.DataFrame, cols: list[str], lower: float, upper: float) -> pd.DataFrame:
    """Écrête chaque colonne à ses quantiles [lower, upper] (sans supprimer de ligne)."""
    out = df.copy()
    for c in cols:
        lo, hi = out[c].quantile([lower, upper])
        out[c] = out[c].clip(lo, hi)
    return out


def outlier_table(df: pd.DataFrame, cols: list[str], k: float = 3.0) -> pd.DataFrame:
    """Valeurs hors [Q1 − k·IQR, Q3 + k·IQR] (règle de Tukey, k=3 = « extrêmes »)."""
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = (df[c] < lo) | (df[c] > hi)
        for _, r in df.loc[mask].iterrows():
            rows.append({"Variable": LABELS.get(c, c), "Ticker": r["ticker"],
                         "Société": r["name"], "Valeur": r[c],
                         "Borne basse": lo, "Borne haute": hi})
    return pd.DataFrame(rows)


def sector_relative_z(df: pd.DataFrame, cols: list[str], group: str = "sector") -> pd.DataFrame:
    """Z-score de chaque titre par rapport à la médiane / écart-type robuste (MAD) de son groupe."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        g = df.groupby(group)[c]
        med = g.transform("median")
        mad = g.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
        mad = mad.replace(0, np.nan)
        out[c] = (df[c] - med) / mad
    return out


def group_stats(df: pd.DataFrame, group: str, cols: list[str]) -> pd.DataFrame:
    """Médiane, moyenne et dispersion (écart-type inter-titres) par groupe."""
    agg = df.groupby(group)[cols].agg(["count", "median", "mean", "std"])
    return agg


def ols_fit(x: pd.Series, y: pd.Series) -> dict:
    m = x.notna() & y.notna()
    x, y = x[m].to_numpy(float), y[m].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    pearson = np.corrcoef(x, y)[0, 1]
    spearman = pd.Series(x).rank().corr(pd.Series(y).rank())
    return {"slope": slope, "intercept": intercept, "r2": r2,
            "pearson": pearson, "spearman": spearman, "n": len(x)}


def ols_multi(y: pd.Series, X: pd.DataFrame) -> dict:
    """Régression linéaire multiple (MCO) avec constante : coefficients, t de Student, R²."""
    m = y.notna() & X.notna().all(axis=1)
    yv = y[m].to_numpy(float)
    Xv = np.c_[np.ones(m.sum()), X[m].to_numpy(float)]
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    e = yv - Xv @ beta
    s2 = e @ e / (len(yv) - Xv.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xv.T @ Xv)))
    names = ["const"] + list(X.columns)
    r2 = 1 - e @ e / ((yv - yv.mean()) @ (yv - yv.mean()))
    return {"coef": dict(zip(names, beta)), "t": dict(zip(names, beta / se)), "r2": float(r2), "n": int(m.sum())}

