"""Calculs de risque / rendement à partir des seules colonnes du fichier.

- Volatilité estimée sur 3 sous-périodes disjointes (année N-1, YTD hors dernier mois, dernier mois)
- Rétrécissement (shrinkage) vers la médiane de la sous-industrie
- Rendement attendu (plusieurs définitions au choix) et ratio de Sharpe
- Covariance « à blocs » (sous-industrie / secteur / univers) et frontière efficiente
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Hypothèses de corrélation (même sous-industrie, même secteur, secteurs différents) et jeux de sensibilité
# Corrélations supposées (même sous-industrie, même secteur, autres) : niveaux élevés, les valeurs technologiques
# et de communication partageant des moteurs communs (IA, taux, appétit pour la croissance)
RHO_CENTRAL = (0.70, 0.50, 0.35)
RHO_SETS = {"Basse": (0.60, 0.40, 0.25), "Centrale": RHO_CENTRAL, "Haute": (0.80, 0.60, 0.45)}
RF_DEFAULT = 4.32  # rendement du T-bill américain 3 mois au 13/11/2026 (FRED, série DGS3MO)
RF_SOURCE = "https://fred.stlouisfed.org/series/DGS3MO"


def period_lengths(data_date: pd.Timestamp) -> tuple[float, float, float]:
    """Durées (en années) des 3 sous-périodes disjointes observables dans le fichier."""
    t_ytd = (data_date - pd.Timestamp(year=data_date.year, month=1, day=1)).days / 365.25
    t_1m = 1 / 12
    return 1.0, max(t_ytd - t_1m, 1 / 52), t_1m


def multi_period_vol(df: pd.DataFrame, data_date: pd.Timestamp) -> pd.Series:
    """Volatilité annualisée (%) estimée sur 3 rendements non chevauchants.

    Périodes : année civile N-1 (total return), début d'année → il y a 1 mois
    (déduit de YTD et 1M), dernier mois. Modèle : log-rendement de la période k
    ~ N(μ·T_k, σ²·T_k). Estimateur : μ̂ = Σl_k / ΣT_k ;
    σ̂² = Σ (l_k − μ̂T_k)² / T_k / (3 − 1).
    """
    t1, t2, t3 = period_lengths(data_date)
    l1 = np.log1p(df["tr_prev_year"] / 100)
    l3 = np.log1p(df["ret_1m"] / 100)
    l2 = np.log1p(df["ret_ytd"] / 100) - l3          # YTD hors dernier mois
    L = np.c_[l1, l2, l3]
    T = np.array([t1, t2, t3])
    mu = L.sum(axis=1) / T.sum()
    var = ((L - mu[:, None] * T) ** 2 / T).sum(axis=1) / 2
    return pd.Series(np.sqrt(var) * 100, index=df.index)


def shrink_to_group(s: pd.Series, groups: pd.Series, alpha: float) -> pd.Series:
    """α × estimation individuelle + (1 − α) × médiane du groupe (réduit le bruit)."""
    med = s.groupby(groups).transform("median")
    return alpha * s + (1 - alpha) * med


def block_correlation(df: pd.DataFrame, rho_sub: float, rho_sec: float, rho_all: float) -> np.ndarray:
    """Corrélation constante par blocs : même sous-industrie > même secteur > reste."""
    sub = df["subindustry"].to_numpy()
    sec = df["sector"].to_numpy()
    R = np.full((len(df), len(df)), rho_all)
    R[sec[:, None] == sec[None, :]] = rho_sec
    R[sub[:, None] == sub[None, :]] = rho_sub
    np.fill_diagonal(R, 1.0)
    return R


def covariance(vol_pct: pd.Series, R: np.ndarray) -> np.ndarray:
    v = vol_pct.to_numpy() / 100
    return np.outer(v, v) * R


def port_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """Rendement en excès (%) et volatilité (%) d'un portefeuille."""
    return float(w @ mu), float(np.sqrt(w @ cov @ w) * 100)


def _min_var(mu, cov, cap, target=None):
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}]
    if target is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mu - target, "jac": lambda w: mu})
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), jac=lambda w: 2 * cov @ w,
                   bounds=[(0, cap)] * n, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x if res.success else None


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, cap: float, n_points: int = 25):
    """Frontière long-only avec plafond par titre : min variance pour chaque rendement cible.

    Renvoie (table des points, poids du portefeuille de variance minimale,
    poids du point de Sharpe maximal sur la frontière).
    """
    w_mv = _min_var(mu, cov, cap)
    r_min = float(w_mv @ mu) if w_mv is not None else float(np.min(mu))
    # rendement max atteignable avec le plafond : on remplit les meilleurs titres à cap
    k = int(np.floor(1 / cap))
    top = np.sort(mu)[::-1]
    r_max = float(top[:k].sum() * cap + (1 - k * cap) * (top[k] if k < len(top) else 0))
    rows, weights = [], []
    for tgt in np.linspace(r_min, r_max * 0.999, n_points):
        w = _min_var(mu, cov, cap, tgt)
        if w is None:
            continue
        r, v = port_stats(w, mu, cov)
        rows.append({"ret": r, "vol": v, "sharpe": r / v if v else np.nan})
        weights.append(w)
    front = pd.DataFrame(rows)
    w_ms = weights[int(front["sharpe"].idxmax())] if len(front) else None
    return front, w_mv, w_ms
