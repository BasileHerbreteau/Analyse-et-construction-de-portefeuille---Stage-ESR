"""Allocation dynamique ex post : track record du 01/01/2026 au 30/10/2026.

Exercice « Exercice allocation 261113 » : construire, sur 29 sous-jacents (indices sectoriels STOXX, Russell 2000,
ETF), le portefeuille dont l'allocation change dans le temps et qui maximise la performance totale réalisée, avec
  * 20 % maximum par sous-jacent,
  * 10 réallocations au plus (l'allocation de départ n'est pas comptée).

Méthode (optimum exact, pas une heuristique) :
  1. Entre deux dates de réallocation, le portefeuille est détenu tel quel (« buy and hold ») : la valeur finale d'une
     période est Σ wᵢ × Pᵢ(fin) / Pᵢ(début), linéaire en w. Sous les contraintes 0 ≤ wᵢ ≤ 20 % et Σ wᵢ = 100 %,
     le maximum est atteint en donnant 20 % aux 5 meilleurs sous-jacents de la période.
  2. Le choix des dates se ramène alors à un problème de plus court chemin : programmation dynamique sur
     (date, nombre de réallocations déjà faites), qui donne l'optimum global en quelques millisecondes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).parent / "data" / "exercice_allocation_261113.xlsx"
START, END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-30")
CAP, MAX_REBAL = 0.20, 10
ANN_DAYS = 252  # jours de cotation par an pour annualiser la volatilité

# Libellés indicatifs (codes Bloomberg du fichier). Zone : EUR = indices et ETF cotés en euros, USD = cotés en dollars.
INFO = {
    "SXKP Index": ("STOXX Europe 600 Télécommunications", "Europe", "EUR"),
    "SX6E Index": ("EURO STOXX Services aux collectivités", "Zone euro", "EUR"),
    "SX3P Index": ("STOXX Europe 600 Alimentation & boissons", "Europe", "EUR"),
    "SXEP Index": ("STOXX Europe 600 Énergie", "Europe", "EUR"),
    "SXPP Index": ("STOXX Europe 600 Matières premières", "Europe", "EUR"),
    "SXIP Index": ("STOXX Europe 600 Assurance", "Europe", "EUR"),
    "SX7E Index": ("EURO STOXX Banques", "Zone euro", "EUR"),
    "SXAP Index": ("STOXX Europe 600 Automobile", "Europe", "EUR"),
    "SX8P Index": ("STOXX Europe 600 Technologie", "Europe", "EUR"),
    "SXQE Index": ("EURO STOXX Produits ménagers & de soin", "Zone euro", "EUR"),
    "SX86P Index": ("STOXX Europe 600 Immobilier", "Europe", "EUR"),
    "SXNP Index": ("STOXX Europe 600 Biens & services industriels", "Europe", "EUR"),
    "SXRP Index": ("STOXX Europe 600 Distribution", "Europe", "EUR"),
    "SXFP Index": ("STOXX Europe 600 Services financiers", "Europe", "EUR"),
    "SXOP Index": ("STOXX Europe 600 Construction & matériaux", "Europe", "EUR"),
    "SX4P Index": ("STOXX Europe 600 Chimie", "Europe", "EUR"),
    "SXMP Index": ("STOXX Europe 600 Médias", "Europe", "EUR"),
    "SXTP Index": ("STOXX Europe 600 Voyages & loisirs", "Europe", "EUR"),
    "SX6P Index": ("STOXX Europe 600 Services aux collectivités", "Europe", "EUR"),
    "SX7P Index": ("STOXX Europe 600 Banques", "Europe", "EUR"),
    "SX8E Index": ("EURO STOXX Technologie", "Zone euro", "EUR"),
    "SXDP Index": ("STOXX Europe 600 Santé", "Europe", "EUR"),
    "SXKE Index": ("EURO STOXX Télécommunications", "Zone euro", "EUR"),
    "RTY Index": ("Russell 2000 (petites capitalisations US)", "États-Unis", "USD"),
    "WELE GY Equity": ("ETF Amundi S&P 500 Equal Weight ESG (Xetra)", "États-Unis", "EUR"),
    "IXCPR Index": ("S&P Communication Services Select Sector", "États-Unis", "USD"),
    "SXQP Index": ("STOXX Europe 600 Produits ménagers & de soin", "Europe", "EUR"),
    "IXN US Equity": ("ETF iShares Global Tech", "Monde", "USD"),
    "WELD GY Equity": ("ETF Amundi S&P Global Utilities ESG (Xetra)", "Monde", "EUR"),
}
US_MARKET = ["RTY Index", "IXCPR Index", "IXN US Equity"]  # cotés à New York


def clean_ticker(t: str) -> str:
    """'sxrp index' -> 'SXRP Index' : homogénéise la casse des codes Bloomberg du fichier."""
    parts = str(t).split()
    return " ".join(parts[:-1]).upper() + " " + parts[-1].capitalize()


def label(t: str) -> str:
    return INFO.get(t, (t,))[0]


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def load_prices(path: Path | str = DATA_FILE) -> pd.DataFrame:
    """Prix quotidiens (jours calendaires, week-ends et fériés reportés), du plus ancien au plus récent."""
    px = pd.read_excel(path, sheet_name=0, index_col=0)
    px.index = pd.to_datetime(px.index)
    px.index.name = "Date"
    px.columns = [clean_ticker(c) for c in px.columns]
    return px.sort_index().astype(float)


def load_example(path: Path | str = DATA_FILE) -> tuple[list[pd.Timestamp], pd.DataFrame]:
    """Allocation d'exemple fournie dans le fichier (onglet « Exmple de date de rebal »)."""
    raw = pd.read_excel(path, sheet_name=1, header=None)
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    dates = [pd.Timestamp(x) for x in raw.iloc[1, 1:].tolist()]
    w = raw.iloc[2:, 1:].astype(float).fillna(0.0)
    w.index = [clean_ticker(t) for t in raw.iloc[2:, 0]]
    w.columns = dates
    return dates, w


def market_days(px: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque date : marché européen ouvert, marché américain ouvert (prix majoritairement modifiés)."""
    moved = px.diff().abs() > 0
    us = [c for c in US_MARKET if c in px.columns]
    eu = [c for c in px.columns if c not in us]
    out = pd.DataFrame(index=px.index)
    out["europe"] = moved[eu].mean(axis=1) >= 0.5
    out["usa"] = moved[us].mean(axis=1) >= 0.5 if us else True
    out["cotation"] = moved.any(axis=1)             # au moins un prix a bougé : jour de cotation
    out["commun"] = out.europe & out.usa            # toutes les places ouvertes : réallocation possible
    return out


def period_prices(px: pd.DataFrame, start=START, end=END) -> pd.DataFrame:
    return px.loc[start:end]


def candidate_dates(px: pd.DataFrame, start=START, end=END, rule: str = "commun") -> pd.DatetimeIndex:
    """Dates où une réallocation est permise, strictement entre le départ et la fin.

    rule = "commun"     : toutes les places ouvertes (aucun prix figé)  -> choix par défaut
           "cotation"   : au moins une place ouverte
           "calendaire" : tous les jours, week-ends compris (prix reportés)
    """
    p = period_prices(px, start, end)
    inner = p.index[(p.index > start) & (p.index < end)]
    if rule == "calendaire":
        return inner
    md = market_days(px)
    return inner[md.loc[inner, rule].values]


# --------------------------------------------------------------------------- #
# Brique 1 : meilleure allocation sur une période
# --------------------------------------------------------------------------- #
def cap_profile(n: int, cap: float) -> np.ndarray:
    """Poids triés par ordre décroissant : cap, cap, ..., reste, 0, ... (somme = 100 %)."""
    w = np.zeros(n)
    left = 1.0
    for i in range(n):
        w[i] = min(cap, left)
        left -= w[i]
        if left <= 1e-12:
            break
    if left > 1e-9:
        raise ValueError(f"Plafond {cap:.0%} trop bas pour {n} sous-jacents : impossible d'investir 100 %.")
    return w


def best_weights(gross: pd.Series, cap: float = CAP) -> pd.Series:
    """Allocation qui maximise Σ wᵢ Rᵢ sous 0 ≤ wᵢ ≤ cap et Σ wᵢ = 1 : remplir les meilleurs Rᵢ jusqu'au plafond."""
    order = gross.sort_values(ascending=False).index
    w = pd.Series(cap_profile(len(gross), cap), index=order)
    return w[w > 0]


def segment_values(prices: np.ndarray, cap: float) -> np.ndarray:
    """G[a, b] = log de la meilleure valeur atteignable entre les nœuds a et b (b > a), -inf sinon."""
    n = prices.shape[0]
    prof = cap_profile(prices.shape[1], cap)
    gross = prices[None, :, :] / prices[:, None, :]            # (a, b, actif)
    best = np.sort(gross, axis=2)[:, :, ::-1] @ prof            # meilleure combinaison plafonnée
    g = np.full((n, n), -np.inf)
    iu = np.triu_indices(n, k=1)
    g[iu] = np.log(best[iu])
    return g


# --------------------------------------------------------------------------- #
# Brique 2 : choix optimal des dates (programmation dynamique)
# --------------------------------------------------------------------------- #
@dataclass
class Solution:
    dates: list[pd.Timestamp]          # départ + dates de réallocation
    weights: pd.DataFrame              # sous-jacents x dates
    gross: float                       # valeur finale de 1 € investi
    n_rebal: int


def optimise(px: pd.DataFrame, max_rebal: int = MAX_REBAL, cap: float = CAP, start=START, end=END,
             rule: str = "commun") -> Solution:
    """Optimum global : dates de réallocation et poids qui maximisent la performance totale réalisée."""
    sol, _ = _dp(px, max_rebal, cap, start, end, rule)
    return sol


def frontier(px: pd.DataFrame, k_max: int = 20, cap: float = CAP, start=START, end=END,
             rule: str = "commun") -> pd.Series:
    """Meilleure performance totale atteignable avec au plus k réallocations, k = 0..k_max."""
    _, best_by_k = _dp(px, k_max, cap, start, end, rule)
    return pd.Series(np.exp(np.maximum.accumulate(best_by_k)) - 1, index=range(k_max + 1), name="perf")


def _dp(px, max_rebal, cap, start, end, rule):
    p = period_prices(px, start, end)
    nodes = pd.DatetimeIndex([start]).append(candidate_dates(px, start, end, rule)).append(pd.DatetimeIndex([end]))
    P = p.loc[nodes].to_numpy()
    G = segment_values(P, cap)
    n, K = len(nodes), max_rebal
    V = np.full((K + 1, n), -np.inf)       # V[m, j] : meilleur log-gain en arrivant au nœud j après m réallocations
    arg = np.full((K + 1, n), -1, dtype=int)
    V[0, 0] = 0.0
    for m in range(1, K + 1):
        cand = V[m - 1][:, None] + G       # (i, j) : venir de i, réallouer en j
        cand[:, -1] = -np.inf              # on ne réalloue pas le dernier jour
        arg[m] = np.argmax(cand, axis=0)
        V[m] = cand[arg[m], np.arange(n)]
        V[m, 0] = -np.inf
    final = V + G[:, -1][None, :]          # dernière période jusqu'à la fin
    final[:, -1] = -np.inf
    best_by_k = final.max(axis=1)          # exactement m réallocations
    m_best = int(np.argmax(best_by_k))     # le plus petit nombre de réallocations qui atteint l'optimum
    # Retour arrière : nœuds de réallocation
    j = int(np.argmax(final[m_best]))
    path = [j]
    for m in range(m_best, 0, -1):
        j = int(arg[m, j])
        path.append(j)
    path = path[::-1]                      # commence par 0 (départ)
    dates = [nodes[i] for i in path]
    bounds = dates + [end]
    cols = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        cols[a] = best_weights(p.loc[b] / p.loc[a], cap)
    w = pd.DataFrame(cols).reindex(px.columns).fillna(0.0)
    sol = Solution(dates=dates, weights=w, gross=float(np.exp(best_by_k[m_best])), n_rebal=m_best)
    return sol, best_by_k


def daily_oracle(px: pd.DataFrame, cap: float = CAP, start=START, end=END) -> float:
    """Borne théorique : réallouer chaque jour de cotation vers les 5 meilleurs du lendemain (réallocations illimitées)."""
    p = period_prices(px, start, end)
    nodes = pd.DatetimeIndex([start]).append(candidate_dates(px, start, end, "commun")).append(pd.DatetimeIndex([end]))
    P = p.loc[nodes].to_numpy()
    prof = cap_profile(P.shape[1], cap)
    step = np.sort(P[1:] / P[:-1], axis=1)[:, ::-1] @ prof
    return float(np.prod(step) - 1)


# --------------------------------------------------------------------------- #
# Track record et indicateurs
# --------------------------------------------------------------------------- #
def backtest(px: pd.DataFrame, weights: pd.DataFrame, start=START, end=END, cost_bps: float = 0.0,
             base: float = 100.0) -> tuple[pd.Series, pd.DataFrame]:
    """Valeur liquidative quotidienne (base 100) d'une allocation : poids fixés aux dates de réallocation (colonnes de
    `weights`), puis détention sans intervention jusqu'à la date suivante. Frais : cost_bps × rotation (Σ|Δw|).

    Renvoie la VL et un tableau par période (dates, performance, rotation, frais)."""
    p = period_prices(px, start, end)
    w = weights.reindex(px.columns).fillna(0.0)
    dates = [pd.Timestamp(d) for d in w.columns]
    if dates[0] != start:
        raise ValueError("La première allocation doit être datée du jour de départ.")
    nav = pd.Series(np.nan, index=p.index, name="VL")
    rows, level, drift = [], base, None
    bounds = dates + [end]
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        wk = w[dates[k]]
        # Rotation = Σ|nouveau poids − poids dérivé| ; l'investissement initial n'est pas facturé.
        turnover = np.nan if drift is None else float((wk - drift).abs().sum())
        fee = 0.0 if drift is None else level * turnover * cost_bps / 1e4
        level -= fee                                   # VL du jour de réallocation : nette des frais
        seg = p.loc[a:b]
        val = level * (seg / seg.iloc[0] * wk).sum(axis=1)
        nav.loc[a:b] = val.values
        end_val = (seg.iloc[-1] / seg.iloc[0]) * wk
        drift = end_val / end_val.sum()                # poids dérivés à la fin de la période
        rows.append({"Début": a, "Fin": b, "Jours": (b - a).days,
                     "Performance": float(val.iloc[-1] / level - 1), "Rotation": turnover, "Frais (pts)": fee})
        level = float(val.iloc[-1])
    return nav, pd.DataFrame(rows)


def equal_weight(px: pd.DataFrame, start=START) -> pd.DataFrame:
    return pd.DataFrame({start: pd.Series(1.0 / px.shape[1], index=px.columns)})


def metrics(nav: pd.Series, px: pd.DataFrame, rf: float = 0.0) -> dict:
    """Indicateurs du track record. Volatilité calculée sur les jours de cotation (week-ends exclus)."""
    md = market_days(px).reindex(nav.index)
    tr = nav[md.cotation.values | (nav.index == nav.index[0])]
    r = tr.pct_change().dropna()
    days = (nav.index[-1] - nav.index[0]).days
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann = (1 + total) ** (365 / days) - 1
    vol = r.std() * np.sqrt(ANN_DAYS)
    dd = nav / nav.cummax() - 1
    return {
        "Performance totale": total,
        "Performance annualisée": ann,
        "Volatilité annualisée": vol,
        "Ratio de Sharpe": (ann - rf) / vol if vol > 0 else np.nan,
        "Perte maximale (max drawdown)": dd.min(),
        "Ratio de Calmar": ann / abs(dd.min()) if dd.min() < 0 else np.nan,
        "Jours de hausse": (r > 0).mean(),
        "Pire jour": r.min(),
        "Meilleur jour": r.max(),
    }


def contributions(px: pd.DataFrame, weights: pd.DataFrame, start=START, end=END, base: float = 100.0) -> pd.Series:
    """Contribution de chaque sous-jacent à la performance, en points de VL (somme = VL finale − 100)."""
    p = period_prices(px, start, end)
    w = weights.reindex(px.columns).fillna(0.0)
    dates = [pd.Timestamp(d) for d in w.columns] + [end]
    level, contrib = base, pd.Series(0.0, index=px.columns)
    for k, (a, b) in enumerate(zip(dates[:-1], dates[1:])):
        gain = w[dates[k]] * (p.loc[b] / p.loc[a] - 1) * level
        contrib += gain
        level += gain.sum()
    return contrib


def periods_table(px: pd.DataFrame, weights: pd.DataFrame, start=START, end=END) -> pd.DataFrame:
    """Une ligne par période : sous-jacents détenus, performance du portefeuille, de l'équipondéré et du meilleur."""
    p = period_prices(px, start, end)
    dates = [pd.Timestamp(d) for d in weights.columns] + [end]
    rows = []
    for k, (a, b) in enumerate(zip(dates[:-1], dates[1:])):
        R = p.loc[b] / p.loc[a] - 1
        wk = weights[dates[k]]
        held = wk[wk > 0].index
        rows.append({
            "Période": "Départ" if k == 0 else f"Réallocation {k}",
            "Début": a.date(), "Fin": b.date(), "Jours": (b - a).days,
            "Sous-jacents détenus (20 % chacun)": ", ".join(sorted(held, key=lambda t: -R[t])),
            "Portefeuille": float((wk * R).sum()),
            "Équipondéré (29)": float(R.mean()),
            "Meilleur sous-jacent": f"{R.idxmax()} ({R.max():+.1%})",
            "Pire sous-jacent": f"{R.idxmin()} ({R.min():+.1%})",
        })
    return pd.DataFrame(rows)


def max_drift(px: pd.DataFrame, weights: pd.DataFrame, start=START, end=END) -> tuple[float, str, pd.Timestamp]:
    """Poids le plus élevé atteint entre deux réallocations (les poids dérivent avec les prix)."""
    p = period_prices(px, start, end)
    dates = [pd.Timestamp(d) for d in weights.columns] + [end]
    best = (0.0, "", start)
    for k, (a, b) in enumerate(zip(dates[:-1], dates[1:])):
        v = p.loc[a:b] / p.loc[a] * weights[dates[k]]
        d = v.div(v.sum(axis=1), axis=0)
        if d.max().max() > best[0]:
            best = (float(d.max().max()), d.max().idxmax(), d.max(axis=1).idxmax())
    return best
