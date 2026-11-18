"""Construction du portefeuille final.

1. Univers éligible : tout l'univers, ou (interrupteur ESG activé) seulement les titres dont le score
   ESG est strictement supérieur à la médiane ESG de leur sous-industrie.
2. Quotas : le nombre de titres retenus dans chaque sous-industrie est proportionnel à sa taille dans
   l'univers (méthode des plus forts restes), pour un total N.
3. Sélection : dans chaque sous-industrie, on prend les meilleurs scores finaux, mais seulement s'ils
   battent la médiane de leur sous-industrie (on ne force pas un titre faible pour remplir un quota).
   Les places non pourvues vont aux meilleurs titres restants de l'univers (toujours au-dessus de la
   médiane de leur groupe), puis, à défaut, aux meilleurs scores restants.
4. Pondération : maximisation du Sharpe ex ante, 0 ≤ w ≤ plafond, Σw = 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import analytics as an


def quotas(sizes: pd.Series, n_total: int) -> pd.Series:
    """Répartition de n_total places proportionnellement aux tailles (méthode des plus forts restes)."""
    raw = sizes / sizes.sum() * n_total
    q = np.floor(raw).astype(int)
    rest = int(n_total - q.sum())
    if rest > 0:
        order = (raw - q).sort_values(ascending=False).index[:rest]
        q[order] += 1
    return q


def select(df: pd.DataFrame, n_total: int, eligible: pd.Series, score: str = "Score final") -> pd.DataFrame:
    """Renvoie les titres retenus avec la raison de leur sélection."""
    med = df.groupby("subindustry")[score].transform("median")
    strong = eligible & (df[score] > med)
    q = quotas(df["subindustry"].value_counts(), n_total)
    picked, reason = [], {}
    for g, n_g in q.items():
        if n_g == 0:
            continue
        cand = df[strong & (df.subindustry == g)].sort_values(score, ascending=False).head(n_g)
        for i in cand.index:
            picked.append(i)
            reason[i] = "Quota de sa sous-industrie"
    for pool, label in [(strong, "Place restante (meilleur score restant, > médiane)"),
                        (eligible, "Complément (meilleur score éligible restant)")]:
        need = n_total - len(picked)
        if need <= 0:
            break
        rest = df[pool & ~df.index.isin(picked)].sort_values(score, ascending=False).head(need)
        for i in rest.index:
            picked.append(i)
            reason[i] = label
    out = df.loc[picked].copy()
    out["Raison"] = pd.Series(reason)
    out["Quota sous-industrie"] = out["subindustry"].map(q)
    return out


def max_sharpe(mu: np.ndarray, cov: np.ndarray, cap: float) -> np.ndarray:
    """Poids qui maximisent (w·μ) / √(wᵀΣw) sous 0 ≤ w ≤ cap et Σw = 1.

    Optimisation SLSQP lancée depuis plusieurs points de départ (équipondéré et meilleurs points
    de la frontière), on garde la meilleure solution.
    """
    n = len(mu)

    def neg_sharpe(w):
        return -(w @ mu) / np.sqrt(w @ cov @ w)

    starts = [np.full(n, 1 / n)]
    front, w_mv, w_ms = an.efficient_frontier(mu, cov, cap, n_points=15)
    starts += [w for w in (w_ms, w_mv) if w is not None]
    best, best_val = None, np.inf
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}]
    for w0 in starts:
        res = minimize(neg_sharpe, w0, bounds=[(0, cap)] * n, constraints=cons, method="SLSQP",
                       options={"maxiter": 1000, "ftol": 1e-12})
        w = np.clip(res.x, 0, cap)
        w = w / w.sum()
        val = neg_sharpe(w)
        if val < best_val and abs(w.sum() - 1) < 1e-6 and w.max() <= cap + 1e-6:
            best, best_val = w, val
        v0 = neg_sharpe(w0)
        if v0 < best_val:
            best, best_val = w0, v0
    best[best < 1e-5] = 0
    return best / best.sum()


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Part de chaque titre dans la variance du portefeuille : w_i (Σw)_i / wᵀΣw (somme = 1)."""
    m = cov @ w
    return w * m / (w @ m)


def summary(name: str, w: pd.Series, df: pd.DataFrame, cov_full: np.ndarray, rf: float) -> dict:
    """Indicateurs d'un portefeuille (w indexé comme df, poids nuls hors portefeuille)."""
    w = w.reindex(df.index).fillna(0.0)
    wa = w.to_numpy()
    r, v = an.port_stats(wa, df["mu"].to_numpy(float), cov_full)
    tr = float(wa @ df["tr_prev_year"].to_numpy(float))
    ytd = float(wa @ df["ret_ytd"].to_numpy(float))
    esg_w = w[df["esg"].notna()] if "esg" in df else pd.Series(dtype=float)
    esg = float((esg_w * df.loc[esg_w.index, "esg"]).sum() / esg_w.sum()) if len(esg_w) and esg_w.sum() > 0 else np.nan
    sub_w = w.groupby(df["subindustry"]).sum()
    return {
        "Portefeuille": name,
        "Lignes": int((wa > 0).sum()),
        "Nb effectif (1/Σw²)": 1 / np.sum(wa ** 2),
        "Rendement attendu (%)": r,
        "Volatilité estimée (%)": v,
        "Sharpe ex ante": r / v,
        "Total return N-1 (%)": tr,
        "Sharpe ex post N-1": (tr - rf) / v,
        "Perf. YTD (%)": ytd,
        "Score final moyen": float(wa @ df["Score final"].to_numpy(float)),
        "Upside moyen (%)": float(wa @ df["upside"].to_numpy(float)),
        "Prime de risque moy. (%)": float(wa @ df["risk_prem"].to_numpy(float)),
        "PEG moyen (x)": float(wa @ df["peg"].to_numpy(float)),
        "Part passant la Rule of 40 (%)": 100 * float(w[df["rule_of_40"] >= 40].sum()),
        "Score ESG moyen": esg,
        "Part technologie (%)": 100 * float(w[df["sector"] == "Information Technology"].sum()),
        "Part États-Unis (%)": 100 * float(w[df["country"] == "UNITED STATES"].sum()),
        "Sous-industries": int((sub_w > 0).sum()),
        "Poids max. d'une sous-industrie (%)": 100 * float(sub_w.max()),
        "Poids max. d'un titre (%)": 100 * float(w.max()),
    }
