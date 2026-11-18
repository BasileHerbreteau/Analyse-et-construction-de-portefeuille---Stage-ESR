"""Critère de comparaison agrégé (score multi-facteurs) et rendement attendu ex ante.

1. Chaque facteur est orienté (plus haut = mieux), écrêté, puis transformé en z-score
   À L'INTÉRIEUR DE SA SOUS-INDUSTRIE : on compare un titre à ses vrais pairs.
   Pour les sous-industries de petite taille, la moyenne et l'écart-type du groupe sont
   « rétrécis » vers ceux du secteur (estimateur bayésien empirique) afin d'éviter des
   z-scores extrêmes calculés sur 1 ou 2 titres.
2. Score d'un pilier = moyenne des z-scores de ses facteurs.
   Score final = moyenne des piliers (chaque pilier pèse autant).
3. Rendement attendu en excès (ex ante) = niveau de marché + alpha issu des z-scores :
       μ_i = (1 − λ)·Prime_i + λ·(Upside_i − r_f) + IC · σ_i · Z^α_i
   Z^α = moyenne des piliers Croissance, Rentabilité, Valorisation et du z-score de l'upside.
   λ = 0 par défaut : l'upside entre par son z-score (information relative, sans le biais
   d'optimisme de son niveau) ; la prime de risque implicite donne le niveau.
   Z^α est re-standardisé (moyenne 0, écart-type 1) avant d'appliquer la règle de Grinold :
   alpha = coefficient d'information × volatilité × score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PILLARS: dict[str, dict[str, int]] = {
    "Croissance": {"ltg_eps": +1, "sales_growth": +1},
    "Rentabilité": {"profit_margin": +1, "fcf_yield": +1},
    "Valorisation": {"peg": -1, "earnings_yield": +1},
    "Attentes du marché": {"upside": +1, "risk_prem": +1},
}
FACTORS = {f: s for p in PILLARS.values() for f, s in p.items()}


def _shrunk_group_stats(x: pd.Series, group: pd.Series, parent: pd.Series, k: float):
    """Moyenne et écart-type du groupe, rétrécis vers le groupe parent.

    m = (n·m_groupe + k·m_parent) / (n + k) ; idem pour la variance.
    k = « nombre de titres fictifs » apportés par le parent (k = 0 : pas de rétrécissement).
    """
    n = x.groupby(group).transform("count")
    m_g = x.groupby(group).transform("mean")
    v_g = x.groupby(group).transform(lambda s: s.var(ddof=0))
    m_p = x.groupby(parent).transform("mean")
    v_p = x.groupby(parent).transform(lambda s: s.var(ddof=1))
    m = (n * m_g + k * m_p) / (n + k)
    v = (n * v_g + k * v_p) / (n + k)
    return m, np.sqrt(v)


def factor_zscores(df: pd.DataFrame, level: str = "subindustry", k: float = 5.0,
                   clip: float = 3.0) -> pd.DataFrame:
    """z-score orienté de chaque facteur, relatif aux pairs du niveau choisi."""
    out = pd.DataFrame(index=df.index)
    for f, sign in FACTORS.items():
        x = df[f].astype(float) * sign
        if level == "univers":
            m, s = x.mean(), x.std()
        elif level == "sector":
            m, s = _shrunk_group_stats(x, df["sector"], pd.Series("all", index=df.index), 0)
        else:
            m, s = _shrunk_group_stats(x, df["subindustry"], df["sector"], k)
        out[f] = ((x - m) / s).clip(-clip, clip)
    return out


def composite_score(z: pd.DataFrame) -> pd.DataFrame:
    """Scores par pilier + score final (moyenne des piliers)."""
    res = pd.DataFrame(index=z.index)
    for p, facs in PILLARS.items():
        res[p] = z[list(facs)].mean(axis=1)
    res["Score final"] = res[list(PILLARS)].mean(axis=1)
    res["Score fondamental"] = res[["Croissance", "Rentabilité", "Valorisation"]].mean(axis=1)
    # score utilisé pour l'alpha : fondamentaux + upside (la prime de risque est déjà le niveau)
    res["Score alpha"] = pd.concat([res[["Croissance", "Rentabilité", "Valorisation"]], z["upside"]],
                                   axis=1).mean(axis=1)
    return res


def expected_return(df: pd.DataFrame, z_alpha: pd.Series, vol: pd.Series, rf: float,
                    lam: float, ic: float) -> pd.DataFrame:
    """Rendement attendu en excès du sans-risque (%), décomposé en trois briques."""
    z_alpha = (z_alpha - z_alpha.mean()) / z_alpha.std()   # score re-standardisé (écart-type 1), règle de Grinold
    prem = (1 - lam) * df["risk_prem"]
    up = lam * (df["upside"] - rf)
    alpha = ic * vol * z_alpha
    return pd.DataFrame({"Ancrage prime de risque": prem, "Ancrage upside (niveau)": up,
                         "Alpha z-scores": alpha, "mu": prem + up + alpha}, index=df.index)


def rule40_comparison(score: pd.Series, r40: pd.Series, top_share: float = 0.30) -> dict:
    """Comparaison du score agrégé avec la Rule of 40."""
    passe = r40 >= 40
    thr = score.quantile(1 - top_share)
    top = score >= thr
    ct = pd.crosstab(top.map({True: f"Top {top_share:.0%} du score", False: "Reste"}),
                     passe.map({True: "Passe la Rule of 40", False: "Ne passe pas"}))
    return {"spearman": score.rank().corr(r40.rank()), "kendall": score.corr(r40, method="kendall"),
            "crosstab": ct, "top": top, "passe": passe,
            "accord": float((top == passe).mean()), "kappa": cohen_kappa(top, passe),
            "accord_hasard": float(top.mean() * passe.mean() + (1 - top.mean()) * (1 - passe.mean()))}


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """κ de Cohen : accord observé corrigé de l'accord dû au hasard (0 = hasard, 1 = parfait)."""
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan
