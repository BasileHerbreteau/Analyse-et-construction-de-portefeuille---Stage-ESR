"""Variante de l'allocation dynamique : plafond de 20 % respecté CHAQUE JOUR, pas seulement aux dates d'allocation.

Entre deux dates, les poids dérivent avec les prix. Pour qu'aucun sous-jacent ne dépasse 20 % un jour donné t :
    wᵢ Rᵢ(t) ≤ 20 % × Σⱼ wⱼ Rⱼ(t)      pour tout i et tout jour de cotation t de la période,
contraintes linéaires en w. Sur chaque période possible, on résout donc un programme linéaire (HiGHS), puis la même
programmation dynamique que dans allocation.py choisit les dates. Calcul long (≈ 15 min, 21 000 programmes linéaires) :
le résultat est enregistré dans data/variante_plafond_quotidien.csv et lu par l'application et le classeur Excel.

    python outils/variante_plafond_quotidien.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import allocation as al  # noqa: E402

OUT = ROOT / "data" / "variante_plafond_quotidien.csv"


def main():
    px = al.load_prices()
    p = al.period_prices(px)
    md = al.market_days(px)
    nodes = pd.DatetimeIndex([al.START]).append(al.candidate_dates(px)).append(pd.DatetimeIndex([al.END]))
    check = p.index[md.loc[p.index, "cotation"].values | (p.index == al.START)]   # jours contrôlés
    P = p.loc[check].to_numpy()
    pos = {d: i for i, d in enumerate(check)}
    ni = [pos[d] for d in nodes]
    n, N, cap = len(nodes), P.shape[1], al.CAP
    G = np.full((n, n), -np.inf)
    W = {}
    t0 = time.time()
    for a in range(n):
        for b in range(a + 1, n):
            R = P[ni[a]:ni[b] + 1] / P[ni[a]]                      # (jours, sous-jacents)
            A = (R[:, :, None] * np.eye(N)[None]) - cap * R[:, None, :]
            res = linprog(-R[-1], A_ub=A.reshape(-1, N), b_ub=np.zeros(R.shape[0] * N), A_eq=np.ones((1, N)),
                          b_eq=[1], bounds=[(0, cap)] * N, method="highs")
            G[a, b], W[(a, b)] = np.log(-res.fun), res.x
        if a % 20 == 0:
            print(f"{a}/{n} dates de départ traitées ({time.time() - t0:.0f} s)", flush=True)
    K = al.MAX_REBAL
    V = np.full((K + 1, n), -np.inf)
    arg = np.full((K + 1, n), -1)
    V[0, 0] = 0
    for m in range(1, K + 1):
        c = V[m - 1][:, None] + G
        c[:, -1] = -np.inf
        arg[m] = np.argmax(c, axis=0)
        V[m] = c[arg[m], np.arange(n)]
        V[m, 0] = -np.inf
    fin = V + G[:, -1][None]
    fin[:, -1] = -np.inf
    bk = fin.max(axis=1)
    mb = int(np.argmax(bk))
    j = int(np.argmax(fin[mb]))
    path = [j]
    for m in range(mb, 0, -1):
        j = int(arg[m, j])
        path.append(j)
    path = path[::-1] + [n - 1]
    rows = []
    for a, b in zip(path[:-1], path[1:]):
        w = pd.Series(W[(a, b)], index=px.columns)
        w = w[w > 1e-6].sort_values(ascending=False)
        rows.append({"debut": nodes[a].date(), "fin": nodes[b].date(), "perf": np.exp(G[a, b]) - 1, "nb_lignes": len(w),
                     "poids": ", ".join(f"{t} {x:.1%}" for t, x in w.items())})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Performance totale : {np.exp(bk[mb]) - 1:.2%} avec {mb} réallocations -> {OUT}")


if __name__ == "__main__":
    main()
