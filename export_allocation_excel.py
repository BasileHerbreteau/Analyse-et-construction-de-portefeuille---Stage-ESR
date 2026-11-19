"""Génère le classeur Excel du track record d'allocation dynamique (partie 3).

    python export_allocation_excel.py            -> resultats/track_record_allocation_dynamique.xlsx

Les poids et les dates viennent de l'optimisation Python (allocation.py). Tout le reste (valeurs liquidatives,
performances par période, contributions, indicateurs) est calculé par des formules Excel : on peut modifier un poids ou
une date (cellules bleues) et le track record se recalcule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.chart.axis import DateAxis
from openpyxl.comments import Comment
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L

import allocation as al

OUT = Path(__file__).parent / "resultats" / "track_record_allocation_dynamique.xlsx"

FONT = "Arial"
NAVY, BLUE_FILL, GREY_FILL = "002855", "DCE9F7", "F2F2F2"
F_TITLE = Font(name=FONT, size=14, bold=True, color=NAVY)
F_H = Font(name=FONT, size=10, bold=True, color="FFFFFF")
F_B = Font(name=FONT, size=10, bold=True)
F_N = Font(name=FONT, size=10)
F_NOTE = Font(name=FONT, size=9, italic=True, color="595959")
F_IN = Font(name=FONT, size=10, color="0000FF")          # saisie
F_LINK = Font(name=FONT, size=10, color="008000")        # lien vers une autre feuille
FILL_H = PatternFill("solid", fgColor=NAVY)
FILL_SUB = PatternFill("solid", fgColor=BLUE_FILL)
FILL_KEY = PatternFill("solid", fgColor="FFFF00")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(top=THIN, bottom=THIN, left=THIN, right=THIN)
PCT, PCT2, DATE, NUM = '0.0%;-0.0%;"-"', "0.00%", "dd/mm/yyyy", "#,##0.00"

N_PER = 1 + al.MAX_REBAL                 # colonnes de poids : départ + 10 réallocations
W_COL0 = 3                               # colonne C
W_ROW0 = 6                               # première ligne de poids


def style_header(ws, row, c0, c1, fill=FILL_H, font=F_H):
    for c in range(c0, c1 + 1):
        cell = ws.cell(row, c)
        cell.fill, cell.font, cell.border = fill, font, BOX
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def note(ws, ref, text):
    ws[ref] = text
    ws[ref].font = F_NOTE


def main():
    px = al.load_prices()
    p = al.period_prices(px)
    tickers = list(px.columns)
    n_t = len(tickers)
    n_d = len(p)
    sol = al.optimise(px)
    static = al.optimise(px, max_rebal=0)
    ex_dates, ex_w = al.load_example()
    ew = al.equal_weight(px)
    fr = al.frontier(px, k_max=20)
    oracle = al.daily_oracle(px)

    wb = Workbook()
    ws_syn = wb.active
    ws_syn.title = "Synthèse"
    ws_px = wb.create_sheet("Prix")
    strat_sheets = {
        "Optimal": ("Allocation optimale", sol.weights, "Portefeuille dynamique optimal (10 réallocations)"),
        "Statique": ("Allocation statique", static.weights,
                     "Meilleur portefeuille sans réallocation (même méthode, 0 réallocation)"),
        "Exemple": ("Exemple du fichier", ex_w, "Allocation d'exemple fournie dans le fichier (onglet « Exmple de "
                                                 "date de rebal »)"),
        "Équipondéré": ("Équipondéré", ew, "Équipondéré sur les 29 sous-jacents, sans réallocation"),
    }
    ws_tr = wb.create_sheet("Track record")
    ws_alloc = {k: wb.create_sheet(v[0]) for k, v in strat_sheets.items()}
    ws_sens = wb.create_sheet("Sensibilité")
    ws_cal = wb.create_sheet("Calcul")
    ws_met = wb.create_sheet("Méthodologie")

    # ------------------------------------------------------------------ Prix
    ws = ws_px
    ws["A1"] = "Prix des sous-jacents du 01/01/2026 au 30/10/2026"
    ws["A1"].font = F_TITLE
    note(ws, "A2", "Source : fichier « Exercice allocation 261113 », onglet « Valeurs des sousjacents ». Jours calendaires ;"
                   " week-ends et jours fériés reprennent le dernier cours connu, comme dans le fichier.")
    ws.cell(3, 1, "Libellé").font = F_NOTE
    ws.cell(4, 1, "Date")
    for j, t in enumerate(tickers):
        ws.cell(3, 2 + j, al.label(t)).font = F_NOTE
        ws.cell(3, 2 + j).alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(4, 2 + j, t)
    style_header(ws, 4, 1, 1 + n_t)
    for i, (d, row) in enumerate(p.iterrows()):
        r = 5 + i
        ws.cell(r, 1, d.to_pydatetime()).number_format = DATE
        ws.cell(r, 1).font = F_N
        for j, v in enumerate(row.values):
            c = ws.cell(r, 2 + j, float(v))
            c.number_format, c.font = NUM, F_N
    ws.row_dimensions[3].height = 48
    ws.column_dimensions["A"].width = 12
    for j in range(n_t):
        ws.column_dimensions[L(2 + j)].width = 12
    ws.freeze_panes = "B5"
    PX_FIRST, PX_LAST = 5, 4 + n_d
    px_last_col = L(1 + n_t)
    PX_DATES = f"Prix!$A${PX_FIRST}:$A${PX_LAST}"
    PX_GRID = f"Prix!$B${PX_FIRST}:${px_last_col}${PX_LAST}"
    PX_HEAD = f"Prix!$B$4:${px_last_col}$4"

    # ------------------------------------------------------------------ Synthèse : paramètres (référencés partout)
    ws = ws_syn
    ws["A1"] = "Track record d'un portefeuille à allocation dynamique, 01/01/2026 – 30/10/2026"
    ws["A1"].font = F_TITLE
    note(ws, "A2", "Exercice « Exercice allocation 261113 » : maximiser la performance totale réalisée avec 20 % maximum par "
                   "sous-jacent et 10 réallocations au plus (allocation de départ non comptée). Optimisation ex post.")
    params = [("Date de départ", al.START.to_pydatetime(), DATE, "Consigne"),
              ("Date de fin", al.END.to_pydatetime(), DATE, "Consigne"),
              ("Poids maximum par sous-jacent", al.CAP, PCT, "Consigne"),
              ("Nombre maximum de réallocations", al.MAX_REBAL, "0", "Consigne (départ non compté)"),
              ("Base de la valeur liquidative", 100, "0", "Convention"),
              ("Jours de cotation par an", al.ANN_DAYS, "0", "Convention d'annualisation de la volatilité"),
              ("Taux sans risque (annuel)", None, PCT2,
               "€STR moyen du 01/01 au 30/10/2026, base 360, annualisé (tableau €STR plus bas)")]
    ws.cell(4, 1, "Paramètres")
    ws.cell(4, 2, "Valeur")
    ws.cell(4, 3, "Origine")
    style_header(ws, 4, 1, 3)
    for i, (k, v, fmt, src) in enumerate(params):
        r = 5 + i
        ws.cell(r, 1, k).font = F_N
        c = ws.cell(r, 2, v)
        c.font, c.number_format = F_IN, fmt
        ws.cell(r, 3, src).font = F_NOTE
    ws["B11"].fill = FILL_KEY
    P_START, P_END, P_CAP, P_MAXR, P_BASE, P_ANN, P_RF = ("Synthèse!$B$5", "Synthèse!$B$6", "Synthèse!$B$7",
                                                          "Synthèse!$B$8", "Synthèse!$B$9", "Synthèse!$B$10",
                                                          "Synthèse!$B$11")

    # ------------------------------------------------------------------ Feuilles d'allocation (même gabarit)
    TR_FIRST, TR_LAST = 6, 5 + n_d
    VL_COL = {"Optimal": "C", "Statique": "D", "Exemple": "E", "Équipondéré": "F"}

    def alloc_sheet(key):
        name, w, desc = strat_sheets[key]
        ws = ws_alloc[key]
        dates = [pd.Timestamp(d) for d in w.columns]
        ws["A1"] = name
        ws["A1"].font = F_TITLE
        src = {"Optimal": "Poids et dates : résultat de l'optimisation Python (allocation.py, programmation dynamique,"
                          " optimum exact). ",
               "Statique": "Poids : résultat de l'optimisation Python sans réallocation (5 meilleurs sous-jacents sur "
                           "toute la période). ",
               "Exemple": "Poids et dates recopiés de l'onglet d'exemple du fichier source. ",
               "Équipondéré": "Poids : 1/29 pour chaque sous-jacent. "}[key]
        note(ws, "A2", desc + ".")
        note(ws, "A3", src + "Cellules bleues modifiables : le track record et les indicateurs se recalculent.")
        ws.cell(4, 1, "Code Bloomberg")
        ws.cell(4, 2, "Sous-jacent")
        for k in range(N_PER):
            ws.cell(4, W_COL0 + k, "Départ" if k == 0 else f"Réallocation {k}")
        ws.cell(4, W_COL0 + N_PER, "Total")
        style_header(ws, 4, 1, W_COL0 + N_PER)
        ws.cell(5, 1, "Date de l'allocation").font = F_B
        for k in range(N_PER):
            c = ws.cell(5, W_COL0 + k)
            if k < len(dates):
                c.value = dates[k].to_pydatetime()
            c.number_format, c.font, c.fill, c.border = DATE, Font(name=FONT, size=10, bold=True, color="0000FF"), \
                FILL_SUB, BOX
        for i, t in enumerate(tickers):
            r = W_ROW0 + i
            ws.cell(r, 1, t).font = F_N
            ws.cell(r, 2, al.label(t)).font = F_N
            for k in range(N_PER):
                c = ws.cell(r, W_COL0 + k)
                if k < len(dates):
                    if key == "Équipondéré":
                        c.value = f"=1/{n_t}"
                        c.font = F_N
                    else:
                        c.value = float(round(w.loc[t, dates[k]], 6))
                        c.font = F_IN
                c.number_format, c.border = PCT, BOX
        last_w = W_ROW0 + n_t - 1
        wl0, wl1 = L(W_COL0), L(W_COL0 + N_PER - 1)
        ws.conditional_formatting.add(f"{wl0}{W_ROW0}:{wl1}{last_w}",
                                      ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF",
                                                     end_type="num", end_value=0.2, end_color="5FB3E4"))
        # Lignes de contrôle et performance par période
        r_tot, r_max, r_nb, r_fin, r_len, r_vl, r_pf, r_ew = range(last_w + 1, last_w + 9)
        r_bh = r_ew + 2                       # bloc : performance de chaque sous-jacent sur la période
        r_bc = r_bh + n_t + 3                 # bloc : contributions
        labels = {r_tot: "Total des poids", r_max: "Poids maximum", r_nb: "Nombre de lignes",
                  r_fin: "Fin de la période", r_len: "Durée (jours)", r_vl: "VL en début de période",
                  r_pf: "Performance du portefeuille sur la période", r_ew: "Performance de l'équipondéré (29)"}
        for r, lab_ in labels.items():
            ws.cell(r, 1, lab_).font = F_B
        for k in range(N_PER):
            col = L(W_COL0 + k)
            nxt = L(W_COL0 + k + 1)
            has = f'{col}$5=""'
            wr = f"{col}{W_ROW0}:{col}{last_w}"
            ws[f"{col}{r_tot}"] = f'=IF({has},"",SUM({wr}))'
            ws[f"{col}{r_max}"] = f'=IF({has},"",MAX({wr}))'
            ws[f"{col}{r_nb}"] = f'=IF({has},"",COUNTIF({wr},">0"))'
            if k < N_PER - 1:
                ws[f"{col}{r_fin}"] = f'=IF({has},"",IF({nxt}$5="",{P_END},{nxt}$5))'
            else:
                ws[f"{col}{r_fin}"] = f'=IF({has},"",{P_END})'
            ws[f"{col}{r_len}"] = f'=IF({has},"",{col}{r_fin}-{col}$5)'
            vl = f"'Track record'!${VL_COL[key]}${TR_FIRST}:${VL_COL[key]}${TR_LAST}"
            ws[f"{col}{r_vl}"] = (f'=IF({has},"",INDEX({vl},MATCH({col}$5,\'Track record\'!$A${TR_FIRST}:'
                                  f'$A${TR_LAST},0)))')
            ws[f"{col}{r_pf}"] = f'=IF({has},"",SUMPRODUCT({wr},{col}{r_bh + 1}:{col}{r_bh + n_t}))'
            ws[f"{col}{r_ew}"] = f'=IF({has},"",AVERAGE({col}{r_bh + 1}:{col}{r_bh + n_t}))'
            for r, fmt in [(r_tot, PCT), (r_max, PCT), (r_nb, "0"), (r_fin, DATE), (r_len, "0"), (r_vl, NUM),
                           (r_pf, PCT), (r_ew, PCT)]:
                ws[f"{col}{r}"].number_format = fmt
                ws[f"{col}{r}"].font = F_N
                ws[f"{col}{r}"].border = BOX
            ws[f"{col}{r_pf}"].font = F_B
        # Totaux de contrôle (colonne Total)
        tc = L(W_COL0 + N_PER)
        for i in range(n_t):
            r = W_ROW0 + i
            ws[f"{tc}{r}"] = f"=COUNTIF({wl0}{r}:{wl1}{r},\">0\")"
            ws[f"{tc}{r}"].number_format = '0" période(s)";;"-"'
            ws[f"{tc}{r}"].font = F_NOTE
        ws.cell(4, W_COL0 + N_PER, "Périodes détenues")
        # Bloc performances des sous-jacents
        ws.cell(r_bh, 1, "Performance de chaque sous-jacent sur la période")
        style_header(ws, r_bh, 1, W_COL0 + N_PER - 1, fill=FILL_SUB, font=F_B)
        for i, t in enumerate(tickers):
            r = r_bh + 1 + i
            ws.cell(r, 1, t).font = F_N
            ws.cell(r, 2, al.label(t)).font = F_N
            for k in range(N_PER):
                col = L(W_COL0 + k)
                pc = f"MATCH($A{r},{PX_HEAD},0)"
                ws[f"{col}{r}"] = (f'=IF({col}$5="","",INDEX({PX_GRID},MATCH({col}${r_fin},{PX_DATES},0),{pc})'
                                   f'/INDEX({PX_GRID},MATCH({col}$5,{PX_DATES},0),{pc})-1)')
                ws[f"{col}{r}"].number_format, ws[f"{col}{r}"].font = PCT, F_N
        # Bloc contributions (points de VL)
        ws.cell(r_bc, 1, "Contribution à la performance (points de VL, base 100)")
        style_header(ws, r_bc, 1, W_COL0 + N_PER, fill=FILL_SUB, font=F_B)
        ws.cell(r_bc, W_COL0 + N_PER, "Total")
        for i, t in enumerate(tickers):
            r = r_bc + 1 + i
            ws.cell(r, 1, t).font = F_N
            ws.cell(r, 2, al.label(t)).font = F_N
            for k in range(N_PER):
                col = L(W_COL0 + k)
                ws[f"{col}{r}"] = f'=IF({col}$5="","",{col}{W_ROW0 + i}*{col}{r_bh + 1 + i}*{col}${r_vl})'
                ws[f"{col}{r}"].number_format, ws[f"{col}{r}"].font = '0.00;-0.00;"-"', F_N
            ws[f"{tc}{r}"] = f"=SUM({wl0}{r}:{wl1}{r})"
            ws[f"{tc}{r}"].number_format, ws[f"{tc}{r}"].font = '0.00;-0.00;"-"', F_B
        r_ct = r_bc + n_t + 1
        ws.cell(r_ct, 1, "Total (= VL finale − base)").font = F_B
        ws[f"{tc}{r_ct}"] = f"=SUM({tc}{r_bc + 1}:{tc}{r_bc + n_t})"
        ws[f"{tc}{r_ct}"].number_format, ws[f"{tc}{r_ct}"].font = NUM, F_B
        ws.cell(r_ct + 1, 1, "Contrôle : VL finale − base").font = F_NOTE
        ws[f"{tc}{r_ct + 1}"] = f"='Track record'!{VL_COL[key]}{TR_LAST}-{P_BASE}"
        ws[f"{tc}{r_ct + 1}"].number_format, ws[f"{tc}{r_ct + 1}"].font = NUM, F_NOTE
        ws.column_dimensions["A"].width = 18
        ws.column_dimensions["B"].width = 42
        for k in range(N_PER + 1):
            ws.column_dimensions[L(W_COL0 + k)].width = 13
        ws.freeze_panes = "C6"
        return {"r_pf": r_pf, "r_max": r_max, "r_tot": r_tot, "r_nb": r_nb, "last_w": last_w}

    rows_info = {k: alloc_sheet(k) for k in strat_sheets}

    # ------------------------------------------------------------------ Calcul : poids transposés (dates en lignes)
    ws = ws_cal
    ws["A1"] = "Calcul : poids transposés (une ligne par allocation), utilisés par le track record. Ne pas modifier."
    ws["A1"].font = F_B
    blocks = {}
    r = 3
    for key, (name, _, _) in strat_sheets.items():
        ws.cell(r, 1, name)
        ws.cell(r, 2, "Date")
        for j, t in enumerate(tickers):
            ws.cell(r, 3 + j, t)
        style_header(ws, r, 1, 2 + n_t, fill=FILL_SUB, font=F_B)
        r0 = r + 1
        for k in range(N_PER):
            rr = r0 + k
            ws.cell(rr, 1, "Départ" if k == 0 else f"Réallocation {k}").font = F_N
            src_date = f"'{name}'!{L(W_COL0 + k)}$5"
            c = ws.cell(rr, 2, f'=IF({src_date}="","",{src_date})')
            c.number_format, c.font = DATE, F_LINK
            for j in range(n_t):
                c = ws.cell(rr, 3 + j, f"=INDEX('{name}'!${L(W_COL0)}${W_ROW0}:${L(W_COL0 + N_PER - 1)}"
                                       f"${W_ROW0 + n_t - 1},{j + 1},{k + 1})")
                c.number_format, c.font = PCT, F_LINK
        blocks[key] = (r0, r0 + N_PER - 1)
        r = r0 + N_PER + 1
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 12
    cal_last_col = L(2 + n_t)

    # ------------------------------------------------------------------ Track record
    ws = ws_tr
    ws["A1"] = "Track record quotidien (valeur liquidative, base 100)"
    ws["A1"].font = F_TITLE
    note(ws, "A2", "Entre deux dates d'allocation, le portefeuille est conservé tel quel (pas de rééquilibrage) : "
                   "VL(t) = VL(date d'allocation) × Σ poids × Prix(t) / Prix(date d'allocation). Performance brute de frais.")
    note(ws, "A3", "Rendements quotidiens calculés seulement les jours de cotation (week-ends et fériés exclus).")
    names = list(strat_sheets)
    groups = [("Valeur liquidative", 3), ("Allocation en cours (n°)", 7), ("Rendement quotidien", 11),
              ("Perte depuis le plus haut", 15)]
    ws.cell(5, 1, "Date")
    ws.cell(5, 2, "Jour de cotation")
    for gname, c0 in groups:
        ws.cell(4, c0, gname)
        ws.merge_cells(start_row=4, start_column=c0, end_row=4, end_column=c0 + 3)
        for j, key in enumerate(names):
            ws.cell(5, c0 + j, key)
    style_header(ws, 4, 3, 18, fill=FILL_SUB, font=F_B)
    style_header(ws, 5, 1, 18)
    ws.row_dimensions[5].height = 30
    for i in range(n_d):
        r = TR_FIRST + i
        pr = PX_FIRST + i                     # même ligne de prix
        c = ws.cell(r, 1, f"=Prix!A{pr}")
        c.number_format, c.font = DATE, F_LINK
        if i == 0:
            ws.cell(r, 2, 0)
        else:
            ws.cell(r, 2, f"=IF(SUMPRODUCT(--(Prix!$B{pr}:${px_last_col}{pr}<>Prix!$B{pr - 1}:${px_last_col}{pr - 1}))>0,1,0)")
        ws.cell(r, 2).font = F_N
        for j, key in enumerate(names):
            b0, b1 = blocks[key]
            vl, kc, rc, dc = L(3 + j), L(7 + j), L(11 + j), L(15 + j)
            cal_dates = f"Calcul!$B${b0}:$B${b1}"
            cal_w = f"Calcul!$C${b0}:${cal_last_col}${b1}"
            if i == 0:
                ws[f"{vl}{r}"] = f"={P_BASE}"
                ws[f"{kc}{r}"] = 1
                ws[f"{rc}{r}"] = ""
            else:
                ws[f"{kc}{r}"] = f'=COUNTIF({cal_dates},"<"&$A{r})'
                start = f"INDEX({cal_dates},{kc}{r})"
                ws[f"{vl}{r}"] = (f"=INDEX({vl}${TR_FIRST}:{vl}{r - 1},MATCH({start},$A${TR_FIRST}:$A{r - 1},0))"
                                  f"*SUMPRODUCT(INDEX({cal_w},{kc}{r},0),Prix!$B{pr}:${px_last_col}{pr}"
                                  f"/INDEX({PX_GRID},MATCH({start},{PX_DATES},0),0))")
                ws[f"{rc}{r}"] = f'=IF($B{r}=1,{vl}{r}/{vl}{r - 1}-1,"")'
            ws[f"{dc}{r}"] = f"={vl}{r}/MAX({vl}${TR_FIRST}:{vl}{r})-1"
            ws[f"{vl}{r}"].number_format = NUM
            ws[f"{kc}{r}"].number_format = "0"
            ws[f"{rc}{r}"].number_format = PCT2
            ws[f"{dc}{r}"].number_format = PCT
            for cc in (vl, kc, rc, dc):
                ws[f"{cc}{r}"].font = F_N
    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 10
    for c in range(3, 19):
        ws.column_dimensions[L(c)].width = 12
    ws.freeze_panes = "C6"

    # ------------------------------------------------------------------ Synthèse : indicateurs
    ws = ws_syn
    r0 = 14
    ws.cell(r0, 1, "Indicateurs")
    for j, key in enumerate(names):
        ws.cell(r0, 2 + j, {"Optimal": "Dynamique optimal", "Statique": "Statique optimal (0 réallocation)",
                            "Exemple": "Exemple du fichier", "Équipondéré": "Équipondéré (29)"}[key])
    style_header(ws, r0, 1, 1 + len(names))
    ws.row_dimensions[r0].height = 32
    kpis = ["VL finale (base 100)", "Performance totale", "Performance annualisée", "Volatilité annualisée",
            "Ratio de Sharpe", "Perte maximale (max drawdown)", "Ratio de Calmar", "Part des jours en hausse",
            "Écart de performance avec l'équipondéré", "Nombre de réallocations", "Poids maximum à une date d'allocation",
            "Contrôle : contraintes respectées"]
    for i, kname in enumerate(kpis):
        ws.cell(r0 + 1 + i, 1, kname).font = F_N
    ew_col = L(2 + names.index("Équipondéré"))
    for j, key in enumerate(names):
        col = L(2 + j)
        vl, rc, dc = L(3 + j), L(11 + j), L(15 + j)
        sh = f"'{strat_sheets[key][0]}'"
        ri = rows_info[key]
        tr = "'Track record'!"
        f = {
            0: f"={tr}{vl}{TR_LAST}",
            1: f"={col}{r0 + 1}/{P_BASE}-1",
            2: f"=(1+{col}{r0 + 2})^(365/({P_END}-{P_START}))-1",
            3: f"=STDEV({tr}{rc}{TR_FIRST}:{rc}{TR_LAST})*SQRT({P_ANN})",
            4: f"=({col}{r0 + 3}-{P_RF})/{col}{r0 + 4}",
            5: f"=MIN({tr}{dc}{TR_FIRST}:{dc}{TR_LAST})",
            6: f"={col}{r0 + 3}/ABS({col}{r0 + 6})",
            7: f'=COUNTIF({tr}{rc}{TR_FIRST}:{rc}{TR_LAST},">0")/COUNT({tr}{rc}{TR_FIRST}:{rc}{TR_LAST})',
            8: f"={col}{r0 + 2}-${ew_col}${r0 + 2}",
            9: f"=COUNT({sh}!{L(W_COL0 + 1)}5:{L(W_COL0 + N_PER - 1)}5)",
            10: f"=MAX({sh}!{L(W_COL0)}{W_ROW0}:{L(W_COL0 + N_PER - 1)}{ri['last_w']})",
            11: (f'=IF(AND({col}{r0 + 10}<={P_MAXR},{col}{r0 + 11}<={P_CAP}+1E-9,'
                 f'ABS(SUM({sh}!{L(W_COL0)}{ri["r_tot"]}:{L(W_COL0 + N_PER - 1)}{ri["r_tot"]})'
                 f'-COUNT({sh}!{L(W_COL0)}5:{L(W_COL0 + N_PER - 1)}5))<1E-6),"Oui","Non")'),
        }
        fmts = [NUM, PCT, PCT, PCT, "0.00", PCT, "0.00", PCT, PCT, "0", PCT, "@"]
        for i in range(len(kpis)):
            c = ws.cell(r0 + 1 + i, 2 + j, f[i])
            c.number_format, c.font, c.border = fmts[i], F_N, BOX
            c.alignment = Alignment(horizontal="right")
        ws.cell(r0 + 2, 2 + j).font = F_B
    for i in range(len(kpis)):
        ws.cell(r0 + 1 + i, 1).border = BOX
    ws.cell(r0 + 2, 1).font = F_B
    rn = r0 + len(kpis) + 2
    note(ws, f"A{rn}", "Volatilité : écart-type des rendements des jours de cotation × √252. Sharpe = (performance "
                       "annualisée − taux sans risque) / volatilité. Calmar = performance annualisée / perte maximale.")
    note(ws, f"A{rn + 1}", "Le portefeuille dynamique est un optimum ex post (connaissance parfaite des prix) : c'est la "
                           "borne haute atteignable sous les contraintes, pas une stratégie investissable.")
    note(ws, f"A{rn + 2}", "Le contrôle vérifie : nombre de réallocations ≤ 10, poids ≤ 20 % à chaque date d'allocation, "
                           "somme des poids = 100 % à chaque date.")

    # Taux sans risque : €STR moyen mensuel de la période
    re0 = rn + 7
    ws.cell(re0, 1, "Taux sans risque : €STR moyen mensuel")
    ws.cell(re0, 2, "€STR moyen")
    ws.cell(re0, 3, "Jours dans la période")
    style_header(ws, re0, 1, 3)
    months = list(al.ESTR_MONTHLY.items())
    for i, (m, v) in enumerate(months):
        r = re0 + 1 + i
        c = ws.cell(r, 1, pd.Timestamp(m + "-01").to_pydatetime())
        c.number_format, c.font = "mmmm yyyy", F_N
        c = ws.cell(r, 2, v / 100)
        c.number_format, c.font = "0.000%", F_IN
        c = ws.cell(r, 3, f"=MIN(EOMONTH(A{r},0)+1,{P_END})-MAX(A{r},{P_START})")
        c.number_format, c.font = "0", F_N
    r1 = re0 + len(months)
    rt = r1 + 1
    ws.cell(rt, 1, "Moyenne de la période (pondérée par les jours)").font = F_B
    c = ws.cell(rt, 2, f"=SUMPRODUCT(B{re0 + 1}:B{r1},C{re0 + 1}:C{r1})/SUM(C{re0 + 1}:C{r1})")
    c.number_format, c.font = "0.000%", F_B
    c = ws.cell(rt, 3, f"=SUM(C{re0 + 1}:C{r1})")
    c.number_format, c.font = "0", F_B
    note(ws, f"A{rt + 1}", f"Source : moyennes mensuelles de l'€STR (série quotidienne de la BCE) relevées le 19/11/2026 sur "
                           f"{al.ESTR_SOURCE} ; paliers conformes aux décisions de la BCE (taux de dépôt 2,00 %, puis "
                           "2,25 % le 17/06 et 2,50 % le 16/09/2026).")
    note(ws, f"A{rt + 2}", "Taux annuel = (1 + Σ €STR × jours / 360)^(365 / jours de la période) − 1 : même annualisation "
                           "que la performance des portefeuilles.")
    ws["B11"] = (f"=(1+SUMPRODUCT(B{re0 + 1}:B{r1},C{re0 + 1}:C{r1})/360)^(365/({P_END}-{P_START}))-1")
    ws["B11"].font, ws["B11"].number_format = F_N, PCT2
    ws.column_dimensions["A"].width = 40
    for c in range(2, 6):
        ws.column_dimensions[L(c)].width = 19

    # Graphique : VL
    ch = LineChart()
    ch.title = "Valeur liquidative (base 100)"
    ch.height, ch.width = 9, 22
    data = Reference(ws_tr, min_col=3, max_col=6, min_row=5, max_row=TR_LAST)
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(Reference(ws_tr, min_col=1, min_row=TR_FIRST, max_row=TR_LAST))
    for s, color in zip(ch.series, ["002855", "00A19A", "5FB3E4", "8A99AD"]):
        s.graphicalProperties.line.solidFill = color
        s.graphicalProperties.line.width = 22000
        s.smooth = False
    ch.x_axis = DateAxis(crossAx=100)
    ch.x_axis.number_format = "mmm-yy"
    ch.x_axis.majorTimeUnit = "months"
    ch.x_axis.title = None
    ch.y_axis.crossAx = 500
    ch.y_axis.title = "VL"
    ch.y_axis.majorGridlines = None
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ch.legend.position = "b"
    ch.legend.overlay = False
    ch.roundedCorners = False
    ws.add_chart(ch, "G4")

    # ------------------------------------------------------------------ Sensibilité
    ws = ws_sens
    ws["A1"] = "Sensibilité : valeur des réallocations, robustesse et frais"
    ws["A1"].font = F_TITLE
    note(ws, "A2", "Valeurs calculées en Python (allocation.py) : l'optimisation des dates n'est pas exprimable en formules "
                   "Excel. Chaque ligne est l'optimum exact pour le nombre de réallocations indiqué.")
    ws.cell(4, 1, "Réallocations autorisées (au plus)")
    ws.cell(4, 2, "Performance totale maximale")
    ws.cell(4, 3, "Gain de la réallocation supplémentaire")
    ws.cell(4, 4, "Dans la consigne")
    style_header(ws, 4, 1, 4)
    ws.row_dimensions[4].height = 30
    for k, v in fr.items():
        r = 5 + k
        ws.cell(r, 1, int(k)).font = F_N
        c = ws.cell(r, 2, float(v))
        c.number_format, c.font = PCT, F_IN
        if k > 0:
            ws.cell(r, 3, f"=B{r}-B{r - 1}").number_format = PCT
            ws.cell(r, 3).font = F_N
        ws.cell(r, 4, "Oui" if k <= al.MAX_REBAL else "Non").font = F_N
        if k == al.MAX_REBAL:
            for c in range(1, 5):
                ws.cell(r, c).fill = FILL_SUB
    r = 5 + len(fr) + 1
    ws.cell(r, 1, "Borne théorique : réallocation chaque jour de cotation (illimitée)").font = F_B
    c = ws.cell(r, 2, oracle)
    c.number_format, c.font = '0%', F_IN
    note(ws, f"C{r}", "Hors consigne : montre que la contrainte de 10 réallocations est très liante.")

    # Robustesse au calendrier de réallocation
    r += 2
    ws.cell(r, 1, "Dates de réallocation autorisées")
    ws.cell(r, 2, "Performance totale (10 réallocations)")
    style_header(ws, r, 1, 2)
    for lab_, rule in [("Jours où toutes les places sont ouvertes (retenu)", "commun"),
                       ("Jours où au moins une place est ouverte", "cotation"),
                       ("Tous les jours calendaires (prix reportés le week-end)", "calendaire")]:
        r += 1
        ws.cell(r, 1, lab_).font = F_N
        c = ws.cell(r, 2, float(al.optimise(px, rule=rule).gross - 1))
        c.number_format, c.font = PCT, F_IN

    # Frais de transaction
    r += 2
    ws.cell(r, 1, "Frais de transaction (pb du montant échangé ; 1 pb = 0,01 %)")
    ws.cell(r, 2, "Performance totale nette")
    ws.cell(r, 3, "Écart de performance (points, base 100)")
    ws.cell(r, 4, "Coût en % de la valeur finale")
    style_header(ws, r, 1, 4)
    r_gross = r + 1
    for bps in [0, 5, 10, 20, 50]:
        r += 1
        nav, _ = al.backtest(px, sol.weights, cost_bps=bps)
        ws.cell(r, 1, bps).font = F_N
        c = ws.cell(r, 2, float(nav.iloc[-1] / 100 - 1))
        c.number_format, c.font = PCT, F_IN
        c = ws.cell(r, 3, f"=($B${r_gross}-B{r})*{P_BASE}")
        c.number_format, c.font = '0.0" pts";-0.0" pts";"-"', F_N
        c = ws.cell(r, 4, f"=1-(1+B{r})/(1+$B${r_gross})")
        c.number_format, c.font = PCT2 + ';-0.00%;"-"', F_N
    note(ws, f"A{r + 1}", "Frais payés à chaque réallocation sur le montant échangé, Σ|nouveau poids − poids dérivé| "
                          "(ventes + achats) ; allocation optimale brute conservée.")
    note(ws, f"A{r + 2}", "Lecture (10 pb) : chaque réallocation échange environ 190 % du portefeuille (≈ 95 % vendus, "
                          "≈ 95 % rachetés), soit ≈ 0,19 % de frais ; sur 10 réallocations, la valeur finale baisse "
                          "d'environ 1,9 %, ce qui fait 5,4 points sur une VL passée de 100 à 289.")
    r += 1

    strict = Path(__file__).parent / "data" / "variante_plafond_quotidien.csv"
    if strict.exists():
        sv = pd.read_csv(strict)
        r += 3
        ws.cell(r, 1, "Variante : plafond de 20 % respecté chaque jour (et pas seulement aux dates d'allocation)")
        style_header(ws, r, 1, 4, fill=FILL_SUB, font=F_B)
        r += 1
        note(ws, f"A{r}", "Calcul séparé (outils/variante_plafond_quotidien.py) : sur chaque période, programme linéaire "
                          "avec wᵢRᵢ(t) ≤ 20 % × ΣⱼwⱼRⱼ(t) chaque jour de cotation, puis même choix des dates.")
        r += 1
        for j, h in enumerate(["Début", "Fin", "Performance de la période", "Poids (lignes détenues)"]):
            ws.cell(r, 1 + j, h).font = F_B
        for _, row in sv.iterrows():
            r += 1
            ws.cell(r, 1, pd.Timestamp(row["debut"]).to_pydatetime()).number_format = DATE
            ws.cell(r, 2, pd.Timestamp(row["fin"]).to_pydatetime()).number_format = DATE
            ws.cell(r, 3, float(row["perf"])).number_format = PCT
            ws.cell(r, 4, str(row["poids"])).font = F_NOTE
        r += 1
        ws.cell(r, 1, "Performance totale de la variante").font = F_B
        c = ws.cell(r, 3, f"=EXP(SUMPRODUCT(LN(1+C{r - len(sv)}:C{r - 1})))-1")
        c.number_format, c.font = PCT, F_B
        rs = rn + 4                               # rappel dans la synthèse
        ws_syn.cell(rs, 1, "Variante : plafond 20 % tenu chaque jour").font = F_B
        c = ws_syn.cell(rs, 2, f"=Sensibilité!C{r}")
        c.number_format, c.font = PCT, F_LINK
        note(ws_syn, f"C{rs}", "Performance totale, 10 réallocations, 6 lignes par période : la dérive des poids ne "
                               "coûte que quelques points.")
    ws.column_dimensions["A"].width = 58
    for c in "BCD":
        ws.column_dimensions[c].width = 22

    bc = BarChart()
    bc.type = "col"
    bc.title = "Performance maximale selon le nombre de réallocations"
    bc.height, bc.width = 8, 18
    bc.add_data(Reference(ws, min_col=2, min_row=4, max_row=4 + len(fr)), titles_from_data=True)
    bc.set_categories(Reference(ws, min_col=1, min_row=5, max_row=4 + len(fr)))
    bc.series[0].graphicalProperties.solidFill = "0062A3"
    bc.series[0].graphicalProperties.line.noFill = True
    bc.y_axis.number_format = "0%"
    bc.y_axis.majorGridlines = None
    bc.x_axis.title = "Nombre maximal de réallocations"
    bc.legend = None
    bc.x_axis.delete = False
    bc.y_axis.delete = False
    bc.roundedCorners = False
    bc.gapWidth = 40
    ws.add_chart(bc, "F4")

    # ------------------------------------------------------------------ Méthodologie
    ws = ws_met
    ws["A1"] = "Méthodologie"
    ws["A1"].font = F_TITLE
    lines = [
        ("Objectif", "Construire, du 01/01/2026 au 30/10/2026, le portefeuille à allocation dynamique qui maximise la "
                     "performance totale réalisée, avec 20 % maximum par sous-jacent et au plus 10 réallocations "
                     "(l'allocation de départ ne compte pas)."),
        ("Univers", f"{n_t} sous-jacents : 23 indices sectoriels STOXX (Europe 600 et Euro Stoxx), le Russell 2000, un "
                    "indice sectoriel américain et 3 ETF. Prix quotidiens du fichier, en devise de cotation."),
        ("Hypothèse de gestion", "Entre deux dates d'allocation, le portefeuille est conservé tel quel (« buy and hold ») : "
                                 "les poids dérivent avec les prix. Le plafond de 20 % s'applique à chaque date "
                                 "d'allocation, comme dans les règles d'indices."),
        ("Étape 1 : poids", "Sur une période donnée, la valeur finale vaut Σ wᵢ × Pᵢ(fin)/Pᵢ(début) : elle est linéaire "
                            "dans les poids. Avec 0 ≤ wᵢ ≤ 20 % et Σ wᵢ = 100 %, le maximum est atteint en mettant 20 % "
                            "sur les 5 sous-jacents qui montent le plus sur la période (vérifié par programmation linéaire)."),
        ("Étape 2 : dates", "La performance totale est le produit des performances des périodes. On cherche donc les dates "
                            "qui maximisent ce produit : programmation dynamique sur (date, nombre de réallocations déjà "
                            "faites). L'algorithme examine toutes les combinaisons de dates implicitement et donne "
                            "l'optimum global, vérifié par recherche exhaustive pour 1 et 2 réallocations."),
        ("Dates possibles", "Réallocation permise les jours où toutes les places (Europe et États-Unis) sont ouvertes, "
                            "pour ne jamais trader sur un prix figé. Le résultat est identique si l'on autorise tous les "
                            "jours (feuille Sensibilité)."),
        ("Portefeuilles de comparaison", "Statique optimal : meilleur portefeuille sans réallocation (mêmes règles). "
                                         "Exemple du fichier : allocation fournie à titre d'exemple. Équipondéré : 1/29 "
                                         "sur chaque sous-jacent, sans réallocation."),
        ("Taux sans risque", "€STR moyen du 01/01 au 30/10/2026 (2,08 % en moyenne, 2,11 % en taux annuel équivalent), "
                             "taux de référence du marché monétaire en euros, cohérent avec un univers majoritairement "
                             "européen. Il sert au ratio de Sharpe."),
        ("Lecture", "Le résultat est un optimum ex post, construit avec la connaissance des prix futurs : il mesure ce "
                    "qu'il était possible de gagner sous les contraintes, pas ce qu'une stratégie aurait gagné en temps "
                    "réel."),
        ("Limites", "Indices de prix (dividendes non inclus) ; devises de cotation mélangées sans conversion (3 "
                    "sous-jacents en dollars) ; frais de transaction non déduits dans le track record (impact chiffré "
                    "dans Sensibilité) ; liquidité et exécution au cours de clôture supposées."),
        ("Recalcul", "Les cellules bleues (dates et poids des feuilles d'allocation, paramètres) sont modifiables : "
                     "les VL, indicateurs, performances par période et contributions se recalculent."),
    ]
    for i, (h, t) in enumerate(lines):
        r = 3 + i
        ws.cell(r, 1, h).font = F_B
        ws.cell(r, 2, t).font = F_N
        ws.cell(r, 2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(r, 1).alignment = Alignment(vertical="top")
        ws.row_dimensions[r].height = 15 * (1 + len(t) // 105)
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 110

    for w_ in wb.worksheets:
        w_.sheet_view.showGridLines = False
        w_.page_setup.orientation = "landscape"
        w_.page_setup.fitToWidth = 1
        w_.page_setup.fitToHeight = 0
        w_.sheet_properties.pageSetUpPr.fitToPage = True
    OUT.parent.mkdir(exist_ok=True)
    wb.save(OUT)
    print(OUT)


if __name__ == "__main__":
    sys.exit(main())
