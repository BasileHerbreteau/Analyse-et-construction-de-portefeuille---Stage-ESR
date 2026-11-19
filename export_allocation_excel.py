"""Classeur Excel du track record de portefeuille dynamique (onglet « Bonus - Track record » de l'application).

    python export_allocation_excel.py      -> resultats/track_record_allocation_dynamique.xlsx

Les dates et les poids viennent de l'optimisation Python (allocation.py). Valeurs liquidatives, performances par
période, contributions et indicateurs sont calculés par formules Excel : modifier une date ou un poids (en bleu)
recalcule le track record. Charte graphique Amundi ; logo : assets/logo_amundi.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.chart.axis import DateAxis
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L

import allocation as al

ROOT = Path(__file__).parent
OUT = ROOT / "resultats" / "track_record_allocation_dynamique.xlsx"
LOGO = ROOT / "assets" / "logo_amundi.png"

# Charte Amundi (mêmes couleurs que l'application)
NAVY, BLUE, LIGHT, TEAL, PALE, GREY = "002855", "0062A3", "5FB3E4", "00A19A", "A9CBEA", "8A99AD"
ICE, ZEBRA, LINE, INK, MUTED = "DCE9F7", "F3F7FC", "C9DBF0", "0B1F3A", "5B6B80"
FONT = "Arial"
F_TITLE = Font(name=FONT, size=16, bold=True, color=NAVY)
F_SUBT = Font(name=FONT, size=10, color=MUTED)
F_H = Font(name=FONT, size=10, bold=True, color="FFFFFF")
F_SUBH = Font(name=FONT, size=10, bold=True, color=NAVY)
F_B = Font(name=FONT, size=10, bold=True, color=INK)
F_N = Font(name=FONT, size=10, color=INK)
F_SRC = Font(name=FONT, size=9, color=MUTED)
F_NOTE = Font(name=FONT, size=8, italic=True, color=MUTED)
F_IN = Font(name=FONT, size=10, color="0000FF")          # donnée saisie
F_IN_B = Font(name=FONT, size=10, bold=True, color="0000FF")
F_LINK = Font(name=FONT, size=10, color="008000")        # lien vers une autre feuille
FILL_H = PatternFill("solid", fgColor=NAVY)
FILL_SUB = PatternFill("solid", fgColor=ICE)
FILL_Z = PatternFill("solid", fgColor=ZEBRA)
S_LINE = Side(style="thin", color=LINE)
S_RULE = Side(style="medium", color=NAVY)
S_ACCENT = Side(style="thin", color=LIGHT)
BOX = Border(top=S_LINE, bottom=S_LINE, left=S_LINE, right=S_LINE)
PCT, PCT2, DATE, NUM = '0.0%;-0.0%;"-"', '0.00%;-0.00%;"-"', "dd/mm/yyyy", "#,##0.00"

N_PER = 1 + al.MAX_REBAL                 # colonnes de poids : départ + 10 réallocations
W_COL0, W_ROW0 = 3, 6                    # première cellule de poids : C6
TAB = {"Synthèse": NAVY, "Data": GREY, "Track record": BLUE, "Allocation optimale": "2F86C6",
       "Sensibilité": LIGHT, "Allocation statique": PALE, "Équipondéré": ICE}


def header(ws, row, c0, c1, fill=FILL_H, font=F_H, height=None):
    for c in range(c0, c1 + 1):
        x = ws.cell(row, c)
        x.fill, x.font, x.border = fill, font, BOX
        x.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    if height:
        ws.row_dimensions[row].height = height


def title(ws, text, subtitle, last_col, logo_col=None):
    """Bandeau : titre, sous-titre, filet bleu marine ; logo Amundi à droite."""
    ws["A1"], ws["A2"] = text, subtitle
    ws["A1"].font, ws["A2"].font = F_TITLE, F_SUBT
    ws["A1"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 34
    for c in range(1, last_col + 1):
        ws.cell(2, c).border = Border(bottom=S_RULE)
    if LOGO.exists() and logo_col:
        img = XLImage(str(LOGO))
        h = 40
        img.width, img.height = int(img.width * h / img.height), h
        ws.add_image(img, f"{logo_col}1")


def section(ws, row):
    """Intitulé de bloc sur les colonnes A:B, aligné à gauche."""
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
    ws.cell(row, 1).alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[row].height = 20


def note(ws, ref, text):
    ws[ref] = text
    ws[ref].font = F_NOTE


def cell(ws, r, c, value, fmt=None, font=F_N, border=True, fill=None, align=None):
    x = ws.cell(r, c, value)
    x.font = font
    if fmt:
        x.number_format = fmt
    if border:
        x.border = BOX
    if fill:
        x.fill = fill
    if align:
        x.alignment = Alignment(horizontal=align, vertical="center")
    return x


def main():
    px = al.load_prices()
    p = al.period_prices(px)
    tickers = list(px.columns)
    n_t, n_d = len(tickers), len(p)
    sol = al.optimise(px)
    static = al.optimise(px, max_rebal=0)
    fr = al.frontier(px, k_max=20)
    oracle = al.daily_oracle(px)

    strat = {   # clé : (feuille, poids, sous-titre, intitulé de colonne)
        "Optimal": ("Allocation optimale", sol.weights,
                    "10 réallocations · 5 lignes à 20 % sur chaque période · optimum ex post", "Dynamique optimal"),
        "Statique": ("Allocation statique", static.weights,
                     "Sans réallocation · 5 meilleurs sous-jacents de la période à 20 %",
                     "Statique optimal (0 réallocation)"),
        "Équipondéré": ("Équipondéré", al.equal_weight(px), "1/29 par sous-jacent · sans réallocation",
                        "Équipondéré (29)"),
    }
    names = list(strat)
    n_s = len(names)

    wb = Workbook()
    ws_syn = wb.active
    ws_syn.title = "Synthèse"
    ws_px = wb.create_sheet("Data")
    ws_tr = wb.create_sheet("Track record")
    ws_alloc = {"Optimal": wb.create_sheet("Allocation optimale")}
    ws_sens = wb.create_sheet("Sensibilité")
    ws_alloc["Statique"] = wb.create_sheet("Allocation statique")
    ws_alloc["Équipondéré"] = wb.create_sheet("Équipondéré")
    ws_cal = wb.create_sheet("Calcul")

    # ------------------------------------------------------------------ Data
    ws = ws_px
    title(ws, "Cours des sous-jacents", "Source : fichier « Exercice allocation 261113 » (codes Bloomberg) · cours de "
          "clôture en devise de cotation · week-ends et jours fériés : dernier cours connu", 1 + n_t, "M")
    ws.cell(3, 1, "Libellé").font = F_NOTE
    ws.cell(4, 1, "Date")
    for j, t in enumerate(tickers):
        ws.cell(3, 2 + j, al.label(t)).font = F_NOTE
        ws.cell(3, 2 + j).alignment = Alignment(wrap_text=True, vertical="bottom")
        ws.cell(4, 2 + j, t)
    header(ws, 4, 1, 1 + n_t, height=30)
    for i, (d, row) in enumerate(p.iterrows()):
        r = 5 + i
        fill = FILL_Z if i % 2 else None
        cell(ws, r, 1, d.to_pydatetime(), DATE, border=False, fill=fill)
        for j, v in enumerate(row.values):
            cell(ws, r, 2 + j, float(v), NUM, border=False, fill=fill)
    ws.row_dimensions[3].height = 52
    ws.column_dimensions["A"].width = 12
    for j in range(n_t):
        ws.column_dimensions[L(2 + j)].width = 12
    ws.freeze_panes = "B5"
    PX_FIRST, PX_LAST = 5, 4 + n_d
    pl = L(1 + n_t)
    PX_DATES = f"Data!$A${PX_FIRST}:$A${PX_LAST}"
    PX_GRID = f"Data!$B${PX_FIRST}:${pl}${PX_LAST}"
    PX_HEAD = f"Data!$B$4:${pl}$4"

    # ------------------------------------------------------------------ Synthèse : paramètres
    ws = ws_syn
    title(ws, "Track record · Portefeuille à allocation dynamique",
          "Du 01/01/2026 au 30/10/2026 · 29 sous-jacents · 20 % maximum par ligne · 10 réallocations au plus", 4, "D")
    params = [("Date de départ", al.START.to_pydatetime(), DATE, "Consigne"),
              ("Date de fin", al.END.to_pydatetime(), DATE, "Consigne"),
              ("Poids maximum par sous-jacent", al.CAP, PCT, "Consigne"),
              ("Nombre maximum de réallocations", al.MAX_REBAL, "0", "Consigne, allocation initiale non comptée"),
              ("Base de la valeur liquidative", 100, "0", "Convention"),
              ("Jours de cotation par an", al.ANN_DAYS, "0", "Annualisation de la volatilité"),
              ("Taux sans risque annuel", None, PCT2, "€STR moyen de la période")]
    for c, h in enumerate(["Paramètres", "Valeur", "Source"], start=1):
        ws.cell(4, c, h)
    header(ws, 4, 1, 3)
    for i, (k, v, fmt, src) in enumerate(params):
        r = 5 + i
        z = FILL_Z if i % 2 else None
        cell(ws, r, 1, k, fill=z)
        cell(ws, r, 2, v, fmt, font=F_IN, fill=z, align="right")
        cell(ws, r, 3, src, font=F_SRC, fill=z)
    P_START, P_END, P_CAP, P_MAXR, P_BASE, P_ANN, P_RF = (f"Synthèse!$B${r}" for r in range(5, 12))

    # ------------------------------------------------------------------ Feuilles d'allocation
    TR_FIRST, TR_LAST = 6, 5 + n_d
    VL_COL = {k: L(3 + j) for j, k in enumerate(names)}

    def alloc_sheet(key):
        name, w, subt, _ = strat[key]
        ws = ws_alloc[key]
        dates = [pd.Timestamp(d) for d in w.columns]
        last = W_COL0 + N_PER
        title(ws, name, subt, last, L(last - 1))
        ws.cell(4, 1, "Code Bloomberg")
        ws.cell(4, 2, "Sous-jacent")
        for k in range(N_PER):
            ws.cell(4, W_COL0 + k, "Départ" if k == 0 else f"Réallocation {k}")
        ws.cell(4, last, "Périodes détenues")
        header(ws, 4, 1, last, height=30)
        cell(ws, 5, 1, "Date d'allocation", font=F_SUBH, fill=FILL_SUB)
        cell(ws, 5, 2, None, fill=FILL_SUB)
        for k in range(N_PER):
            cell(ws, 5, W_COL0 + k, dates[k].to_pydatetime() if k < len(dates) else None, DATE, font=F_IN_B,
                 fill=FILL_SUB, align="center")
        cell(ws, 5, last, None, fill=FILL_SUB)
        for i, t in enumerate(tickers):
            r = W_ROW0 + i
            cell(ws, r, 1, t)
            cell(ws, r, 2, al.label(t))
            for k in range(N_PER):
                v = None
                if k < len(dates):
                    v = f"=1/{n_t}" if key == "Équipondéré" else float(round(w.loc[t, dates[k]], 6))
                cell(ws, r, W_COL0 + k, v, PCT, font=F_N if key == "Équipondéré" else F_IN, align="center")
        last_w = W_ROW0 + n_t - 1
        wl0, wl1 = L(W_COL0), L(W_COL0 + N_PER - 1)
        ws.conditional_formatting.add(f"{wl0}{W_ROW0}:{wl1}{last_w}",
                                      ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF",
                                                     end_type="num", end_value=0.2, end_color=LIGHT))
        r_tot, r_max, r_nb, r_fin, r_len, r_vl, r_pf, r_ew = range(last_w + 1, last_w + 9)
        r_bh = r_ew + 2
        r_bc = r_bh + n_t + 3
        labels = {r_tot: "Total des poids", r_max: "Poids maximum", r_nb: "Nombre de lignes",
                  r_fin: "Fin de période", r_len: "Durée (jours)", r_vl: "VL en début de période",
                  r_pf: "Performance du portefeuille", r_ew: "Performance de l'équipondéré (29)"}
        for r, lab_ in labels.items():
            key_row = r == r_pf
            cell(ws, r, 1, lab_, font=F_B if key_row else F_N, fill=FILL_SUB if key_row else None)
            cell(ws, r, 2, None, fill=FILL_SUB if key_row else None)
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
        for k in range(N_PER):
            col, nxt = L(W_COL0 + k), L(W_COL0 + k + 1)
            has = f'{col}$5=""'
            wr = f"{col}{W_ROW0}:{col}{last_w}"
            vl = f"'Track record'!${VL_COL[key]}${TR_FIRST}:${VL_COL[key]}${TR_LAST}"
            f = {r_tot: f'=IF({has},"",SUM({wr}))', r_max: f'=IF({has},"",MAX({wr}))',
                 r_nb: f'=IF({has},"",COUNTIF({wr},">0"))',
                 r_fin: (f'=IF({has},"",IF({nxt}$5="",{P_END},{nxt}$5))' if k < N_PER - 1
                         else f'=IF({has},"",{P_END})'),
                 r_len: f'=IF({has},"",{col}{r_fin}-{col}$5)',
                 r_vl: f"=IF({has},\"\",INDEX({vl},MATCH({col}$5,'Track record'!$A${TR_FIRST}:$A${TR_LAST},0)))",
                 r_pf: f'=IF({has},"",SUMPRODUCT({wr},{col}{r_bh + 1}:{col}{r_bh + n_t}))',
                 r_ew: f'=IF({has},"",AVERAGE({col}{r_bh + 1}:{col}{r_bh + n_t}))'}
            fmts = {r_tot: PCT, r_max: PCT, r_nb: "0", r_fin: DATE, r_len: "0", r_vl: NUM, r_pf: PCT, r_ew: PCT}
            for r in f:
                cell(ws, r, W_COL0 + k, f[r], fmts[r], font=F_B if r == r_pf else F_N,
                     fill=FILL_SUB if r == r_pf else None, align="center")
        tc = L(last)
        for i in range(n_t):
            r = W_ROW0 + i
            cell(ws, r, last, f'=COUNTIF({wl0}{r}:{wl1}{r},">0")', '0;;"-"', align="center")
        # performances des sous-jacents par période
        ws.cell(r_bh, 1, "Performance de chaque sous-jacent sur la période")
        header(ws, r_bh, 1, last - 1, fill=FILL_SUB, font=F_SUBH)
        section(ws, r_bh)
        for i, t in enumerate(tickers):
            r = r_bh + 1 + i
            cell(ws, r, 1, t)
            cell(ws, r, 2, al.label(t))
            pc = f"MATCH($A{r},{PX_HEAD},0)"
            for k in range(N_PER):
                col = L(W_COL0 + k)
                cell(ws, r, W_COL0 + k,
                     f'=IF({col}$5="","",INDEX({PX_GRID},MATCH({col}${r_fin},{PX_DATES},0),{pc})'
                     f'/INDEX({PX_GRID},MATCH({col}$5,{PX_DATES},0),{pc})-1)', PCT, align="center")
        # contributions
        ws.cell(r_bc, 1, "Contribution à la performance (points de VL, base 100)")
        ws.cell(r_bc, last, "Total")
        header(ws, r_bc, 1, last, fill=FILL_SUB, font=F_SUBH)
        section(ws, r_bc)
        for i, t in enumerate(tickers):
            r = r_bc + 1 + i
            cell(ws, r, 1, t)
            cell(ws, r, 2, al.label(t))
            for k in range(N_PER):
                col = L(W_COL0 + k)
                cell(ws, r, W_COL0 + k, f'=IF({col}$5="","",{col}{W_ROW0 + i}*{col}{r_bh + 1 + i}*{col}${r_vl})',
                     '0.00;-0.00;"-"', align="center")
            cell(ws, r, last, f"=SUM({wl0}{r}:{wl1}{r})", '0.00;-0.00;"-"', font=F_B, align="center")
        r_ct = r_bc + n_t + 1
        cell(ws, r_ct, 1, "Total", font=F_B, fill=FILL_SUB)
        cell(ws, r_ct, 2, None, fill=FILL_SUB)
        ws.merge_cells(start_row=r_ct, start_column=1, end_row=r_ct, end_column=2)
        cell(ws, r_ct, last, f"=SUM({tc}{r_bc + 1}:{tc}{r_bc + n_t})", NUM, font=F_B, fill=FILL_SUB, align="center")
        cell(ws, r_ct + 1, 1, "Contrôle : VL finale − base", font=F_NOTE, border=False)
        cell(ws, r_ct + 1, last, f"='Track record'!{VL_COL[key]}{TR_LAST}-{P_BASE}", NUM, font=F_NOTE, border=False,
             align="center")
        ws.column_dimensions["A"].width = 18
        ws.column_dimensions["B"].width = 44
        for k in range(N_PER + 1):
            ws.column_dimensions[L(W_COL0 + k)].width = 13
        ws.freeze_panes = "C6"
        return {"r_tot": r_tot, "last_w": last_w}

    rows_info = {k: alloc_sheet(k) for k in names}

    # ------------------------------------------------------------------ Calcul (masquée) : poids transposés
    ws = ws_cal
    ws["A1"] = "Poids transposés (une ligne par allocation), utilisés par le track record"
    ws["A1"].font = F_B
    blocks, r = {}, 3
    cal_last = L(2 + n_t)
    for key in names:
        name = strat[key][0]
        ws.cell(r, 1, name)
        ws.cell(r, 2, "Date")
        for j, t in enumerate(tickers):
            ws.cell(r, 3 + j, t)
        header(ws, r, 1, 2 + n_t, fill=FILL_SUB, font=F_SUBH)
        r0 = r + 1
        for k in range(N_PER):
            rr = r0 + k
            ws.cell(rr, 1, "Départ" if k == 0 else f"Réallocation {k}").font = F_N
            src = f"'{name}'!{L(W_COL0 + k)}$5"
            c = ws.cell(rr, 2, f'=IF({src}="","",{src})')
            c.number_format, c.font = DATE, F_LINK
            for j in range(n_t):
                c = ws.cell(rr, 3 + j, f"=INDEX('{name}'!${L(W_COL0)}${W_ROW0}:${L(W_COL0 + N_PER - 1)}"
                                       f"${W_ROW0 + n_t - 1},{j + 1},{k + 1})")
                c.number_format, c.font = PCT, F_LINK
        blocks[key] = (r0, r0 + N_PER - 1)
        r = r0 + N_PER + 1
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 12
    ws.sheet_state = "hidden"

    # ------------------------------------------------------------------ Track record
    ws = ws_tr
    g = [("Valeur liquidative", 3), ("Allocation en cours (n°)", 3 + n_s), ("Rendement quotidien", 3 + 2 * n_s),
         ("Perte depuis le plus haut", 3 + 3 * n_s)]
    last = 2 + 4 * n_s
    title(ws, "Track record quotidien", "VL base 100 · positions conservées entre deux dates d'allocation · "
          "performance brute de frais", last, L(last - 1))
    ws.cell(5, 1, "Date")
    ws.cell(5, 2, "Jour de cotation")
    for gname, c0 in g:
        ws.cell(4, c0, gname)
        ws.merge_cells(start_row=4, start_column=c0, end_row=4, end_column=c0 + n_s - 1)
        for j, key in enumerate(names):
            ws.cell(5, c0 + j, key)
    header(ws, 4, 3, last, fill=FILL_SUB, font=F_SUBH)
    header(ws, 5, 1, last, height=30)
    for i in range(n_d):
        r, pr = TR_FIRST + i, PX_FIRST + i
        fill = FILL_Z if i % 2 else None
        cell(ws, r, 1, f"=Data!A{pr}", DATE, font=F_LINK, border=False, fill=fill)
        cell(ws, r, 2, 0 if i == 0 else
             f"=IF(SUMPRODUCT(--(Data!$B{pr}:${pl}{pr}<>Data!$B{pr - 1}:${pl}{pr - 1}))>0,1,0)", "0",
             border=False, fill=fill, align="center")
        for j, key in enumerate(names):
            b0, b1 = blocks[key]
            vl, kc = L(3 + j), L(3 + n_s + j)
            cal_d = f"Calcul!$B${b0}:$B${b1}"
            cal_w = f"Calcul!$C${b0}:${cal_last}${b1}"
            if i == 0:
                fv, fk, fr_ = f"={P_BASE}", 1, None
            else:
                fk = f'=COUNTIF({cal_d},"<"&$A{r})'
                start = f"INDEX({cal_d},{kc}{r})"
                fv = (f"=INDEX({vl}${TR_FIRST}:{vl}{r - 1},MATCH({start},$A${TR_FIRST}:$A{r - 1},0))"
                      f"*SUMPRODUCT(INDEX({cal_w},{kc}{r},0),Data!$B{pr}:${pl}{pr}"
                      f"/INDEX({PX_GRID},MATCH({start},{PX_DATES},0),0))")
                fr_ = f'=IF($B{r}=1,{vl}{r}/{vl}{r - 1}-1,"")'
            cell(ws, r, 3 + j, fv, NUM, border=False, fill=fill)
            cell(ws, r, 3 + n_s + j, fk, "0", border=False, fill=fill, align="center")
            cell(ws, r, 3 + 2 * n_s + j, fr_, PCT2, border=False, fill=fill)
            cell(ws, r, 3 + 3 * n_s + j, f"={vl}{r}/MAX({vl}${TR_FIRST}:{vl}{r})-1", PCT, border=False, fill=fill)
    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 11
    for c in range(3, last + 1):
        ws.column_dimensions[L(c)].width = 13
    ws.freeze_panes = "C6"

    # ------------------------------------------------------------------ Synthèse : indicateurs
    ws = ws_syn
    r0 = 14
    ws.cell(r0, 1, "Indicateurs")
    for j, key in enumerate(names):
        ws.cell(r0, 2 + j, strat[key][3])
    header(ws, r0, 1, 1 + n_s, height=32)
    kpis = ["VL finale (base 100)", "Performance totale", "Performance annualisée", "Volatilité annualisée",
            "Ratio de Sharpe", "Perte maximale", "Ratio de Calmar", "Part des jours en hausse",
            "Écart de performance avec l'équipondéré", "Nombre de réallocations",
            "Poids maximum à une date d'allocation", "Contraintes respectées"]
    fmts = [NUM, PCT, PCT, PCT, "0.00", PCT, "0.00", PCT, PCT, "0", PCT, "@"]
    ew_col = L(2 + names.index("Équipondéré"))
    tr = "'Track record'!"
    for i, kname in enumerate(kpis):
        key_row = kname == "Performance totale"
        z = FILL_SUB if key_row else (FILL_Z if i % 2 else None)
        cell(ws, r0 + 1 + i, 1, kname, font=F_B if key_row else F_N, fill=z)
        for j, key in enumerate(names):
            col = L(2 + j)
            vl, rc, dc = L(3 + j), L(3 + 2 * n_s + j), L(3 + 3 * n_s + j)
            sh = f"'{strat[key][0]}'"
            ri = rows_info[key]
            f = [f"={tr}{vl}{TR_LAST}",
                 f"={col}{r0 + 1}/{P_BASE}-1",
                 f"=(1+{col}{r0 + 2})^(365/({P_END}-{P_START}))-1",
                 f"=STDEV({tr}{rc}{TR_FIRST}:{rc}{TR_LAST})*SQRT({P_ANN})",
                 f"=({col}{r0 + 3}-{P_RF})/{col}{r0 + 4}",
                 f"=MIN({tr}{dc}{TR_FIRST}:{dc}{TR_LAST})",
                 f"={col}{r0 + 3}/ABS({col}{r0 + 6})",
                 f'=COUNTIF({tr}{rc}{TR_FIRST}:{rc}{TR_LAST},">0")/COUNT({tr}{rc}{TR_FIRST}:{rc}{TR_LAST})',
                 f"={col}{r0 + 2}-${ew_col}${r0 + 2}",
                 f"=COUNT({sh}!{L(W_COL0 + 1)}5:{L(W_COL0 + N_PER - 1)}5)",
                 f"=MAX({sh}!{L(W_COL0)}{W_ROW0}:{L(W_COL0 + N_PER - 1)}{ri['last_w']})",
                 (f'=IF(AND({col}{r0 + 10}<={P_MAXR},{col}{r0 + 11}<={P_CAP}+1E-9,'
                  f'ABS(SUM({sh}!{L(W_COL0)}{ri["r_tot"]}:{L(W_COL0 + N_PER - 1)}{ri["r_tot"]})'
                  f'-COUNT({sh}!{L(W_COL0)}5:{L(W_COL0 + N_PER - 1)}5))<1E-6),"Oui","Non")')][i]
            cell(ws, r0 + 1 + i, 2 + j, f, fmts[i], font=F_B if key_row else F_N, fill=z, align="right")
    rn = r0 + len(kpis) + 2
    note(ws, f"A{rn}", "Simulation ex post : allocation optimisée avec la connaissance des cours de la période. "
                       "Indices de prix en devise de cotation, performances brutes de frais.")
    note(ws, f"A{rn + 1}", "Volatilité : rendements des jours de cotation, annualisés (√252). Sharpe : (performance "
                           "annualisée − €STR) / volatilité. Calmar : performance annualisée / perte maximale.")
    note(ws, f"A{rn + 2}", "En bleu : données saisies, modifiables (paramètres, dates et poids) ; en noir : calculs.")

    # €STR moyen mensuel -> taux sans risque
    re0 = rn + 5
    for c, h in enumerate(["€STR (moyenne mensuelle)", "Taux", "Jours"], start=1):
        ws.cell(re0, c, h)
    header(ws, re0, 1, 3)
    months = list(al.ESTR_MONTHLY.items())
    for i, (m, v) in enumerate(months):
        r = re0 + 1 + i
        z = FILL_Z if i % 2 else None
        cell(ws, r, 1, pd.Timestamp(m + "-01").to_pydatetime(), '[$-40C]mmmm yyyy', fill=z, align="left")
        cell(ws, r, 2, v / 100, "0.000%", font=F_IN, fill=z, align="right")
        cell(ws, r, 3, f"=MIN(EOMONTH(A{r},0)+1,{P_END})-MAX(A{r},{P_START})", "0", fill=z, align="right")
    r1 = re0 + len(months)
    rt = r1 + 1
    cell(ws, rt, 1, "Moyenne de la période", font=F_B, fill=FILL_SUB)
    cell(ws, rt, 2, f"=SUMPRODUCT(B{re0 + 1}:B{r1},C{re0 + 1}:C{r1})/SUM(C{re0 + 1}:C{r1})", "0.000%", font=F_B,
         fill=FILL_SUB, align="right")
    cell(ws, rt, 3, f"=SUM(C{re0 + 1}:C{r1})", "0", font=F_B, fill=FILL_SUB, align="right")
    note(ws, f"A{rt + 1}", f"Source : €STR quotidien publié par la BCE, moyennes mensuelles ({al.ESTR_SOURCE}). "
                           "Taux annuel = (1 + Σ taux × jours / 360)^(365 / jours) − 1.")
    ws["B11"] = f"=(1+SUMPRODUCT(B{re0 + 1}:B{r1},C{re0 + 1}:C{r1})/360)^(365/({P_END}-{P_START}))-1"
    ws["B11"].font = F_N
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 19
    ws.column_dimensions["C"].width = 30
    ws.column_dimensions["D"].width = 19

    ch = LineChart()
    ch.title = "Valeur liquidative (base 100)"
    ch.height, ch.width = 9.5, 22
    ch.add_data(Reference(ws_tr, min_col=3, max_col=2 + n_s, min_row=5, max_row=TR_LAST), titles_from_data=True)
    ch.set_categories(Reference(ws_tr, min_col=1, min_row=TR_FIRST, max_row=TR_LAST))
    for s, color in zip(ch.series, [NAVY, TEAL, GREY]):
        s.graphicalProperties.line.solidFill = color
        s.graphicalProperties.line.width = 22000
        s.smooth = False
    ch.x_axis = DateAxis(crossAx=100)
    ch.x_axis.number_format = '[$-40C]mmm yy'
    ch.x_axis.baseTimeUnit = "days"
    ch.x_axis.majorUnit = 1
    ch.x_axis.majorTimeUnit = "months"
    ch.y_axis.number_format = "0"
    ch.y_axis.scaling.min = 50
    ch.x_axis.title = None
    ch.y_axis.crossAx = 500
    ch.y_axis.majorGridlines = None
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ch.legend.position = "b"
    ch.legend.overlay = False
    ch.roundedCorners = False
    ws.add_chart(ch, "F4")

    # ------------------------------------------------------------------ Sensibilité
    ws = ws_sens
    title(ws, "Sensibilités", "Optimum exact pour chaque nombre de réallocations (programmation dynamique, Python) · "
          "frais de transaction", 4, "D")
    for c, h in enumerate(["Réallocations autorisées (au plus)", "Performance totale maximale",
                           "Gain marginal", "Dans la consigne"], start=1):
        ws.cell(4, c, h)
    header(ws, 4, 1, 4, height=30)
    for k, v in fr.items():
        r = 5 + k
        z = FILL_SUB if k == al.MAX_REBAL else (FILL_Z if k % 2 else None)
        fb = F_B if k == al.MAX_REBAL else F_N
        cell(ws, r, 1, int(k), "0", font=fb, fill=z, align="center")
        cell(ws, r, 2, float(v), PCT, font=F_IN_B if k == al.MAX_REBAL else F_IN, fill=z, align="right")
        cell(ws, r, 3, f"=B{r}-B{r - 1}" if k > 0 else None, PCT, font=fb, fill=z, align="right")
        cell(ws, r, 4, "Oui" if k <= al.MAX_REBAL else "Non", font=fb, fill=z, align="center")
    r = 5 + len(fr) + 1
    cell(ws, r, 1, "Borne théorique : réallocation quotidienne, sans limite", font=F_B)
    cell(ws, r, 2, oracle, "0%", font=F_IN, align="right")

    r += 2
    for c, h in enumerate(["Dates de réallocation autorisées", "Performance totale (10 réallocations)"], start=1):
        ws.cell(r, c, h)
    header(ws, r, 1, 2, height=30)
    for i, (lab_, rule) in enumerate([("Toutes les places ouvertes (retenu)", "commun"),
                                      ("Au moins une place ouverte", "cotation"),
                                      ("Tous les jours calendaires", "calendaire")]):
        r += 1
        z = FILL_Z if i % 2 else None
        cell(ws, r, 1, lab_, fill=z)
        cell(ws, r, 2, float(al.optimise(px, rule=rule).gross - 1), PCT, font=F_IN, fill=z, align="right")

    r += 2
    for c, h in enumerate(["Frais de transaction (pb du montant échangé)", "Performance totale nette",
                           "Écart de performance (points)", "Coût en % de la valeur finale"], start=1):
        ws.cell(r, c, h)
    header(ws, r, 1, 4, height=30)
    r_gross = r + 1
    for i, bps in enumerate([0, 5, 10, 20, 50]):
        r += 1
        z = FILL_Z if i % 2 else None
        nav, _ = al.backtest(px, sol.weights, cost_bps=bps)
        cell(ws, r, 1, bps, "0", fill=z, align="center")
        cell(ws, r, 2, float(nav.iloc[-1] / 100 - 1), PCT, font=F_IN, fill=z, align="right")
        cell(ws, r, 3, f"=($B${r_gross}-B{r})*{P_BASE}", '0.0;-0.0;"-"', fill=z, align="right")
        cell(ws, r, 4, f"=1-(1+B{r})/(1+$B${r_gross})", PCT2, fill=z, align="right")
    note(ws, f"A{r + 1}", "Frais appliqués au montant échangé à chaque réallocation (ventes + achats) ; 1 pb = 0,01 %.")
    ws.column_dimensions["A"].width = 50
    for c in "BCD":
        ws.column_dimensions[c].width = 22

    bc = BarChart()
    bc.type = "col"
    bc.title = "Performance maximale selon le nombre de réallocations"
    bc.height, bc.width = 8.5, 18
    bc.add_data(Reference(ws, min_col=2, min_row=4, max_row=4 + len(fr)), titles_from_data=True)
    bc.set_categories(Reference(ws, min_col=1, min_row=5, max_row=4 + len(fr)))
    bc.series[0].graphicalProperties.solidFill = BLUE
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

    # ------------------------------------------------------------------ Mise en page commune
    for w_ in wb.worksheets:
        w_.sheet_view.showGridLines = False
        w_.page_setup.orientation = "landscape"
        w_.page_setup.fitToWidth = 1
        w_.page_setup.fitToHeight = 0
        w_.sheet_properties.pageSetUpPr.fitToPage = True
        if w_.title in TAB:
            w_.sheet_properties.tabColor = TAB[w_.title]
    OUT.parent.mkdir(exist_ok=True)
    wb.save(OUT)
    print(OUT)


if __name__ == "__main__":
    sys.exit(main())
