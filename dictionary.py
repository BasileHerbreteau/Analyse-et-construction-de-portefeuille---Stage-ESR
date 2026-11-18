"""Dictionnaire des colonnes du fichier « universe of stock 261113 ».

Les définitions décrivent ce que mesure chaque champ (libellés de type Bloomberg)
et comment le lire. Elles ne s'appuient sur aucune donnée extérieure au fichier.
"""

COLUMNS = [
    {
        "col": "Upside from Analyst Aggregate view", "code": "upside", "famille": "Consensus analystes",
        "unite": "fraction (0,25 = +25 %), convertie en % dans l'app",
        "definition": "Potentiel de hausse implicite : écart entre l'objectif de cours moyen des analystes (consensus) et le cours actuel, rapporté au cours. (Objectif − Cours) / Cours.",
        "lecture": "Plus il est élevé, plus les analystes jugent le titre sous-évalué. Négatif = cours au-dessus de l'objectif. Indicateur prospectif mais sujet aux biais d'optimisme des analystes.",
    },
    {
        "col": "Ticker", "code": "ticker", "famille": "Identification",
        "unite": "texte", "definition": "Code mnémonique de cotation de l'action (ex. BESI, ASM, ZS).",
        "lecture": "Clé unique de chaque ligne (99 titres, aucun doublon).",
    },
    {
        "col": "Short Name", "code": "name", "famille": "Identification",
        "unite": "texte", "definition": "Nom abrégé de la société (tronqué à 16 caractères).",
        "lecture": "Purement descriptif.",
    },
    {
        "col": "GICS Sector", "code": "sector", "famille": "Classification",
        "unite": "catégorie", "definition": "Secteur selon la classification GICS (niveau 1 sur 4). Ici seulement deux secteurs : Information Technology (79 titres) et Communication Services (20).",
        "lecture": "Sert aux comparaisons sectorielles et aux contraintes de diversification. L'univers est très concentré sur la tech.",
    },
    {
        "col": "GICS SubInd Name", "code": "subindustry", "famille": "Classification",
        "unite": "catégorie", "definition": "Sous-industrie GICS (niveau le plus fin, 4/4) : Semiconductors, Application Software, Systems Software, Integrated Telecom… 19 sous-industries, libellés tronqués à 30 caractères.",
        "lecture": "Plus pertinent que le secteur pour comparer des pairs réellement similaires et pour diversifier (un semi-conducteur et un opérateur télécom n'ont pas le même profil).",
    },
    {
        "col": "Cntry Terrtry Fl Name", "code": "country", "famille": "Classification",
        "unite": "catégorie", "definition": "Pays de domiciliation (nom complet). 14 pays, dont 66 titres américains et 33 non américains (Europe surtout, plus Israël, Australie, Canada).",
        "lecture": "Diversification géographique, exposition devises/pays. Univers dominé par les États-Unis.",
    },
    {
        "col": "RSI 14D", "code": "rsi14", "famille": "Technique / momentum court terme",
        "unite": "indice 0–100", "definition": "Relative Strength Index sur 14 jours : 100 − 100 / (1 + moyenne des hausses / moyenne des baisses) sur les 14 dernières séances.",
        "lecture": "> 70 : zone de surachat (hausse récente forte, risque de repli) ; < 30 : survente ; ~50 : neutre. Indicateur de timing, pas de valeur fondamentale.",
    },
    {
        "col": "%1M", "code": "ret_1m", "famille": "Performance",
        "unite": "%", "definition": "Variation du cours sur le dernier mois.",
        "lecture": "Momentum très court terme (souvent sujet à retournement). Seule information de mouvement récent : sert aussi de proxy d'amplitude/volatilité faute d'historique de prix.",
    },
    {
        "col": "%YTD", "code": "ret_ytd", "famille": "Performance",
        "unite": "%", "definition": "Variation du cours depuis le 1er janvier de l'année en cours (Year-To-Date) jusqu'à la date d'extraction.",
        "lecture": "Momentum moyen terme. Attention aux valeurs extrêmes (ex. +345 % pour SOI).",
    },
    {
        "col": "Previous 1 Calendar Year - Historical Total Return", "code": "tr_prev_year", "famille": "Performance",
        "unite": "%", "definition": "Rendement total (variation de cours + dividendes réinvestis) sur l'année civile précédente complète.",
        "lecture": "Rendement historique « complet » sur 12 mois ; à comparer au YTD pour voir la persistance (ou le retournement) de la performance.",
    },
    {
        "col": "FCF Yld", "code": "fcf_yield", "famille": "Valorisation",
        "unite": "%", "definition": "Free cash-flow yield : flux de trésorerie disponible (cash-flow opérationnel − capex) par action ÷ cours (équivalent FCF ÷ capitalisation).",
        "lecture": "Rendement « cash » de l'action : plus il est élevé, moins le titre est cher par rapport au cash qu'il génère. Négatif = l'entreprise consomme du cash (ex. VUSION −14,5 %).",
    },
    {
        "col": "Est P/E Nxt Yr", "code": "pe_next", "famille": "Valorisation",
        "unite": "multiple (x)", "definition": "Price/Earnings estimé de l'année prochaine : cours ÷ BPA (bénéfice par action) prévu par le consensus pour l'exercice suivant.",
        "lecture": "Combien d'années de bénéfices futurs on paie. Élevé = valeur chère ou forte croissance attendue (NET 168x, CRWD 143x). Son inverse est l'earnings yield.",
    },
    {
        "col": "BEst Est LTG EPS", "code": "ltg_eps", "famille": "Croissance (consensus)",
        "unite": "% par an", "definition": "Bloomberg Estimate – Long-Term Growth du BPA : taux de croissance annuel moyen du bénéfice par action attendu par les analystes à long terme (en général 3 à 5 ans).",
        "lecture": "Mesure de croissance prospective des bénéfices. Très corrélé à l'upside (ρ≈0,70) : les analystes qui attendent beaucoup de croissance mettent des objectifs élevés.",
    },
    {
        "col": "BEst PEG Ratio", "code": "peg", "famille": "Valorisation ajustée de la croissance",
        "unite": "multiple (x)", "definition": "Price/Earnings-to-Growth : P/E ÷ croissance attendue du BPA (en %).",
        "lecture": "< 1 : croissance « pas chère » ; ≈ 1 : prix cohérent avec la croissance ; > 2 : on paie cher la croissance. Permet de comparer des titres de croissance différente.",
    },
    {
        "col": "Rsk Prem", "code": "risk_prem", "famille": "Risque / rendement attendu",
        "unite": "%", "definition": "Prime de risque implicite : rendement attendu implicite du titre (dérivé du cours et des estimations des analystes, type modèle d'actualisation) moins le taux sans risque. Interprétation déduite du libellé et des données : elle varie fortement au sein d'un même pays (US de 3,3 % à 25,4 %), ce n'est donc pas une prime pays.",
        "lecture": "Élevée = le marché « exige » (ou le modèle implique) un rendement supérieur au sans-risque : soit opportunité, soit titre perçu comme risqué. Corrélée à la croissance attendue (ρ≈0,63) et à l'upside (ρ≈0,58).",
    },
    {
        "col": "SALES_GROWTH", "code": "sales_growth", "famille": "Fondamentaux",
        "unite": "%", "definition": "Croissance du chiffre d'affaires sur un an (dernier exercice vs précédent).",
        "lecture": "Premier composant de la « Rule of 40 ». Attention : VEND affiche +5 288 % (effet de base / opération exceptionnelle) : valeur aberrante à traiter (écrêtage).",
    },
    {
        "col": "PROF_MARGIN", "code": "profit_margin", "famille": "Fondamentaux",
        "unite": "%", "definition": "Marge bénéficiaire nette : résultat net ÷ chiffre d'affaires.",
        "lecture": "Rentabilité finale de l'activité. Second composant de la « Rule of 40 » (croissance + marge ≥ 40 %). Négative = entreprise déficitaire (ex. VEND −103 %, SOI −37 %, SNOW −28 %).",
    },
]

DERIVED_DOC = [
    ("Earnings yield N+1", "100 ÷ P/E N+1", "Rendement bénéficiaire attendu, comparable au FCF yield."),
    ("Rule of 40", "Croissance CA + marge nette", "Seuil de 40 % : arbitrage croissance / rentabilité."),
    ("Volatilité estimée (méthode principale)", "écart-type annualisé de 3 log-rendements disjoints (année N-1, YTD hors dernier mois, dernier mois), rapproché de la médiane de la sous-industrie", "Volatilité par titre sans historique de prix (voir Méthodes)."),
    ("Amplitude 1 mois annualisée (méthode alternative)", "|perf 1M| × √12", "Une seule observation : proxy grossier, gardé pour comparaison."),
    ("z-score d'un facteur", "(x − moyenne des pairs) ÷ écart-type des pairs, signe inversé pour le PEG", "Met tous les facteurs dans la même unité, relativement à la sous-industrie."),
    ("Score final (critère agrégé)", "moyenne des 4 piliers (Croissance, Rentabilité, Valorisation, Attentes du marché)", "Critère de comparaison agrégé, comparé à la Rule of 40."),
    ("Rendement attendu en excès", "(1 − λ)·prime de risque + λ·(upside − rf) + IC·σ·Z^α, λ = 0 par défaut ; Z^α = moyenne (croissance, rentabilité, valorisation, z-score de l'upside)", "Numérateur du Sharpe ex ante, axe vertical de la frontière, entrée de l'optimisation du portefeuille."),
    ("Sharpe ex ante", "rendement attendu en excès ÷ volatilité", "Rendement espéré par unité de risque."),
    ("Sharpe ex post", "(total return N-1 − rf) ÷ volatilité", "Rendement constaté par unité de risque."),
    ("Distance au RSI neutre", "|RSI − 50|", "Intensité du mouvement récent, quel que soit son sens."),
    ("Écart YTD − 1M", "perf YTD − perf 1M", "Retournement de tendance récent (forte hausse annuelle mais baisse récente, ou l'inverse)."),
    ("FCF yield − earnings yield", "FCF yld − 100/PE", "Qualité de la conversion des bénéfices en cash."),
    ("Dispersion sectorielle", "écart-type inter-titres d'un groupe", "Volatilité transversale : hétérogénéité des performances au sein d'un secteur / sous-industrie."),
]
