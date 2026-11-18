# Analyse et construction de portefeuille : univers Technologie & Communication

Application Streamlit : analyse d'un univers de **99 actions** (fichier *universe of stock 261113*, données au 13/11/2026),
critère de comparaison agrégé confronté à la **Rule of 40**, et **portefeuille diversifié (5 % maximum par titre)**
comparé à l'équipondéré, avec ou sans **filtre ESG**.

## Sommaire de l'application

| Onglet | Contenu |
| --- | --- |
| 1 · Univers | Composition par secteur, sous-industrie, pays |
| 2 · Synthèse | Statistiques globales et par secteur |
| 3 · Rendement & risque | Méthode de volatilité, rendement vs risque, persistance |
| 4 · Valorisation & attentes | PEG, P/E, FCF yield, upside |
| 5 · Relations entre facteurs | Corrélations et pondération du score sans redondance |
| 6 · Rule of 40 & critère agrégé | Construction du score et comparaison avec la Rule of 40 |
| 7 · ESG | Scores S&P Global, lien avec rendement et risque |
| 8 · Rendement attendu & Sharpe | Rendement ex ante, pertinence du score, Sharpe |
| 9 · Frontière efficiente | Frontière de Markowitz, hypothèses de corrélation et sensibilité |
| 10 · Portefeuille final | Sélection, pondération, comparaison, filtre ESG, fiche titre |

## Méthode en bref

**Données**
- Critères financiers issus du seul fichier Excel ; scores ESG S&P Global sourcés titre par titre.
- Valeurs extrêmes écrêtées aux quantiles 2 % / 98 % (winsorisation).
- Taux sans risque : 4,32 %, T-bill américain 3 mois au 13/11/2026 (FRED, DGS3MO).

**Volatilité**
- Écart-type annualisé de 3 rendements disjoints (année N-1, YTD hors dernier mois, dernier mois).
- Moyenne 50/50 entre l'estimation du titre et la médiane de sa sous-industrie.

**Critère agrégé**
- 8 facteurs transformés en z-scores relatifs à la sous-industrie.
- 4 piliers à poids égal : Croissance, Rentabilité, Valorisation, Attentes du marché.
- Score final = moyenne des 4 piliers.

**Comparaison avec la Rule of 40**
- Spearman 0,46 et κ de Cohen 0,47 : les deux critères sont liés, avec un accord modéré.
- La Rule of 40 retient des titres plus chers et plus concentrés sectoriellement.
- Le score intègre en plus le prix payé et les attentes, relativement aux pairs.

**Rendement attendu**
- Niveau : prime de risque implicite, déjà nette du taux sans risque.
- Alpha : IC × volatilité × score standardisé (règle de Grinold), où le score inclut l'upside relatif aux pairs.
- Rendement attendu médian ≈ 9,6 %.

**Portefeuille**
- Quotas par sous-industrie au prorata de leur taille.
- Sélection des titres dont le score dépasse la médiane de leur sous-industrie.
- Pondération qui maximise le Sharpe ex ante, avec 5 % maximum par titre.
- Filtre ESG : score ESG supérieur à la médiane de la sous-industrie, avec repêchage des bons scores pour préserver la diversification.

## Résultats (paramètres par défaut)

| | Équipondéré (99) | Sans ESG | Avec filtre ESG |
| --- | --- | --- | --- |
| Rendement attendu | 11,7 % | 18,9 % | 15,7 % |
| Volatilité estimée | 29,2 % | 24,1 % | 25,1 % |
| Sharpe ex ante | 0,40 | 0,78 | 0,63 |

Le gain de Sharpe par rapport à l'équipondéré reste positif sous des hypothèses de corrélation basses comme hautes.

## Limites

- Pas d'historique de prix : volatilités estimées sur 3 observations, corrélations supposées et testées en sensibilité.
- Scores ESG : 81 titres sur 99 couverts, dont 12 avec un score de 2024.

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Déploiement : [share.streamlit.io](https://share.streamlit.io) → *Create app* → ce dépôt, branche `main`, fichier `app.py`.

Code : `app.py` (interface), `data_utils.py` (données), `analytics.py` (volatilité, frontière), `scoring.py`
(z-scores, rendement attendu, Rule of 40), `portfolio.py` (sélection, optimisation). Logo et marque Amundi : propriété d'Amundi.
