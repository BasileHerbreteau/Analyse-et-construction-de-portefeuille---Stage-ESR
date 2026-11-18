# Analyse et construction de portefeuille : univers Technologie & Communication

> Application Streamlit d'analyse d'un univers de 99 actions (fichier *universe of stock 261113*), de
> construction d'un **critère de comparaison agrégé** confronté à la **Rule of 40**, et de construction d'un
> **portefeuille diversifié (5 % maximum par titre)** comparé au portefeuille équipondéré de l'univers, avec un
> **interrupteur ESG**.

![Python](https://img.shields.io/badge/Python-3.11+-002855) ![Streamlit](https://img.shields.io/badge/Streamlit-app-0062A3) ![Données](https://img.shields.io/badge/Données-99_titres-5FB3E4)

---

## Sommaire

1. [Consignes et réponses](#1-consignes-et-réponses)
2. [Les données](#2-les-données)
3. [Parcours de l'application](#3-parcours-de-lapplication)
4. [Volatilité estimée](#4-volatilité-estimée)
5. [Critère agrégé (z-scores)](#5-critère-agrégé-z-scores)
6. [Comparaison avec la Rule of 40](#6-comparaison-avec-la-rule-of-40)
7. [Rendement attendu et Sharpe ex ante](#7-rendement-attendu-et-sharpe-ex-ante)
8. [Portefeuille final et interrupteur ESG](#8-portefeuille-final-et-interrupteur-esg)
9. [Scores ESG : sources](#9-scores-esg--sources)
10. [Limites](#10-limites)
11. [Lancer l'application](#11-lancer-lapplication)

---

## 1. Consignes et réponses

| Consigne | Où dans l'app | Réponse apportée |
| --- | --- | --- |
| Visualiser l'univers selon des critères pertinents | Onglets 1 à 7 | Lecture en entonnoir : composition → synthèse → rendement/risque → valorisation → croissance → ESG → corrélations |
| Définir un critère de comparaison agrégé et le comparer à la Rule of 40 | Onglet 8 | Score multi-facteurs (z-scores relatifs à la sous-industrie, 4 piliers) ; comparaison en 4 temps (classements, sélections, origine des écarts, profil) |
| Construire un portefeuille diversifié, 5 % max par titre, et le comparer à l'équipondéré | Onglet 12 | Sélection par quotas de sous-industrie, pondération Sharpe maximal ex ante, tableau comparatif complet |

Les critères financiers proviennent **uniquement** du fichier Excel. Seuls les scores ESG sont externes, sourcés titre par titre.

## 2. Les données

99 titres, 17 colonnes : 79 *Information Technology*, 20 *Communication Services* ; 19 sous-industries GICS ; 14 pays (66 titres américains).

| Famille | Colonnes |
| --- | --- |
| Identification / classification | Ticker, nom, secteur GICS, sous-industrie GICS, pays |
| Performance | %1M, %YTD, total return de l'année civile N-1, RSI 14 jours |
| Valorisation | P/E estimé N+1, PEG, FCF yield |
| Croissance / rentabilité | croissance LT du BPA (consensus), croissance du CA, marge nette |
| Attentes | upside des analystes, prime de risque implicite |

Variables dérivées : earnings yield (100/P/E), Rule of 40 (croissance CA + marge), volatilité estimée, Sharpe ex ante et ex post. Les valeurs extrêmes (ex. croissance du CA de +5 288 %) sont écrêtées aux quantiles 2 % / 98 % (réglable).

## 3. Parcours de l'application

| Onglet | Contenu |
| --- | --- |
| 1 · L'univers | Répartition par secteur, sous-industrie, pays (donuts, sunburst, barres) |
| 2 · Synthèse | Moyenne, médiane, écart-type, min, max ; par secteur ; carte de chaleur des profils sectoriels |
| 3 · Rendement & risque | Rendements par horizon, volatilité estimée, couple rendement/risque, dispersion intra-groupe |
| 4 · Valorisation & attentes | PEG, upside, P/E, FCF yield par secteur / sous-industrie / pays |
| 5 · Croissance & Rule of 40 | Taux de passage, plan croissance × marge, upside selon la règle |
| 6 · ESG | Scores par secteur et sous-industrie, lien ESG / rendement / risque, position vs médiane de la sous-industrie |
| 7 · Relations entre facteurs | Matrices de corrélation (Pearson, Spearman), régressions, matrice de nuages |
| 8 · Critère agrégé vs Rule of 40 | Construction du score, classement, comparaison détaillée avec la Rule of 40 |
| 9 · Rendement attendu & Sharpe | Décomposition du rendement attendu, Sharpe ex ante et ex post |
| 10 · Frontière efficiente | Frontière de Markowitz long-only plafonnée, portefeuilles de référence |
| 11 · Fiche titre | Profil d'un titre en percentiles face à l'univers et à son secteur |
| 12 · Portefeuille final | Sélection, pondération, comparaison avec l'équipondéré, interrupteur ESG |
| Annexe | Dictionnaire des colonnes, méthodes, statistiques, valeurs extrêmes, table complète |

Chaque onglet se termine par un encadré **« Points clés »** (chiffres recalculés en direct) et **« Notions utilisées »**.

## 4. Volatilité estimée

Le fichier ne contient pas d'historique de prix, mais trois rendements sur des périodes **disjointes** : l'année N-1, le début d'année jusqu'à il y a un mois (déduit de YTD et 1M), le dernier mois. En supposant des log-rendements ℓ_k ~ N(μT_k, σ²T_k) :

$$\hat\mu = \frac{\sum_k \ell_k}{\sum_k T_k}, \qquad \hat\sigma^2 = \frac{1}{2}\sum_{k=1}^{3}\frac{(\ell_k - \hat\mu T_k)^2}{T_k}$$

Trois observations étant peu, l'estimation est rapprochée de la médiane de la sous-industrie : σ = α·σ̂ + (1 − α)·médiane (α = 0,5 par défaut).

## 5. Critère agrégé (z-scores)

1. Chaque facteur est orienté (« plus haut = mieux », le PEG est inversé) puis transformé en **z-score relatif à la sous-industrie** :
   $$z_{i,f} = s_f\,\frac{x_{i,f} - \bar x_{g(i),f}}{\sigma_{g(i),f}}$$
   Pour les petites sous-industries, moyenne et variance sont **rétrécies vers le secteur** (estimateur bayésien empirique : (n·groupe + k·secteur)/(n + k), k = 5), puis bornées à ±3.
2. Les z-scores sont moyennés en **4 piliers** : *Croissance* (croissance LT du BPA, croissance du CA), *Rentabilité* (marge nette, FCF yield), *Valorisation* (PEG inversé, earnings yield), *Attentes du marché* (upside, prime de risque).
3. **Score final = moyenne des 4 piliers.**

## 6. Comparaison avec la Rule of 40

| | Rule of 40 | Critère agrégé |
| --- | --- | --- |
| Définition | croissance du CA + marge ≥ 40 % | moyenne de 4 piliers de z-scores (8 facteurs) |
| Référence | seuil absolu | relative aux pairs |
| Résultat | binaire | classement continu |
| Prix payé, attentes | ignorés | intégrés |

Comparaison en quatre temps (onglet 8) :

1. **Classements** : corrélation de Spearman ≈ 0,46, tau de Kendall ≈ 0,32. Les deux critères vont dans le même sens mais ne partagent qu'environ 20 % de variance de rang.
2. **Sélections** (même nombre de titres que ceux qui passent la règle, 30 sur 99) : accord de 78 %, contre 58 % attendus par hasard, soit un **κ de Cohen ≈ 0,47** (accord modéré) ; 19 titres sur 30 sont retenus par les deux.
3. **Origine des écarts** : le pilier le plus proche de la Rule of 40 est la rentabilité ; les écarts viennent de la valorisation et des attentes, que la règle ignore, et de la référence relative aux pairs (la règle favorise les éditeurs de logiciels, le score laisse sa chance à chaque sous-industrie). Test de robustesse selon le mode de calcul des z-scores.
4. **Profil des titres retenus** : PEG, upside, rendement attendu, volatilité, Sharpe et diversification des deux sélections.

**Conclusion** : la Rule of 40 est un filtre de qualité opérationnelle ; le score est un critère de sélection d'investissement. Le portefeuille s'appuie sur le score, la Rule of 40 sert d'indicateur de contrôle.

## 7. Rendement attendu et Sharpe ex ante

$$\mu_i = \underbrace{(1-\lambda)\,\text{Prime}_i + \lambda\,(\text{Upside}_i - r_f)}_{\text{niveau}} + \underbrace{IC\cdot\sigma_i\cdot Z^{\alpha}_i}_{\text{alpha z-scores}}, \qquad \text{Sharpe}^{ex\,ante}_i = \frac{\mu_i}{\sigma_i}$$

- **Prime de risque implicite** : rendement en excès du sans-risque déduit des cours et des estimations ; elle fixe le niveau.
- **Upside** : les objectifs de cours sont structurellement optimistes (95 % des titres ont un upside positif, médiane 25 %). Une moyenne 50/50 avec la prime donnait un rendement attendu médian de 18,5 %, jugé irréaliste. L'upside entre donc par son **z-score** (information relative, sans biais de niveau) : λ = 0 par défaut, réglable.
- **Z-scores** : Z^α = moyenne des piliers croissance, rentabilité, valorisation et du z-score de l'upside ; alpha selon la règle de Grinold (IC = 0,10).
- Résultat : rendement attendu médian ≈ 9,8 %.

## 8. Portefeuille final et interrupteur ESG

1. **Éligibilité** : tout l'univers (ESG coupé) ou, ESG activé, les titres dont le score ESG est **strictement supérieur à la médiane ESG de leur sous-industrie**. Aucun titre n'est écarté en amont pour son ESG.
2. **Quotas** : N titres (30 par défaut) répartis entre sous-industries proportionnellement à leur taille (plus forts restes).
3. **Sélection** : meilleurs scores de chaque sous-industrie, **seulement s'ils battent la médiane de leur sous-industrie** ; les places libres vont aux meilleurs scores restants.
4. **Pondération** : maximisation du Sharpe ex ante sous Σw = 1, 0 ≤ w ≤ 5 % (SLSQP). Covariance Σ = D·R·D avec corrélations par blocs (même sous-industrie 0,60, même secteur 0,40, autres 0,25, réglables).
5. **Comparaison** avec l'équipondéré des 99 titres : rendement attendu, volatilité, Sharpe ex ante et ex post, performances N-1 et YTD, score moyen, upside, prime, PEG, part Rule of 40, score ESG, parts tech / États-Unis, concentration.

Résultats avec les paramètres par défaut :

| | Équipondéré (99) | Sans ESG | Avec filtre ESG |
| --- | --- | --- | --- |
| Lignes | 99 | 22 | 21 |
| Rendement attendu | 11,2 % | 15,4 % | 12,0 % |
| Volatilité estimée | 29,3 % | 23,4 % | 21,3 % |
| Sharpe ex ante | 0,38 | 0,66 | 0,56 |
| PEG moyen | 1,57 | 0,90 | 1,26 |
| Part passant la Rule of 40 | 30 % | 65 % | 25 % |
| Score ESG moyen | 47,8 | 47,6 | 59,6 |

## 9. Scores ESG : sources

- **Fournisseur unique : S&P Global** (échelle 0–100, plus haut = meilleur, évaluation relative à l'industrie), un seul fournisseur pour que les scores soient comparables.
- 69 titres : *S&P Global ESG Score* publié sur la page publique de chaque société (`spglobal.com/sustainable1/en/scores/results?cid=…`, scores 2025–2026).
- 12 titres sans page publique : *CSA Score* du 22/11/2024 publié par S&P Global dans le document officiel de rééquilibrage du DJSI.
- 18 titres sans score public trouvé : ils restent dans toutes les analyses mais ne sont pas éligibles quand le filtre ESG est activé.
- Fichier : `data/esg_scores.csv` (score, date, fournisseur, URL de la source pour chaque titre).

## 10. Limites

- Volatilités estimées sur 3 observations par titre et corrélations **supposées** (pas d'historique de prix dans le fichier).
- Rendement attendu = hypothèse (prime implicite + signal des z-scores) ; l'IC est une hypothèse usuelle, pas une estimation.
- Les performances passées des portefeuilles sont descriptives (poids d'aujourd'hui appliqués au passé), pas un backtest.
- Couverture ESG partielle et dates de score hétérogènes.

## 11. Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Déploiement sur Streamlit Community Cloud** : sur [share.streamlit.io](https://share.streamlit.io), *Create app* → ce dépôt, branche `main`, fichier `app.py`.

**Logo** : l'application affiche le logo Amundi depuis le site d'Amundi ; pour un affichage hors ligne, déposer le fichier officiel dans `assets/logo_amundi.png`. Logo et marque Amundi : propriété d'Amundi.

### Structure du code

| Fichier | Rôle |
| --- | --- |
| `app.py` | Interface Streamlit (12 onglets + annexe) |
| `data_utils.py` | Chargement, renommage, variables dérivées, écrêtage, statistiques |
| `analytics.py` | Volatilité multi-périodes, covariance par blocs, frontière efficiente |
| `scoring.py` | z-scores, piliers, score final, rendement attendu, comparaison Rule of 40 |
| `portfolio.py` | Quotas, sélection, Sharpe maximal, contributions au risque, indicateurs |
| `dictionary.py` | Définitions des colonnes et des variables dérivées |
| `data/` | Fichier Excel source, scores ESG sourcés |
