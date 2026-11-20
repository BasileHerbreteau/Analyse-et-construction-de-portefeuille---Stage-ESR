# Analyse et construction de portefeuille : univers Technologie & Communication

Application Streamlit qui analyse un univers de **99 actions** des secteurs Information Technology et Communication
Services (fichier *universe of stock 261113*, données au 13/11/2026), construit un critère de sélection comparé à la
**Rule of 40**, puis un **portefeuille diversifié de 30 titres (5 % maximum par ligne)** comparé à l'équipondéré.
Deux onglets bonus complètent l'ensemble : **Historique Bloomberg** confronte les hypothèses de risque à deux ans de
prix, et **Track record**, indépendant, traite un exercice d'allocation dynamique sur un autre jeu de données.

## Contenu de l'application

### Partie 1 · Analyse de l'univers
* Composition par sous-industrie et par pays.
* Profil comparé des deux secteurs : performance, risque, valorisation, rentabilité, attentes des analystes.
* Estimation de la volatilité à partir de trois périodes de rendement.
* Visualisation de la relation rendement / volatilité et de la persistance des performances.
* Analyse des multiples de valorisation (P/E, PEG) et de l'upside des analystes.

### Partie 2 · Critère agrégé et Rule of 40
* Matrice de corrélation des variables du fichier.
* Regroupement des facteurs en quatre piliers : Croissance, Rentabilité, Valorisation, Attentes du marché.
* Score multi-facteurs relatif aux pairs (z-scores par sous-industrie).
* Comparaison avec la Rule of 40 : corrélations de rang, titres retenus par les deux critères, profil des sélections.

### Partie 3 · Portefeuille
* Scores ESG Bloomberg et position de chaque titre par rapport à sa sous-industrie.
* Rendement attendu ex ante combinant la prime de risque du titre et son score.
* Frontière efficiente, Capital Market Line et portefeuille tangent, sensibilité aux hypothèses de corrélation.
* Portefeuille final, avec ou sans filtre ESG, comparé à l'équipondéré : poids, contributions au risque, composition.

## Méthode en bref

* **Données** : les critères financiers viennent uniquement du fichier Excel. Les valeurs extrêmes sont écrêtées aux
  quantiles 2 % / 98 %, sauf l'upside des analystes, conservé tel quel. Taux sans risque : T-bill américain 3 mois au
  13/11/2026.
* **Volatilité** : écart-type annualisé de trois rendements sur des périodes disjointes, rapproché de la médiane de la
  sous-industrie pour réduire le bruit.
* **Corrélations** : faute d'historique, structure par blocs supposée (0,60 entre titres d'une même sous-industrie,
  0,40 au sein d'un secteur, 0,25 sinon). Une corrélation moyenne par groupe réduit l'erreur d'estimation (modèle à
  corrélation constante, Ledoit et Wolf, 2003) ; les niveaux sont prudents car les corrélations montent en marché
  baissier (Longin et Solnik, 2001). Sensibilité testée de 0,40 / 0,25 / 0,10 à 0,80 / 0,60 / 0,40.
* **Critère agrégé** : chaque facteur est comparé aux titres de la même sous-industrie, puis les quatre piliers sont
  moyennés à poids égal.
* **Rendement attendu** : la prime de risque implicite donne le niveau ; le score ajoute un alpha proportionnel à la
  volatilité du titre (règle de Grinold).
* **Portefeuille** : 30 places réparties entre sous-industries selon leur taille, titres retenus seulement s'ils battent
  la médiane de leur sous-industrie, pondération qui maximise le ratio de Sharpe ex ante.
* **Filtre ESG** : score ESG Bloomberg (0 à 10) ; seuls les titres mieux notés que la médiane de leur sous-industrie
  sont éligibles, avec repêchage des bons scores pour garder la diversification.
* **Capitalisations** : Bloomberg, clôture du 13/11/2026 en dollars. Elles servent seulement au portefeuille de
  référence pondéré par la capitalisation et à la comparaison des tailles, pas à la construction.

## Résultats

| | Équipondéré (99) | Sans ESG | Avec filtre ESG | Pondéré par la capitalisation (99) |
| --- | --- | --- | --- | --- |
| Rendement attendu | 11,6 % | 18,9 % | 14,2 % | 12,9 % |
| Volatilité estimée | 29,3 % | 24,1 % | 21,7 % | 30,6 % |
| Sharpe ex ante | 0,40 | 0,78 | 0,65 | 0,42 |
| Score ESG moyen (Bloomberg) | 4,81 | 4,66 | 5,44 | 4,90 |
| Capitalisation médiane des lignes | 39 Md$ | 142 Md$ | 95 Md$ | 39 Md$ |

Le gain de Sharpe par rapport à l'équipondéré reste positif sous des hypothèses de corrélation basses comme hautes.

## Limites

**Données**
* Une seule date d'observation, sans historique de prix : volatilités estimées sur trois rendements, corrélations
  supposées et non estimées (confrontées à deux ans de prix dans le bonus Historique Bloomberg).
* Rendements en devise locale (convention Bloomberg) : pour un investisseur en euros, la variation du dollar sur les titres américains
  (deux tiers de l'univers) n'est pas prise en compte.
* Upside, croissance attendue et PEG reposent sur le consensus des analystes, structurellement optimiste.

**Modèle**
* Pondération égale des piliers et coefficient d'information de 0,10 : hypothèses raisonnées, non estimées.
* Aucun backtest, faute d'historique des données : le pouvoir prédictif du score n'est pas démontré.

**Portefeuille et ESG**
* La maximisation du Sharpe amplifie les erreurs d'estimation ; le plafond de 5 % et les quotas les limitent.
* Le portefeuille peut pencher vers des titres plus sensibles au marché ou d'un même style (croissance) : une partie
  de sa performance viendrait alors de ces paris plutôt que de la sélection des titres.
* Portefeuille construit une seule fois : ni rééquilibrage dans le temps, ni frais de transaction.

## Bonus · Historique Bloomberg

Information **externe** au fichier Excel, sans effet sur les parties 1 à 3 : 104 semaines de prix en dollars extraites
de Bloomberg (15/11/2024 – 13/11/2026), avec le MSCI World comme marché.

* **Volatilité** : médiane estimée 43,7 % contre 41,8 % mesurée ; corrélation de rang 0,66. L'estimation sur trois
  rendements donne le bon niveau de risque mais classe moins bien les titres entre eux.
* **Bêtas** : pente de la régression des rendements hebdomadaires du titre sur ceux du MSCI World (bêta de 1,40 : le
  titre varie en moyenne de 1,40 % quand le marché varie de 1 %), avec la version ajustée de Blume (0,67 × bêta + 0,33).
* **Corrélations** : mesurées 0,43 / 0,22 / 0,11 contre 0,60 / 0,40 / 0,25 supposées : l'hypothèse est prudente.
* **Portefeuilles reconstruits** avec un modèle à un facteur (Σ = σ²ₘββᵀ + D) et la volatilité mesurée, sélection
  inchangée :

| Modèle à un facteur | Équipondéré (99) | Sans ESG, onglet 8 | Sans ESG, bêtas | Avec ESG, onglet 8 | Avec ESG, bêtas |
| --- | --- | --- | --- | --- | --- |
| Lignes | 99 | 21 | 23 | 21 | 27 |
| Sharpe ex ante | 0,58 | 0,83 | 0,88 | 0,71 | 0,79 |
| Volatilité mesurée (2 ans) | 22,4 % | 25,7 % | 24,0 % | 21,2 % | 19,4 % |
| Bêta mesuré | 1,43 | 1,59 | 1,51 | 1,35 | 1,19 |

19 des 21 lignes de chaque portefeuille sont conservées : la sélection tient sur données réelles, l'historique change
surtout les poids et ajoute quelques lignes.

## Bonus · Track record

Exercice **indépendant** des parties 1 à 3, sur un **autre jeu de données** (fichier *Exercice allocation 261113*) :
cours quotidiens de 29 sous-jacents (indices sectoriels STOXX Europe 600 et EURO STOXX, Russell 2000, S&P
Communication Services, trois ETF). Il ne réutilise ni l'univers d'actions, ni le score, ni les filtres de l'application.

* **Objectif** : track record du 01/01/2026 au 30/10/2026 du portefeuille à allocation dynamique qui maximise la
  performance totale réalisée, avec 20 % maximum par sous-jacent et 10 réallocations au plus (allocation de départ
  non comptée).
* **Méthode** : entre deux dates d'allocation, les positions sont conservées ; la performance d'une période est donc
  linéaire en poids et son maximum est atteint avec 20 % sur les 5 sous-jacents qui progressent le plus. Les dates de
  réallocation sont choisies par programmation dynamique, ce qui donne l'optimum global (contrôlé par recherche
  exhaustive et tirages aléatoires).
* **Présentation** : comparaison avec l'équipondéré et le meilleur portefeuille sans réallocation, périodes de
  détention, contributions par sous-jacent, gain apporté par chaque réallocation, sensibilité aux frais de transaction.
* **Classeur Excel** `resultats/track_record_allocation_dynamique.xlsx`, téléchargeable depuis l'onglet : track
  record recalculé par formules à partir des dates et des poids, modifiables.

| 01/01 – 30/10/2026 | Dynamique optimal | Statique optimal (0 réallocation) | Équipondéré (29) |
| --- | --- | --- | --- |
| Performance totale | 189,1 % | 28,7 % | 7,8 % |
| Volatilité annualisée | 16,4 % | 16,6 % | 11,6 % |
| Perte maximale | -5,2 % | -7,2 % | -8,8 % |

Optimum **ex post** (cours de la période connus à l'avance) : borne haute permise par les contraintes, pas une
stratégie investissable. Indices de prix en devise de cotation, performances brutes de frais (10 pb par réallocation
ramènent la performance à 183,7 %). Taux sans risque du ratio de Sharpe : €STR moyen de la période, soit 2,11 % en
taux annuel.

## Ouvrir l'application

Depuis une copie du dépôt (Python 3.10 ou plus récent) :

```bash
git clone https://github.com/BasileHerbreteau/Analyse-et-construction-de-portefeuille---Stage-ESR.git
cd Analyse-et-construction-de-portefeuille---Stage-ESR
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre dans le navigateur à l'adresse http://localhost:8501. Le classeur Excel du bonus s'ouvre aussi
directement depuis le dossier `resultats/`.

Code : `app.py` (interface), `data_utils.py` (données, ESG, capitalisations, historique Bloomberg), `analytics.py`
(volatilité, frontière), `scoring.py` (z-scores, rendement attendu, Rule of 40), `portfolio.py` (sélection,
optimisation). Bonus : `tab_bloomberg.py` (historique Bloomberg, modèle à un facteur), `allocation.py` (optimisation et
track record), `tab_allocation.py` (onglet), `export_allocation_excel.py` (classeur Excel).
