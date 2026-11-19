# Analyse et construction de portefeuille : univers Technologie & Communication

Application Streamlit qui analyse un univers de **99 actions** des secteurs Information Technology et Communication
Services (fichier *universe of stock 261113*, données au 13/11/2026), construit un critère de sélection comparé à la
**Rule of 40**, puis un **portefeuille diversifié de 30 titres (5 % maximum par ligne)** comparé à l'équipondéré.

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
* Scores ESG S&P Global et position de chaque titre par rapport à sa sous-industrie.
* Rendement attendu ex ante combinant la prime de risque du titre et son score.
* Frontière efficiente, Capital Market Line et portefeuille tangent, sensibilité aux hypothèses de corrélation.
* Portefeuille final, avec ou sans filtre ESG, comparé à l'équipondéré : poids, contributions au risque, composition.
* Comparaison avec un portefeuille pondéré par la capitalisation et taille des sociétés détenues (capitalisation
  médiane, petites capitalisations dont le poids serait à revoir en pratique).

### Onglet 9 · Allocation dynamique (exercice séparé : *Exercice allocation 261113*)
* Track record du 01/01/2026 au 30/10/2026 d'un portefeuille sur 29 sous-jacents (indices sectoriels STOXX,
  Russell 2000, ETF) : 20 % maximum par sous-jacent, 10 réallocations au plus (allocation de départ non comptée).
* Optimum exact ex post : sur chaque période, 20 % sur les 5 sous-jacents qui montent le plus ; dates choisies par
  programmation dynamique (optimum global, contrôlé par recherche exhaustive et tirages aléatoires).
* Comparaison avec l'équipondéré, le meilleur portefeuille sans réallocation et l'allocation d'exemple du fichier ;
  périodes de détention, contributions, gain apporté par chaque réallocation, frais de transaction.
* Classeur Excel `resultats/track_record_allocation_dynamique.xlsx` : track record recalculé par formules (poids et
  dates modifiables), régénéré par `python export_allocation_excel.py`.

| 01/01 – 30/10/2026 | Dynamique optimal | Statique optimal (0 réallocation) | Exemple du fichier | Équipondéré (29) |
| --- | --- | --- | --- | --- |
| Performance totale | 189,1 % | 28,7 % | 4,0 % | 7,8 % |
| Volatilité annualisée | 16,4 % | 16,6 % | 15,0 % | 11,6 % |
| Perte maximale | -5,2 % | -7,2 % | -12,6 % | -8,8 % |

Optimum ex post (prix futurs connus) : borne haute permise par les contraintes, pas une stratégie investissable.
Indices de prix en devise de cotation, sans frais ; poids plafonnés à chaque date d'allocation puis libres de dériver.
Taux sans risque du ratio de Sharpe : €STR moyen de la période (2,08 %, soit 2,11 % en taux annuel équivalent ; moyennes
mensuelles relevées sur globalmacro.expert, paliers conformes aux décisions de la BCE).

## Méthode en bref

* **Données** : les critères financiers viennent uniquement du fichier Excel. Les valeurs extrêmes sont écrêtées aux
  quantiles 2 % / 98 %, sauf l'upside des analystes, conservé tel quel. Taux sans risque : T-bill américain 3 mois au
  13/11/2026.
* **Volatilité** : écart-type annualisé de trois rendements sur des périodes disjointes, rapproché de la médiane de la
  sous-industrie pour réduire le bruit.
* **Corrélations** : faute d'historique, structure par blocs supposée (0,70 entre titres d'une même sous-industrie,
  0,50 au sein d'un secteur, 0,35 sinon), élevée car ces valeurs partagent des moteurs communs ; sensibilité testée.
* **Critère agrégé** : chaque facteur est comparé aux titres de la même sous-industrie, puis les quatre piliers sont
  moyennés à poids égal.
* **Rendement attendu** : la prime de risque implicite donne le niveau ; le score ajoute un alpha proportionnel à la
  volatilité du titre (règle de Grinold).
* **Portefeuille** : 30 places réparties entre sous-industries selon leur taille, titres retenus seulement s'ils battent
  la médiane de leur sous-industrie, pondération qui maximise le ratio de Sharpe ex ante.
* **Filtre ESG** : seuls les titres mieux notés que la médiane de leur sous-industrie sont éligibles, avec repêchage
  des bons scores pour garder la diversification.
* **Capitalisations** : relevées sur companiesmarketcap.com (clôture du 13/11/2026 pour 38 titres, cours du 10 au
  18/11 pour la plupart des autres). Elles servent seulement au portefeuille de référence pondéré par la
  capitalisation et à la comparaison des tailles, pas à la construction.

## Résultats

| | Équipondéré (99) | Sans ESG | Avec filtre ESG | Pondéré par la capitalisation (99) |
| --- | --- | --- | --- | --- |
| Rendement attendu | 11,6 % | 19,0 % | 16,8 % | 12,9 % |
| Volatilité estimée | 32,8 % | 27,0 % | 29,3 % | 33,5 % |
| Sharpe ex ante | 0,36 | 0,71 | 0,57 | 0,39 |
| Capitalisation médiane des lignes | 40 Md$ | 145 Md$ | 42 Md$ | 40 Md$ |

Le gain de Sharpe par rapport à l'équipondéré reste positif sous des hypothèses de corrélation basses comme hautes.

## Limites

**Données**
* Une seule date d'observation : volatilités estimées sur trois rendements, corrélations supposées et non estimées.
* Rendements en devise locale (convention Bloomberg) : pour un investisseur en euros, la variation du dollar sur les titres américains
  (deux tiers de l'univers) n'est pas prise en compte.
* Upside, croissance attendue et PEG reposent sur le consensus des analystes, structurellement optimiste.
* Capitalisations issues d'une source externe, relevées à des dates proches du 13/11 mais pas toutes à cette date.

**Modèle**
* Pondération égale des piliers et coefficient d'information de 0,10 : hypothèses raisonnées, non estimées.
* Aucun backtest, faute d'historique des données : le pouvoir prédictif du score n'est pas démontré.

**Portefeuille et ESG**
* La maximisation du Sharpe amplifie les erreurs d'estimation ; le plafond de 5 % et les quotas les limitent.
* Le portefeuille peut pencher vers des titres plus sensibles au marché ou d'un même style (croissance) : une partie
  de sa performance viendrait alors de ces paris plutôt que de la sélection des titres.
* Portefeuille construit une seule fois : ni rééquilibrage dans le temps, ni frais de transaction.
* Scores ESG relatifs à l'industrie, issus de deux sources (pages S&P Global, Bloomberg) et de dates différentes ;
  un autre fournisseur pourrait classer différemment.

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Déploiement : [share.streamlit.io](https://share.streamlit.io) → *Create app* → ce dépôt, branche `main`, fichier `app.py`.

Code : `app.py` (interface), `data_utils.py` (données, ESG, capitalisations), `analytics.py` (volatilité, frontière), `scoring.py`
(z-scores, rendement attendu, Rule of 40), `portfolio.py` (sélection, optimisation), `allocation.py` (allocation dynamique :
optimisation et track record), `tab_allocation.py` (onglet 9), `export_allocation_excel.py` (classeur Excel).
