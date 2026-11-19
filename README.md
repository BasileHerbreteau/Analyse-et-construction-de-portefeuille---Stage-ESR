# Analyse et construction de portefeuille : univers Technologie & Communication

Application Streamlit qui analyse un univers de **99 actions** des secteurs Information Technology et Communication
Services (fichier *universe of stock 261113*, données au 13/11/2026), construit un critère de sélection comparé à la
**Rule of 40**, puis un **portefeuille diversifié de 30 titres (5 % maximum par ligne)** comparé à l'équipondéré.

## Contenu de l'application

### Partie 1 · Analyse de l'univers
* Composition par sous-industrie et par pays.
* Profil comparé des deux secteurs : performance, risque, valorisation, rentabilité, attentes des analystes.
* Estimation de la volatilité à partir de trois périodes de rendement.
* Comparaison des performances par horizon (1 mois, YTD, année N-1).
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
* Frontière efficiente et sensibilité aux hypothèses de corrélation.
* Portefeuille final, avec ou sans filtre ESG, comparé à l'équipondéré : poids, contributions au risque, composition.

## Méthode en bref

* **Données** : les critères financiers viennent uniquement du fichier Excel. Les valeurs extrêmes sont écrêtées aux
  quantiles 2 % / 98 %, sauf l'upside des analystes, conservé tel quel. Taux sans risque : T-bill américain 3 mois au
  13/11/2026.
* **Volatilité** : écart-type annualisé de trois rendements sur des périodes disjointes, rapproché de la médiane de la
  sous-industrie pour réduire le bruit.
* **Critère agrégé** : chaque facteur est comparé aux titres de la même sous-industrie, puis les quatre piliers sont
  moyennés à poids égal.
* **Rendement attendu** : la prime de risque implicite donne le niveau ; le score ajoute un alpha proportionnel à la
  volatilité du titre (règle de Grinold).
* **Portefeuille** : 30 places réparties entre sous-industries selon leur taille, titres retenus seulement s'ils battent
  la médiane de leur sous-industrie, pondération qui maximise le ratio de Sharpe ex ante.
* **Filtre ESG** : seuls les titres mieux notés que la médiane de leur sous-industrie sont éligibles, avec repêchage
  des bons scores pour garder la diversification.

## Résultats

| | Équipondéré (99) | Sans ESG | Avec filtre ESG |
| --- | --- | --- | --- |
| Rendement attendu | 11,6 % | 18,9 % | 15,5 % |
| Volatilité estimée | 29,3 % | 24,1 % | 24,9 % |
| Sharpe ex ante | 0,40 | 0,78 | 0,62 |

Le gain de Sharpe par rapport à l'équipondéré reste positif sous des hypothèses de corrélation basses comme hautes.

## Limites

**Données**
* Une seule date d'observation : volatilités estimées sur trois rendements, corrélations supposées et non estimées.
* Ni capitalisation ni liquidité : capacité, coûts de transaction et liquidité ne sont pas mesurés.
* Devise des rendements non précisée : le risque de change n'est pas modélisé.
* Upside, croissance attendue et PEG reposent sur le consensus des analystes, structurellement optimiste.
* Sous-industries de 1 ou 2 titres : comparaisons entre pairs fragiles.

**Modèle**
* Pondération égale des piliers et coefficient d'information de 0,10 : hypothèses raisonnées, non estimées.
* Aucun backtest : le pouvoir prédictif du score n'est pas démontré.
* Loi normale pour la volatilité : les risques extrêmes sont sous-estimés.
* Rule of 40 pensée pour les éditeurs de logiciels, moins pertinente pour les télécoms ou le matériel.

**Portefeuille et ESG**
* La maximisation du Sharpe amplifie les erreurs d'estimation ; le plafond de 5 % et les quotas les limitent.
* Expositions factorielles (bêta, taille, style, momentum) non contrôlées ; portefeuille statique, sans coûts.
* Scores ESG de dates différentes et relatifs à l'industrie ; un autre fournisseur pourrait classer différemment.

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Déploiement : [share.streamlit.io](https://share.streamlit.io) → *Create app* → ce dépôt, branche `main`, fichier `app.py`.

Code : `app.py` (interface), `data_utils.py` (données), `analytics.py` (volatilité, frontière), `scoring.py`
(z-scores, rendement attendu, Rule of 40), `portfolio.py` (sélection, optimisation).
