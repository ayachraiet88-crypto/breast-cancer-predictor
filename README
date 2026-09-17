#  Breast Cancer Predictor — Prédiction de tumeur (malin/bénin)

Application de Machine Learning prédisant si une tumeur du sein est **maligne** ou **bénigne**, à partir de mesures issues d'images médicales numérisées (dataset Breast Cancer Wisconsin). Le projet met l'accent sur la sélection de variables par analyse statistique (z-score) et l'optimisation du recall — critique dans un contexte médical où rater un cas malin est bien plus grave qu'une fausse alerte.

> Projet à but éducatif — ne remplace en aucun cas un diagnostic médical professionnel.

## Objectif du projet

Construire un classifieur fiable pour distinguer les tumeurs malignes des tumeurs bénignes, en optimisant particulièrement le **recall de la classe maligne** (minimiser les faux négatifs, le risque le plus grave dans ce contexte), et identifier objectivement les variables les plus discriminantes.

## Dataset

- **Source** : Breast Cancer Wisconsin (intégré à scikit-learn, `sklearn.datasets.load_breast_cancer`)
- **569 échantillons**, 30 variables numériques (mesures de rayon, texture, périmètre, aire, concavité... calculées à partir d'images de biopsies)
- **Cible** : 357 bénins / 212 malins — déséquilibre modéré

##  Stack technique

- **Langage** : Python
- **ML** : scikit-learn (KNeighborsClassifier, GridSearchCV, StandardScaler)
- **Déploiement** : Gradio
- **Sérialisation** : joblib

##  Pipeline du projet

1. **Chargement** des données via l'API scikit-learn (`load_breast_cancer`)
2. **Split stratifié** (`stratify=y`) et **standardisation** (`StandardScaler`) — indispensable pour KNN
3. **Classification via K-Nearest Neighbors**, hyperparamètres optimisés par `GridSearchCV` avec **`scoring="recall_macro"`** — la recherche est explicitement alignée sur la détection des cas malins, pas sur l'accuracy brute
4. **Sélection de variables par analyse de séparation des classes (z-score)** : identification des variables dont la moyenne diffère le plus entre malin et bénin sur les données standardisées
5. **Sauvegarde** du modèle et déploiement dans une interface Gradio simplifiée (5 variables clés, pas les 30)

##  Résultats

### Optimisation KNN (GridSearchCV, scoring="recall_macro")

- Meilleurs paramètres : `n_neighbors=7`, `weights='uniform'`

### Performance finale

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| Malignant (malin) | 1.00 | 0.93 | 0.96 |
| Benign (bénin) | 0.96 | 1.00 | 0.98 |
| **Accuracy globale** | | | **0.97** |

### Variables les plus discriminantes (analyse par z-score)

Calculées en mesurant l'écart moyen (en écarts-types) entre les groupes malin/bénin sur les données standardisées :

1. `worst concave points`
2. `worst perimeter`
3. `mean concave points`
4. `worst radius`
5. `mean perimeter`
6. `worst area`
7. `mean radius`
8. `mean area`

Les variables liées à la **taille** (rayon, périmètre, aire) et à la **forme irrégulière** (points de concavité) sont les plus discriminantes — cohérent avec la réalité biologique : les tumeurs malignes ont tendance à être plus grosses et à présenter des contours plus irréguliers.

## Application Gradio

Interface simplifiée à **5 sliders** (parmi les variables les plus discriminantes identifiées ci-dessus), qui prédit la probabilité de malignité/bénignité en temps réel — plutôt que d'exposer les 30 variables originales, l'app se concentre sur les plus informatives pour une meilleure expérience utilisateur.

### Lancer l'application

```bash
git clone <repo-url>
cd breast-cancer-predictor
pip install -r requirements.txt
python app_cancer.py
```

## Structure du projet

```
breast-cancer-predictor/
├── app_cancer.py           # Application Gradio
├── cancer_predictor.ipynb  # Notebook d'entraînement et d'analyse
├── cancer_model.pkl        # Modèle KNN entraîné (via GridSearchCV)
├── cancer_scaler.pkl       # StandardScaler fitté
├── moyennes_cancer.csv     # Valeurs moyennes des variables non exposées dans l'app
└── README.md
```

##  Points clés méthodologiques

- **`scoring="recall_macro"` dans GridSearchCV** : dans un contexte médical, rater un cas malin (faux négatif) est nettement plus grave qu'une fausse alerte — la recherche d'hyperparamètres est donc explicitement orientée vers la détection maximale, pas vers l'accuracy globale
- **Sélection de variables par z-score** : approche simple et transparente pour identifier objectivement quelles variables séparent le mieux les deux classes, sans recourir à un modèle complexe
- **Interface Gradio simplifiée intelligemment** : plutôt que d'exposer 30 sliders (irréaliste pour un utilisateur), seules les 5 variables les plus discriminantes sont ajustables ; les autres sont fixées à leur valeur moyenne (`moyennes_cancer.csv`) pour permettre une prédiction complète malgré l'interface réduite
- **Standardisation avant KNN** : essentielle car KNN est basé sur des distances entre points, sensibles à l'échelle des variables

##  Auteur

Aya Chraiet
