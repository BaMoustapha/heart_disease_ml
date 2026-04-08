# 🫀 Prédiction de Maladies Cardiaques

> Projet individuel — Master 1 IA & Data Science — UMEF University — 2026  
> **Auteur :** Mouhamadou Moustapha BA  
> **Cours :** Généralités sur les IA

---

## 📋 Description

Ce projet implémente une **pipeline complète de Machine Learning** pour prédire la présence de maladies cardiaques à partir de données cliniques. Il s'inscrit dans une démarche d'**aide à la décision médicale par l'IA**, avec un accent particulier sur les métriques adaptées au contexte médical (Recall).

---

## 🎯 Résultats

| Métrique | Score |
|---|---|
| **Accuracy** | 82.0% |
| **Recall** ★ | 87.9% |
| **F1-Score** | 83.8% |
| **AUC-ROC** | 0.872 |
| **Meilleur modèle** | Logistic Regression |

> ★ Le Recall est la métrique prioritaire en médecine — minimiser les faux négatifs (patients malades non détectés).

---

## 📊 Dataset

- **Source :** [Heart Disease Dataset — johnsmith88 (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Origine :** UCI Machine Learning Repository (4 hôpitaux fusionnés)
- **Lignes initiales :** 1 025 → **302 patients uniques** après nettoyage
- **Features :** 13 variables cliniques + 1 cible
- **Cible :** `target` — 1 = Sain / 0 = Malade
- **Valeurs manquantes :** 0

### Features principales

| Feature | Description | Corrélation |
|---|---|---|
| `exang` | Angine d'effort | -0.44 |
| `cp` | Type de douleur thoracique | +0.43 |
| `oldpeak` | Dépression segment ST | -0.43 |
| `thalach` | Fréquence cardiaque max | +0.42 |
| `ca` | Vaisseaux bloqués | -0.41 |

---

## 🏗️ Structure du projet

```
heart_disease_ml/
├── data/
│   └── heart.csv              # Dataset Heart Disease UCI
├── models/
│   ├── best_model.pkl         # Meilleur modèle (Logistic Regression)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
├── output/
│   ├── 01_eda.png             # Distribution + Age + Sexe
│   ├── 02_correlation.png     # Matrice de corrélation
│   ├── 03_comparaison_modeles.png
│   ├── 04_confusion_matrix.png
│   └── 05_roc_curves.png
├── notebook.ipynb             # Pipeline ML complète (Google Colab)
├── dashboard.py               # Dashboard interactif Dash/Plotly
├── requirements.txt           # Dépendances Python
└── render.yaml                # Configuration déploiement Render
```

---

## 🤖 Modèles ML

Trois algorithmes comparés après optimisation **GridSearchCV** (5 folds) :

| Modèle | Accuracy | F1 | Recall | AUC |
|---|---|---|---|---|
| **Logistic Regression** ★ | **0.820** | **0.838** | **0.879** | **0.872** |
| SVM | 0.820 | 0.838 | 0.879 | 0.870 |
| Random Forest | 0.770 | 0.788 | 0.788 | 0.860 |

> ★ La Logistic Regression est retenue comme meilleur modèle — performance identique au SVM mais meilleure interprétabilité (**Occam's Razor**) et probabilités natives pour le simulateur.

---

## 📊 Dashboard interactif

Le dashboard est déployé et accessible en ligne :

🌐 **[https://heart-disease-ml-cni1.onrender.com/](https://heart-disease-ml-cni1.onrender.com/)**

### 4 onglets disponibles

| Onglet | Contenu |
|---|---|
| 🏠 **Vue d'ensemble** | KPIs + Distribution + Corrélation |
| 📊 **Modèles** | Comparaison + Matrice de confusion + ROC |
| 🔮 **Prévision** | Simulateur de risque + Profils sénégalais |
| ℹ️ **À propos** | Auteur + Dataset + Technologies |

### Lancement en local

```bash
pip install -r requirements.txt
python dashboard.py
```

Puis ouvrir → http://localhost:8050

---

## 🛠️ Technologies

| Technologie | Version | Usage |
|---|---|---|
| Python | 3.12 | Langage principal |
| Scikit-learn | 1.6.1 | Modèles ML + Pipeline + GridSearch |
| Pandas | 2.2.3 | Manipulation des données |
| NumPy | 2.1.2 | Calculs numériques |
| Dash | 4.1.0 | Dashboard interactif |
| Plotly | 6.0.1 | Graphiques interactifs |
| Joblib | 1.4.2 | Sauvegarde des modèles |
| Gunicorn | 21.2.0 | Serveur WSGI (Render) |
| Google Colab | — | Développement notebook |

---

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/BaMoustapha/heart_disease_ml.git
cd heart_disease_ml

# Installer les dépendances
pip install -r requirements.txt

# Lancer le dashboard
python dashboard.py
```

---

## ⚖️ Réflexion éthique

- Ce modèle est un **outil d'aide à la décision**, pas un outil de diagnostic médical
- Seul un médecin qualifié peut établir un diagnostic après des examens complets
- **13 faux négatifs** sur le jeu de test — le Recall (87.9%) est priorisé pour les minimiser
- Les données datent de 1988 et proviennent d'une population américaine et européenne — biais potentiel pour d'autres populations

---

## 🌍 Perspectives

- Adapter le modèle à des données africaines pour le dépistage en Afrique subsaharienne
- Intégrer des données supplémentaires (imagerie, génomique)
- Explorer des algorithmes plus avancés (XGBoost, réseaux de neurones)
- Déployer via une API REST accessible aux professionnels de santé

---

## 📚 Références

1. Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease Dataset*. UCI Machine Learning Repository.
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
3. World Health Organization. (2022). *Cardiovascular diseases (CVDs)*. WHO Fact Sheet.

---

*Projet réalisé dans le cadre du cours Généralités sur les IA — UMEF University — 2026*