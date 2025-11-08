# ⚡️ Système de Diagnostic Prédictif et de Certification pour Installations Photovoltaïques ⚡️

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green.svg)](https://github.com/bouattou-rafik/certification-photovoltaique)
> **Auteur :** Bouattou Rafik  
> **Organisation :** LABSU

## 🚀 Résumé Exécutif : Optimisation et Conformité des Actifs PV par l'IA

Ce projet déploie un pipeline de Machine Learning avancé pour transformer la gestion des installations photovoltaïques. Il offre une solution proactive pour **diagnostiquer la performance, prédire la conformité et optimiser la maintenance** des centrales solaires. En exploitant les données de séries temporelles et des algorithmes d'IA, nous permettons aux entreprises de **maximiser leur retour sur investissement (ROI)**, de **réduire les coûts opérationnels** et d'**assurer une conformité réglementaire**.

## 🎯 Problématique : Les Défis de la Gestion des Actifs PV

La croissance du secteur photovoltaïque s'accompagne de défis majeurs :
*   **Coûts de Maintenance Élevés :** Inspections manuelles coûteuses, lentes et réactives.
*   **Perte d'Efficacité :** Anomalies non détectées réduisant la production et le ROI.
*   **Risques de Non-Conformité :** Complexité du respect des normes réglementaires.
*   **Manque de Visibilité :** Difficulté à obtenir une vue d'ensemble de la santé des installations.

## 💡 Solution : Un Système Prédictif de Conformité PV Basé sur l'IA

Notre système est une plateforme analytique qui automatise l'évaluation de la conformité et la détection des dysfonctionnements.

### Valeur Ajoutée
*   **Réduction des Coûts :** Passage à une maintenance prédictive.
*   **Optimisation de la Production :** Identification rapide des facteurs limitants.
*   **Fiabilité Améliorée :** Prévention des pannes majeures.
*   **Conformité Renforcée :** Évaluation objective et traçable.
*   **Aide à la Décision :** Rapports clairs pour ingénieurs et décideurs.

## 📊 Données Utilisées

Ce projet s'appuie sur un jeu de données public de haute qualité pour garantir la reproductibilité et la pertinence des résultats.

*   **Titre :** A HIGH-RESOLUTION THREE-YEAR DATASET SUPPORTING ROOFTOP PHOTOVOLTAICS (PV) GENERATION ANALYTICS
*   **Source :** Dryad
*   **DOI :** [10.5061/dryad.m37pvmd99](https://doi.org/10.5061/dryad.m37pvmd99)
*   **Description :** Le jeu de données contient des mesures de production d'énergie photovoltaïque et des données météorologiques provenant de 60 stations PV sur le campus de l'Université des sciences et technologies de Hong Kong, collectées sur une période de trois ans (2021-2023) à des intervalles de 1 à 5 minutes.

**Citation :**
> Lin, Jian, et al. (2024). A HIGH-RESOLUTION THREE-YEAR DATASET SUPPORTING ROOFTOP PHOTOVOLTAICS (PV) GENERATION ANALYTICS [Dataset]. Dryad. https://doi.org/10.5061/dryad.m37pvmd99

## ⚙️ Approche Technique

Notre pipeline de Machine Learning intègre :

1.  **Ingestion et Traitement de Signal Temporel :** Consolidation et fiabilisation des données hétérogènes (`Pandas` pour le rééchantillonnage et l'interpolation).
2.  **Ingénierie des Indicateurs :** Transformation des données brutes en 7 indicateurs de performance clés (stabilité de tension, équilibre de phase, etc.).
3.  **Étiquetage Automatisé :** Création automatique d'étiquettes de conformité (`CONFORME`, `NON_CONFORME_MINEUR`) basées sur un score de qualité.
4.  **Rééquilibrage des Classes (SMOTE) :** Gestion des jeux de données déséquilibrés pour une meilleure détection des anomalies rares.
5.  **Sélection de Modèles de Classification :** Comparaison de **Random Forest**, **SVC**, et **XGBoost** par validation croisée stratifiée pour sélectionner le plus performant.
6.  **Interprétabilité (SHAP) :** Explication des prédictions pour rendre le système transparent et digne de confiance.

## ✨ Fonctionnalités Clés

*   **Diagnostic Prédictif Avancé :** Anticipez les problèmes de conformité avec l'IA.
*   **Génération de Rapports Automatisés :** Créez des rapports HTML interactifs et des analyses visuelles (`confusion_matrix.png`, `feature_importance.png`, `shap_summary.png`) pour chaque installation.
*   **Dashboard Interactif :** Visualisez la santé globale du parc d'installations via un tableau de bord dynamique.
*   **Simulation en Temps Réel :** Testez et démontrez la réactivité du système avec des données simulées en direct.
*   **Modélisation Avancée :** Inclut des modèles classiques (XGBoost) et une exploration vers des architectures de Deep Learning pour séries temporelles (**Temporal Fusion Transformer**).
*   **Pipeline Robuste et Scalable :** Conçu pour gérer de grands volumes de données.

## 📂 Structure du Projet

```
certification-photovoltaique/
├── Dataset/               # Données brutes (à télécharger depuis Dryad)
│   └── ...
└── certificat/
    ├── data/                # Données traitées et résultats
    ├── model/               # Modèles entraînés
    ├── reports/             # Rapports générés (HTML, images)
    ├── scripts/             # Scripts Python du projet
    │   ├── prepare_data.py            # 1. Préparation des données
    │   ├── calculate_indicators_v4.py # 2. Calcul des indicateurs
    │   ├── generate_labeling_template.py # 3. Étiquetage automatique
    │   ├── balance_classes.py         # 4. Rééquilibrage des classes
    │   ├── train_and_predict_certification.py # 5. Entraînement et prédiction
    │   ├── generate_report.py         # 6. Génération de rapports HTML
    │   ├── generateur_dashboard.py    # Outil: Dashboard interactif
    │   ├── simulateur_live.py       # Outil: Simulation temps réel
    │   └── TFT.py                   # Exploration: Modèle Deep Learning
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

## ⚡ Démarrage Rapide

### Prérequis

*   Python 3.9+
*   Git

### Installation

1.  **Télécharger les données :**
    *   Rendez-vous sur la page du dataset : [Dryad Dataset](https://doi.org/10.5061/dryad.m37pvmd99).
    *   Téléchargez le fichier `dataset.zip` et extrayez son contenu dans un dossier nommé `Dataset`.

2.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/bouattou-rafik/certification-photovoltaique.git
    ```
    Assurez-vous que le dossier `Dataset` que vous venez de créer se trouve au même niveau que le dossier `certification-photovoltaique`.

3.  **Naviguer et créer l'environnement :**
    ```bash
    cd certification-photovoltaique/certificat
    python -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```

4.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

### Utilisation du Pipeline

Exécutez les scripts dans l'ordre depuis la racine du dossier `certificat` :

1.  **Préparation des données :** `python scripts/prepare_data.py`
2.  **Calcul des indicateurs :** `python scripts/calculate_indicators_v4.py`
3.  **Étiquetage automatique :** `python scripts/generate_labeling_template.py`
4.  **Rééquilibrage des classes :** `python scripts/balance_classes.py`
5.  **Entraînement et Prédiction :** `python scripts/train_and_predict_certification.py`
6.  **Génération des rapports :** `python scripts/generate_report.py`

### Outils Additionnels

*   **Lancer le dashboard interactif :**
    ```bash
    python scripts/generateur_dashboard.py
    ```
*   **Lancer le simulateur temps réel :**
    ```bash
    python scripts/simulateur_live.py
    ```

## 📊 Résultats et Visualisations

*   **`data/results/certification_predictions.csv` :** Prédictions de conformité pour chaque site.
*   **`model/best_certification_model.joblib` :** Modèle entraîné et prêt à l'emploi.
*   `reports/` : Rapports HTML interactifs par site et visualisations (`confusion_matrix.png`, `feature_importance.png`, `shap_summary.png`).

![Feature Importance](reports/feature_importance.png)

## 🛠️ Technologies

*   **Langage :** Python
*   **Data Science :** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn
*   **Interprétabilité :** SHAP
*   **Visualisation :** Matplotlib, Seaborn, Plotly (pour les rapports interactifs)
*   **Deep Learning :** PyTorch/TensorFlow (via `TFT.py`)

## 📈 Perspectives d'Évolution

*   **Déploiement API :** Intégration du modèle via une API RESTful (FastAPI) pour un diagnostic continu.
*   **Interface Utilisateur (UI/UX) :** Développement d'une application web (Streamlit, Dash) pour une interaction facilitée.
*   **Scalabilité Cloud :** Optimisation pour un déploiement sur AWS, Azure, ou GCP.

## 🤝 Auteur

*   **Bouattou Rafik** - Développeur Principal ([LABSU](https://github.com/bouattou-rafik))

## 📄 Licence

Ce projet est distribué sous la **Licence MIT**.