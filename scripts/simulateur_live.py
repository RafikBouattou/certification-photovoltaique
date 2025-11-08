
import pandas as pd
from pathlib import Path
import numpy as np
import logging
import joblib
from calculate_indicators_v4 import calculate_site_indicators

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_data(model_path, encoder_path, cleaned_data_path):
    """Charge le modèle, l'encodeur et les données nettoyées."""
    try:
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        df_cleaned = pd.read_csv(cleaned_data_path, index_col='Time', parse_dates=True)
        logging.info("Modèle, encodeur et données chargés avec succès.")
        return model, label_encoder, df_cleaned
    except FileNotFoundError as e:
        logging.error(f"Erreur de chargement : {e}. Assurez-vous que les fichiers du modèle et des données existent.")
        return None, None, None

def simulate_live_analysis(df_cleaned, site_name, model, label_encoder):
    """
    Simule l'analyse en temps réel pour un site donné.
    """
    logging.info(f"--- Lancement de la simulation pour le site : {site_name} ---")
    
    site_df = df_cleaned[df_cleaned['site_name'] == site_name].copy()
    
    if site_df.empty:
        logging.error(f"Aucune donnée disponible pour le site {site_name}.")
        return

    # 1. Simuler la réception de données (prendre un échantillon aléatoire de 2 heures)
    sample_size = 24 # 24 points de 5 minutes = 2 heures
    if len(site_df) < sample_size:
        sample_df = site_df
    else:
        sample_df = site_df.sample(n=sample_size)
    
    logging.info(f"Simulation de la réception de {len(sample_df)} points de données...")

    # 2. Calculer les indicateurs sur l'échantillon
    logging.info("Calcul des indicateurs de performance sur les nouvelles données...")
    indicators = calculate_site_indicators(sample_df, site_name)
    df_indicators = pd.DataFrame([indicators])

    # 3. Préparer les données pour la prédiction
    feature_cols = [
        'dc_voltage_stability',
        'ac_voltage_balance',
        'ac_current_harmony',
        'ac_frequency_stability',
        'power_factor',
        'generation_irradiance_ratio',
        'temporal_variability'
    ]
    X_predict = df_indicators[feature_cols].fillna(0) # Remplacer les NaN par 0 pour la démo

    # 4. Faire la prédiction
    logging.info("Le modèle analyse les indicateurs...")
    prediction_encoded = model.predict(X_predict)
    confidence_scores = model.predict_proba(X_predict)
    
    predicted_class = label_encoder.inverse_transform(prediction_encoded)[0]
    confidence = confidence_scores.max() * 100

    # 5. Afficher le bulletin de certification
    print("\n" + "="*60)
    print("          BULLETIN DE CERTIFICATION EN TEMPS RÉEL")
    print("="*60)
    print(f"Site Analysé         : {site_name}")
    print(f"Date de l'analyse      : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)
    print("Indicateurs Clés de Performance (sur échantillon) :")
    for key, value in indicators.items():
        if key != 'site_name':
            print(f"  - {key:<25} : {value:.2f}")
    print("-"*60)
    print("                     V E R D I C T")
    print("-"*60)
    print(f"Classe de Certification Prédite : {predicted_class}")
    print(f"Score de Confiance de l'IA      : {confidence:.2f}%")
    print("="*60 + "\n")


def main():
    # Définir les chemins relatifs au dossier 'certificat'
    base_path = Path('certificat')
    model_path = base_path / 'model' / 'certification_model.joblib'
    encoder_path = base_path / 'model' / 'label_encoder.joblib'
    cleaned_data_path = base_path / 'data' / 'processed' / 'cleaned_data.csv'
    
    # Créer les dossiers pour le modèle s'ils n'existent pas
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Note: Le modèle et l'encodeur doivent être créés par le script d'entraînement.
    # Pour cette simulation, nous allons essayer de les charger.
    # Si ce script est exécuté avant l'entraînement, il faudra générer ces fichiers.
    
    model, label_encoder, df_cleaned = load_model_and_data(model_path, encoder_path, cleaned_data_path)
    
    if model is None:
        logging.error("Impossible de lancer le simulateur. Veuillez exécuter le script d'entraînement pour générer 'certification_model.joblib' et 'label_encoder.joblib' dans le dossier 'certificat/model/'.")
        return

    site_names = df_cleaned['site_name'].unique()

    while True:
        print("Sites disponibles pour l'analyse en temps réel :")
        for i, name in enumerate(site_names):
            print(f"  {i+1}. {name}")
        print("  0. Quitter")
        
        try:
            choice = input("Veuillez choisir le numéro d'un site à analyser : ")
            choice_idx = int(choice) - 1
            
            if choice_idx == -1:
                print("Arrêt du simulateur.")
                break
            
            if 0 <= choice_idx < len(site_names):
                selected_site = site_names[choice_idx]
                simulate_live_analysis(df_cleaned, selected_site, model, label_encoder)
            else:
                print("\n*** Choix invalide, veuillez réessayer. ***\n")
        except ValueError:
            print("\n*** Veuillez entrer un numéro valide. ***\n")
        except Exception as e:
            logging.error(f"Une erreur inattendue est survenue : {e}")
            break

if __name__ == "__main__":
    # Note: Ce script dépend de l'exécution préalable du script d'entraînement
    # pour générer le modèle. Nous devons ajuster le script d'entraînement pour sauvegarder le modèle.
    print("Lancement du simulateur de certification en temps réel.")
    print("Assurez-vous que le modèle ('certification_model.joblib') et l'encodeur ('label_encoder.joblib') sont présents.")
    main()
