
import pandas as pd
from pathlib import Path
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_status_color(verdict):
    if verdict == "CONFORME":
        return "#2ca02c"  # Vert
    elif verdict == "CONFORME_SOUS_RESERVE":
        return "#ff7f0e"  # Orange
    elif verdict == "NON_CONFORME_MINEUR":
        return "#d62728"  # Rouge
    elif verdict == "NON_CONFORME_CRITIQUE":
        return "#9467bd"  # Violet
    return "#7f7f7f" # Gris par défaut

def generate_dashboard(site_name, df_cleaned, df_predictions, df_indicators):
    """Génère un tableau de bord HTML interactif v2 pour un site spécifique."""
    logging.info(f"Début de la génération du tableau de bord v2 pour : {site_name}")

    # --- 1. Extraire les données spécifiques au site ---
    site_data = df_cleaned[df_cleaned['site_name'] == site_name]
    site_prediction = df_predictions[df_predictions['site_name'] == site_name].iloc[0]
    site_indicators = df_indicators[df_indicators['site_name'] == site_name].iloc[0]

    sample_data = site_data.tail(1000)
    latest_power = sample_data['totalActivePower(W)'].iloc[-1] if not sample_data.empty else 0
    daily_energy = site_data['totalActivePower(W)'].sum() * (5/60) / 1000 # Supposant 5 min d'intervalle, conversion en kWh

    # --- 2. Créer les visualisations interactives ---
    logging.info("Création des graphiques interactifs...")
    plotly_template = "plotly_dark"

    fig_prod = go.Figure()
    fig_prod.add_trace(go.Scatter(x=sample_data.index, y=sample_data['totalActivePower(W)'], name='Puissance Active (W)', yaxis='y1', line=dict(color='#ff9900')))
    fig_prod.add_trace(go.Scatter(x=sample_data.index, y=sample_data['Irradiance (W/m2)'], name='Irradiance (W/m²)', yaxis='y2', line=dict(color='#4c78a8', dash='dot')))
    fig_prod.update_layout(title_text="<b>Production & Météo (Échantillon)</b>", template=plotly_template, yaxis=dict(title="Puissance Active (W)"), yaxis2=dict(title="Irradiance (W/m²)", overlaying='y', side='right'), legend=dict(x=0, y=1.1, orientation='h'))
    graph_production_html = fig_prod.to_html(full_html=False, include_plotlyjs='cdn')

    fig_dc = px.line(sample_data, x=sample_data.index, y='dcVoltage(V)', title='<b>Stabilité de la Tension DC</b>', template=plotly_template)
    fig_dc.update_traces(line_color='#54a24b')
    graph_dc_voltage_html = fig_dc.to_html(full_html=False, include_plotlyjs=False)

    fig_ac = px.line(sample_data, x=sample_data.index, y=['L1_acVoltage(V)', 'L2_acVoltage(V)', 'L3_acVoltage(V)'], title='<b>Équilibre des Tensions AC</b>', template=plotly_template)
    graph_ac_balance_html = fig_ac.to_html(full_html=False, include_plotlyjs=False)

    # --- 3. Assembler le rapport HTML ---
    logging.info("Assemblage du rapport HTML v2...")
    verdict = site_prediction['predicted_class']
    confidence = site_prediction['confidence_score'] * 100
    verdict_color = get_status_color(verdict)
    quality_score = site_indicators['overall_quality_score']

    kpi_table_rows = ""
    feature_cols = {
        'dc_voltage_stability': 150, 'ac_voltage_balance': 5, 'ac_current_harmony': 10,
        'ac_frequency_stability': 50, 'power_factor': 0.9, 'generation_irradiance_ratio': 20, # Seuil bas
        'temporal_variability': 200, 'overall_quality_score': 60 # Seuil bas
    }
    for col, threshold in feature_cols.items():
        value = site_indicators[col]
        status_color = '#2ca02c' # Vert (Bon)
        if col in ['power_factor', 'generation_irradiance_ratio', 'overall_quality_score']:
            if value < threshold: status_color = '#d62728' # Rouge (Mauvais)
        else:
            if value > threshold: status_color = '#d62728' # Rouge (Mauvais)
        kpi_table_rows += f"<tr><td>{col.replace('_', ' ').title()}</td><td>{value:.2f}</td><td><span style='color:{status_color}; font-size: 20px;'>●</span></td></tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Rapport de Certification - {site_name}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1a242a; color: #f0f0f0; margin: 0; padding: 20px; }}
            .container {{ max-width: 1400px; margin: auto; }}
            h1, h2 {{ color: #ff9900; border-bottom: 2px solid #444; padding-bottom: 10px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .status-badge {{ background-color: {verdict_color}; color: white; padding: 15px 25px; text-align: center; border-radius: 8px; margin-bottom: 20px; display: inline-block; }}
            .status-title {{ font-size: 28px; font-weight: bold; margin: 0; }}
            .confidence-score {{ font-size: 16px; margin-top: 5px; opacity: 0.9; }}
            .kpi-banner {{ display: flex; justify-content: space-around; gap: 20px; margin-bottom: 30px; }}
            .kpi-card {{ background-color: #2c3e50; padding: 20px; border-radius: 8px; text-align: center; flex-grow: 1; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }}
            .kpi-card h3 {{ margin-top: 0; color: #f0f0f0; font-size: 16px; text-transform: uppercase; opacity: 0.8; }}
            .kpi-card .value {{ font-size: 36px; font-weight: bold; color: #ff9900; }}
            .main-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .grid-item {{ background-color: #2c3e50; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }}
            .kpi-table table {{ width: 100%; border-collapse: collapse; }}
            .kpi-table th, .kpi-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #444; }}
            .kpi-table th {{ background-color: #34495e; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>Tableau de Bord de Certification : {site_name}</h1></div>
            
            <div style="text-align: center;">
                <div class="status-badge">
                    <div class="status-title">{verdict}</div>
                    <div class="confidence-score">Confiance IA : {confidence:.2f}%</div>
                </div>
            </div>

            <div class="kpi-banner">
                <div class="kpi-card"><h3>Puissance Actuelle</h3><div class="value">{latest_power:.2f} W</div></div>
                <div class="kpi-card"><h3>Énergie du Jour (estimée)</h3><div class="value">{daily_energy:.2f} kWh</div></div>
                <div class="kpi-card"><h3>Score de Qualité</h3><div class="value">{quality_score:.1f} / 100</div></div>
            </div>

            <div class="main-grid">
                <div class="grid-item kpi-table"><h2>Indicateurs de Performance</h2><table><tr><th>Indicateur</th><th>Valeur</th><th>Statut</th></tr>{kpi_table_rows}</table></div>
                <div class="grid-item">{graph_production_html}</div>
            </div>

            <div class="main-grid" style="margin-top: 20px;">
                <div class="grid-item">{graph_dc_voltage_html}</div>
                <div class="grid-item">{graph_ac_balance_html}</div>
            </div>
        </div>
    </body>
    </html>
    """

    report_path = Path(f"C:/Users/Flipper/certificat/reports/rapport_certification_{site_name}.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"Tableau de bord v2 généré avec succès : {report_path}")
    return report_path

def main():
    """Fonction principale pour lancer le générateur de tableau de bord."""
    base_path = Path("C:/Users/Flipper/certificat")
    cleaned_data_path = base_path / "data" / "processed" / "cleaned_data.csv"
    predictions_path = base_path / "data" / "results" / "certification_predictions.csv"
    indicators_path = base_path / "data" / "results" / "indicators_scores.csv"

    try:
        df_cleaned = pd.read_csv(cleaned_data_path, index_col='Time', parse_dates=True)
        df_predictions = pd.read_csv(predictions_path)
        df_indicators = pd.read_csv(indicators_path)
        logging.info("Toutes les données nécessaires ont été chargées.")
    except FileNotFoundError as e:
        logging.error(f"Erreur de chargement de fichier : {e}. Veuillez exécuter la pipeline complète.")
        return

    site_names = df_indicators['site_name'].unique()

    while True:
        print("\n" + "="*60)
        print("GÉNÉRATEUR DE TABLEAU DE BORD DE CERTIFICATION v2")
        print("="*60)
        print("Sites disponibles pour l'analyse :")
        for i, name in enumerate(site_names):
            print(f"  {i+1}. {name}")
        print("  0. Quitter")
        
        try:
            choice = input("Veuillez choisir le numéro d'un site à analyser : ")
            choice_idx = int(choice) - 1
            
            if choice_idx == -1:
                print("Arrêt du programme.")
                break
            
            if 0 <= choice_idx < len(site_names):
                selected_site = site_names[choice_idx]
                generate_dashboard(selected_site, df_cleaned, df_predictions, df_indicators)
            else:
                print("\n*** Choix invalide, veuillez réessayer. ***\n")
        except ValueError:
            print("\n*** Veuillez entrer un numéro valide. ***\n")
        except Exception as e:
            logging.error(f"Une erreur inattendue est survenue : {e}")
            break

if __name__ == "__main__":
    main()
