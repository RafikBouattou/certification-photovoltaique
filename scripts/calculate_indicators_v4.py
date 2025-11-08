import pandas as pd
from pathlib import Path
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_site_indicators(site_df, site_name):
    """
    Calculates the 7 electrical indicators for a single site.
    """
    indicators = {"site_name": site_name}

    # 1. dc_voltage_stability = std(dcVoltage(V))
    # Seuil d'alerte typique > 150V. Une forte volatilité peut indiquer des problèmes de MPPT ou des ombrages intermittents.
    if 'dcVoltage(V)' in site_df.columns and not site_df['dcVoltage(V)'].empty:
        indicators['dc_voltage_stability'] = site_df['dcVoltage(V)'].std()
    else:
        indicators['dc_voltage_stability'] = np.nan
        logging.warning(f"Site {site_name}: 'dcVoltage(V)' column not found or is empty for dc_voltage_stability.")

    # 2. ac_voltage_balance = mean(max(|L1-L2|, |L2-L3|, |L3-L1|))
    # Seuil d'alerte > 5V. Un déséquilibre peut indiquer des problèmes sur le réseau ou sur l'onduleur.
    ac_volt_cols = ['L1_acVoltage(V)', 'L2_acVoltage(V)', 'L3_acVoltage(V)']
    if all(col in site_df.columns for col in ac_volt_cols) and not site_df[ac_volt_cols].empty:
        diff_l1_l2 = abs(site_df['L1_acVoltage(V)'] - site_df['L2_acVoltage(V)'])
        diff_l2_l3 = abs(site_df['L2_acVoltage(V)'] - site_df['L3_acVoltage(V)'])
        diff_l3_l1 = abs(site_df['L3_acVoltage(V)'] - site_df['L1_acVoltage(V)'])
        indicators['ac_voltage_balance'] = pd.concat([diff_l1_l2, diff_l2_l3, diff_l3_l1], axis=1).max(axis=1).mean()
    else:
        indicators['ac_voltage_balance'] = np.nan
        logging.warning(f"Site {site_name}: AC voltage columns not found or are empty for ac_voltage_balance.")

    # 3. ac_current_harmony = count(peaks > 1.5x moyenne)
    # Seuil d'alerte > 0. Des pics de courant inattendus peuvent signaler des anomalies de charge ou des défauts.
    if 'L1_acCurrent(A)' in site_df.columns and not site_df['L1_acCurrent(A)'].empty:
        mean_ac_current = site_df['L1_acCurrent(A)'].mean()
        if not pd.isna(mean_ac_current) and mean_ac_current > 0:
            indicators['ac_current_harmony'] = (site_df['L1_acCurrent(A)'] > 1.5 * mean_ac_current).sum()
        else:
            indicators['ac_current_harmony'] = 0 # No current, no harmony issues
    else:
        indicators['ac_current_harmony'] = np.nan
        logging.warning(f"Site {site_name}: 'L1_acCurrent(A)' column not found or is empty for ac_current_harmony.")

    # 4. ac_frequency_stability = count(|frequency-50Hz| > 0.5Hz)
    # Seuil d'alerte > 50. La stabilité de la fréquence est un indicateur clé de la qualité du réseau local.
    if 'L1_acFrequency(Hz)' in site_df.columns and not site_df['L1_acFrequency(Hz)'].empty:
        indicators['ac_frequency_stability'] = (abs(site_df['L1_acFrequency(Hz)'] - 50) > 0.5).sum()
    else:
        indicators['ac_frequency_stability'] = np.nan
        logging.warning(f"Site {site_name}: 'L1_acFrequency(Hz)' column not found or is empty for ac_frequency_stability.")

    # 5. power_factor = mean(P_total / S_total)
    # Seuil minimum standard de 0.85-0.9. Un facteur de puissance faible indique une mauvaise efficacité de la transmission de l'énergie.
    # Utilisation de la puissance totale pour un calcul robuste.
    active_power_col = 'totalActivePower(W)'
    reactive_power_cols = ['L1_reactivePower(W)', 'L2_reactivePower(W)', 'L3_reactivePower(W)']
    if active_power_col in site_df.columns and all(col in site_df.columns for col in reactive_power_cols):
        total_reactive_power = site_df[reactive_power_cols].sum(axis=1)
        apparent_power = np.sqrt(site_df[active_power_col]**2 + total_reactive_power**2)
        
        # Éviter la division par zéro
        power_factor_series = site_df[active_power_col].divide(apparent_power).replace([np.inf, -np.inf], np.nan)
        indicators['power_factor'] = power_factor_series.mean()
    else:
        indicators['power_factor'] = np.nan
        logging.warning(f"Site {site_name}: Total active/reactive power columns not found or are empty for power_factor.")

    # 6. generation_irradiance_ratio = mean(power(W) / irradiance)
    # Doit être normalisé par rapport à un benchmark pour être comparable. Un ratio faible indique une sous-performance (soiling, ombrage, etc.).
    if 'power(W)' in site_df.columns and 'Irradiance (W/m2)' in site_df.columns and \
       not site_df['power(W)'].empty and not site_df['Irradiance (W/m2)'].empty:
        ratio_series = site_df['power(W)'] / site_df['Irradiance (W/m2)'].replace(0, np.nan)
        indicators['generation_irradiance_ratio'] = ratio_series.replace([np.inf, -np.inf], np.nan).mean()
    else:
        indicators['generation_irradiance_ratio'] = np.nan
        logging.warning(f"Site {site_name}: 'power(W)' or 'Irradiance (W/m2)' column not found or is empty for generation_irradiance_ratio.")

    # 7. temporal_variability = count(|power_variation| > 30% max_power)
    # Seuil d'alerte > 200. Une forte variabilité peut indiquer une instabilité due à des pannes d'onduleur ou à des conditions météorologiques extrêmes.
    if 'power(W)' in site_df.columns and not site_df['power(W)'].empty:
        max_power = site_df['power(W)'].max()
        if not pd.isna(max_power) and max_power > 0:
            power_variation = site_df['power(W)'].diff().abs()
            indicators['temporal_variability'] = (power_variation > 0.3 * max_power).sum()
        else:
            indicators['temporal_variability'] = 0 # No power, no variability
    else:
        indicators['temporal_variability'] = np.nan
        logging.warning(f"Site {site_name}: 'power(W)' column not found or is empty for temporal_variability.")

    return indicators

def normalize_score(value, min_val, max_val, reverse=False):
    """Min-max normalization to a 0-100 scale."""
    if pd.isna(value) or min_val == max_val:
        return 50 # Return a neutral score if data is missing or range is zero
    
    normalized = (value - min_val) / (max_val - min_val) * 100
    if reverse: # For indicators where lower is better
        return 100 - normalized
    return normalized

def main():
    logging.info("Starting Phase 2: Indicator Calculation.")
    base_path = Path(".")
    cleaned_data_path = base_path / "data" / "processed" / "cleaned_data.csv"
    output_path = base_path / "data" / "results" / "indicators_scores.csv"

    if not cleaned_data_path.exists():
        logging.error(f"Cleaned data file not found: {cleaned_data_path}. Please run Phase 1 first.")
        return

    df_cleaned = pd.read_csv(cleaned_data_path, index_col='Time', parse_dates=True)
    
    if 'site_name' in df_cleaned.columns:
        df_cleaned['site_name'] = df_cleaned['site_name'].astype('category')
    else:
        logging.error("The 'cleaned_data.csv' does not contain a 'site_name' column. Cannot proceed.")
        return

    all_site_names = df_cleaned['site_name'].unique()
    logging.info(f"Found {len(all_site_names)} unique sites.")

    site_indicator_results = []

    for site_name in all_site_names:
        logging.info(f"Calculating indicators for site: {site_name}")
        site_df = df_cleaned[df_cleaned['site_name'] == site_name].copy()
        indicators = calculate_site_indicators(site_df, site_name)
        site_indicator_results.append(indicators)

    indicators_df = pd.DataFrame(site_indicator_results)

    logging.info("Calculating overall_quality_score...")
    
    min_dc_stab, max_dc_stab = indicators_df['dc_voltage_stability'].min(), indicators_df['dc_voltage_stability'].max()
    indicators_df['norm_dc_voltage_stability'] = indicators_df['dc_voltage_stability'].apply(lambda x: normalize_score(x, min_dc_stab, max_dc_stab, reverse=True))

    min_ac_bal, max_ac_bal = indicators_df['ac_voltage_balance'].min(), indicators_df['ac_voltage_balance'].max()
    indicators_df['norm_ac_voltage_balance'] = indicators_df['ac_voltage_balance'].apply(lambda x: normalize_score(x, min_ac_bal, max_ac_bal, reverse=True))

    min_ac_harm, max_ac_harm = indicators_df['ac_current_harmony'].min(), indicators_df['ac_current_harmony'].max()
    indicators_df['norm_ac_current_harmony'] = indicators_df['ac_current_harmony'].apply(lambda x: normalize_score(x, min_ac_harm, max_ac_harm, reverse=True))

    min_ac_freq, max_ac_freq = indicators_df['ac_frequency_stability'].min(), indicators_df['ac_frequency_stability'].max()
    indicators_df['norm_ac_frequency_stability'] = indicators_df['ac_frequency_stability'].apply(lambda x: normalize_score(x, min_ac_freq, max_ac_freq, reverse=True))

    min_pf, max_pf = indicators_df['power_factor'].min(), indicators_df['power_factor'].max()
    indicators_df['norm_power_factor'] = indicators_df['power_factor'].apply(lambda x: normalize_score(x, min_pf, max_pf, reverse=False))

    min_gir, max_gir = indicators_df['generation_irradiance_ratio'].min(), indicators_df['generation_irradiance_ratio'].max()
    indicators_df['norm_generation_irradiance_ratio'] = indicators_df['generation_irradiance_ratio'].apply(lambda x: normalize_score(x, min_gir, max_gir, reverse=False))

    min_tv, max_tv = indicators_df['temporal_variability'].min(), indicators_df['temporal_variability'].max()
    indicators_df['norm_temporal_variability'] = indicators_df['temporal_variability'].apply(lambda x: normalize_score(x, min_tv, max_tv, reverse=True))

    normalized_cols = [col for col in indicators_df.columns if col.startswith('norm_')]
    indicators_df['overall_quality_score'] = indicators_df[normalized_cols].mean(axis=1)
    
    indicators_df = indicators_df.drop(columns=normalized_cols)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Exporting indicators to {output_path}...")
    indicators_df.to_csv(output_path, index=False)
    logging.info("Indicator calculation and export complete.")

if __name__ == "__main__":
    main()
