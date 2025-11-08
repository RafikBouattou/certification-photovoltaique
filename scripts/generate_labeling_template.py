import pandas as pd
from pathlib import Path
import numpy as np

def main():
    print("Starting Phase 3A (Automated Labeling): Generating Complete Labeled Template.")
    
    base_path = Path(".")
    input_path = base_path / "data" / "results" / "indicators_scores.csv"
    output_path = base_path / "data" / "results" / "labellisation_auto_complete.csv"

    if not input_path.exists():
        print(f"Error: Indicators scores file not found: {input_path}. Please run Phase 2 first.")
        return

    # 1. Load 'indicators_scores.csv'
    df_indicators = pd.read_csv(input_path)
    print(f"Loaded {len(df_indicators)} rows from {input_path}.")

    # Apply the labeling rules based on overall_quality_score
    def apply_labeling_rules(score):
        if score > 75:
            return "CONFORME"
        elif 60 <= score <= 75:
            return "CONFORME_SOUS_RESERVE"
        elif 40 <= score < 60:
            return "NON_CONFORME_MINEUR"
        elif score < 40:
            return "NON_CONFORME_CRITIQUE"
        else:
            return np.nan # Handle cases where score might be NaN

    df_indicators['statut_certification'] = df_indicators['overall_quality_score'].apply(apply_labeling_rules)
    
    # Ensure all required columns are present for the output
    required_cols = [
        'site_name',
        'dc_voltage_stability',
        'ac_voltage_balance',
        'ac_current_harmony',
        'ac_frequency_stability',
        'power_factor',
        'generation_irradiance_ratio',
        'temporal_variability',
        'overall_quality_score',
        'statut_certification' # Include the newly created label column
    ]
    
    # Check if all required columns exist in df_indicators
    missing_cols = [col for col in required_cols if col not in df_indicators.columns]
    if missing_cols:
        print(f"Error: Missing required columns after labeling: {', '.join(missing_cols)}")
        return

    df_labeled_auto = df_indicators[required_cols].copy()

    print(f"Automatically labeled {len(df_labeled_auto)} sites based on overall_quality_score.")
    print("Distribution of generated labels:")
    print(df_labeled_auto['statut_certification'].value_counts())

    # Export the automatically labeled data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting automatically labeled template to {output_path}...")
    df_labeled_auto.to_csv(output_path, index=False)
    print("Automatically labeled template exported successfully.")

if __name__ == "__main__":
    main()
