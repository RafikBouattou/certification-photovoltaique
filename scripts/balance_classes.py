import pandas as pd
from pathlib import Path
import numpy as np
import logging

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting Class Rebalancing (SMOTE).")
    
    base_path = Path(".")
    input_labeled_data_path = base_path / "data" / "results" / "labellisation_auto_complete.csv"
    output_balanced_data_path = base_path / "data" / "processed" / "balanced_training_data.csv"

    if not input_labeled_data_path.exists():
        logging.error(f"Labeled data file not found: {input_labeled_data_path}. Please ensure it exists.")
        return
    
    df_labeled = pd.read_csv(input_labeled_data_path)
    df_labeled = df_labeled.dropna(subset=['statut_certification']) # Drop rows with missing labels
    if df_labeled.empty:
        logging.error("No valid labeled data found after dropping rows with missing labels.")
        return

    logging.info(f"Loaded {len(df_labeled)} labeled samples from {input_labeled_data_path}.")

    # Analyze initial class distribution
    logging.info("Initial class distribution:")
    logging.info(df_labeled['statut_certification'].value_counts())

    # Define features (X) and target (y)
    feature_cols = [
        'dc_voltage_stability',
        'ac_voltage_balance',
        'ac_current_harmony',
        'ac_frequency_stability',
        'power_factor',
        'generation_irradiance_ratio',
        'temporal_variability',
        'overall_quality_score' # Include overall_quality_score as a feature
    ]
    target_col = 'statut_certification'

    X = df_labeled[feature_cols]
    y_raw = df_labeled[target_col]

    # Handle NaN values in features (fill with median before SMOTE)
    X = X.fillna(X.median())
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    logging.info(f"Encoded labels: {list(label_encoder.classes_)}")

    # Apply SMOTE
    logging.info("Applying SMOTE for class rebalancing...")
    
    # Determine sampling strategy: each minority class to have at least 30 samples
    class_counts = pd.Series(y).value_counts()
    sampling_strategy = {}
    for class_label, count in class_counts.items():
        if count < 30:
            sampling_strategy[class_label] = 30
        else:
            sampling_strategy[class_label] = count # Keep majority class as is

    # SMOTE only on minority classes
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=1)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    logging.info("SMOTE applied.")

    # Decode oversampled labels
    y_resampled_raw = label_encoder.inverse_transform(y_resampled)

    # Combine oversampled features and labels into a new DataFrame
    df_balanced = pd.DataFrame(X_resampled, columns=feature_cols)
    df_balanced[target_col] = y_resampled_raw

    # Analyze final class distribution
    logging.info("Final class distribution after SMOTE:")
    logging.info(df_balanced[target_col].value_counts())

    # Export balanced training data
    output_balanced_data_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Exporting balanced training data to {output_balanced_data_path}...")
    df_balanced.to_csv(output_balanced_data_path, index=False)
    logging.info("Balanced training data exported successfully.")

if __name__ == "__main__":
    main()
