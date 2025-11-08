import pandas as pd
from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting Phase 4: Certification Prediction with Model Comparison and SHAP.")
    
    # Paths are now relative to the project root 'certificat/'
    base_path = Path(".")
    labeled_data_path = base_path / "data" / "processed" / "balanced_training_data.csv"
    all_indicators_path = base_path / "data" / "results" / "indicators_scores.csv"
    predictions_output_path = base_path / "data" / "results" / "certification_predictions.csv"
    model_dir = base_path / "model"
    reports_dir = base_path / "reports"

    # Create model and reports directories if they don't exist
    model_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "best_certification_model.joblib"
    encoder_path = model_dir / "label_encoder.joblib"

    # 1. Load labeled data
    if not labeled_data_path.exists():
        logging.error(f"Labeled data file not found: {labeled_data_path}. Please ensure it exists.")
        return
    
    df_labeled = pd.read_csv(labeled_data_path)
    df_labeled = df_labeled.dropna(subset=['statut_certification'])
    if df_labeled.empty:
        logging.error(f"No valid labeled data found in {labeled_data_path}.")
        return

    logging.info(f"Loaded {len(df_labeled)} labeled samples from {labeled_data_path}.")

    # 2. Load all indicators for prediction
    if not all_indicators_path.exists():
        logging.error(f"All indicators file not found: {all_indicators_path}. Please run Phase 2 first.")
        return
    
    df_all_indicators = pd.read_csv(all_indicators_path)
    logging.info(f"Loaded {len(df_all_indicators)} sites for prediction from {all_indicators_path}.")

    feature_cols = [
        'dc_voltage_stability',
        'ac_voltage_balance',
        'ac_current_harmony',
        'ac_frequency_stability',
        'power_factor',
        'generation_irradiance_ratio',
        'temporal_variability'
    ]
    target_col = 'statut_certification'

    X = df_labeled[feature_cols]
    y_raw = df_labeled[target_col]
    X = X.fillna(X.median())
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    logging.info(f"Encoded labels: {list(label_encoder.classes_)}")
    
    # --- Define Models ---
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }

    results = {}
    best_model = None
    best_accuracy = 0.0

    # --- Train and Evaluate Models ---
    for model_name, model in models.items():
        logging.info(f"--- Evaluating {model_name} ---")
        
        # Cross-validation
        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        mean_accuracy = np.mean(accuracy_scores)
        results[model_name] = mean_accuracy
        logging.info(f"Mean Accuracy from CV for {model_name}: {mean_accuracy:.4f}")

        # Train final model on all data
        model.fit(X, y)
        
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_model = model
            logging.info(f"New best model: {model_name} with accuracy: {best_accuracy:.4f}")

    # --- Save the best model and encoder ---
    logging.info(f"Saving best model ({best_model.__class__.__name__}) to {model_path}")
    joblib.dump(best_model, model_path)
    logging.info(f"Saving label encoder to {encoder_path}")
    joblib.dump(label_encoder, encoder_path)

    # --- Prediction on all sites with the best model ---
    logging.info("Making predictions on all sites with the best model...")
    X_predict = df_all_indicators[feature_cols]
    X_predict = X_predict.fillna(X.median())

    predicted_labels_encoded = best_model.predict(X_predict)
    predicted_classes = label_encoder.inverse_transform(predicted_labels_encoded)
    confidence_scores = best_model.predict_proba(X_predict).max(axis=1)

    df_predictions = df_all_indicators.copy()
    df_predictions['predicted_class'] = predicted_classes
    df_predictions['confidence_score'] = confidence_scores

    output_cols = ['site_name'] + feature_cols + ['predicted_class', 'confidence_score']
    df_predictions = df_predictions[output_cols]

    logging.info(f"Exporting predictions to {predictions_output_path}...")
    df_predictions.to_csv(predictions_output_path, index=False)
    logging.info("Predictions exported successfully.")

    # --- Generate and save reports for the best model ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(label_encoder.classes_)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {best_model.__class__.__name__}')
    plt.savefig(reports_dir / "confusion_matrix.png")
    plt.close()
    logging.info(f"Confusion Matrix saved to {reports_dir / 'confusion_matrix.png'}.")

    # Feature Importance (for RandomForest and XGBoost)
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=feature_importances.index)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance from {best_model.__class__.__name__}')
        plt.tight_layout()
        plt.savefig(reports_dir / "feature_importance.png")
        plt.close()
        logging.info(f"Feature Importance plot saved to {reports_dir / 'feature_importance.png'}.")

    # --- SHAP Analysis (for tree-based models) ---
    if isinstance(best_model, (RandomForestClassifier, XGBClassifier)):
        logging.info("Performing SHAP analysis...")
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(f'SHAP Summary Plot for {best_model.__class__.__name__}')
        plt.savefig(reports_dir / "shap_summary.png")
        plt.close()
        logging.info(f"SHAP summary plot saved to {reports_dir / 'shap_summary.png'}.")

    logging.info("Phase 4 completed.")

if __name__ == "__main__":
    main()