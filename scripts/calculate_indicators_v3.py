import pandas as pd
from pathlib import Path
import numpy as np

def load_csv_files(directory_path, file_pattern):
    """
    Loads all CSV files matching a pattern in a directory into a single DataFrame.
    """
    path = Path(directory_path)
    files = list(path.glob(file_pattern))
    if not files:
        print(f"No files found for pattern {file_pattern} in {directory_path}")
        return pd.DataFrame()
    df_list = [pd.read_csv(file) for file in files]
    return pd.concat(df_list, ignore_index=True)

def calculate_indicators(df):
    """
    Calculates the 7 electrical indicators for the given DataFrame.
    NOTE: The following calculations are placeholders.
    Please replace them with the correct formulas.
    """
    indicators = {}

    # Placeholder calculations
    indicators['dc_voltage_stability'] = df.groupby('site_name')['dc_voltage'].std()
    indicators['ac_voltage_balance'] = df.groupby('site_name')['ac_voltage'].std() # Placeholder
    indicators['ac_current_harmony'] = df.groupby('site_name')['ac_current'].std() # Placeholder
    indicators['ac_frequency_stability'] = df.groupby('site_name')['frequency'].std()
    indicators['power_factor'] = df.groupby('site_name')['power_factor'].mean()
    indicators['generation_irradiance_ratio'] = df.groupby('site_name')['power'].sum() / df.groupby('site_name')['irradiance'].sum()
    indicators['temporal_variability'] = df.groupby('site_name')['power'].std() # Placeholder

    # Combine indicators into a single DataFrame
    indicators_df = pd.DataFrame(indicators)
    return indicators_df.reset_index()

def main():
    """
    Main function to execute the data processing pipeline.
    """
    # --- Configuration ---
    # TODO: User - Please update these paths to the correct directories
    inverter_data_path = Path("C:/Users/Flipper/Dataset/inverters")
    site_data_path = Path("C:/Users/Flipper/Dataset/sites")
    weather_data_path = Path("C:/Users/Flipper/Dataset/weather")
    output_path = Path("C:/Users/Flipper/results.csv")

    # --- Load Data ---
    print("Loading inverter data...")
    inverter_df = load_csv_files(inverter_data_path, "*.csv")
    print(f"Loaded {len(inverter_df)} rows from inverter data.")

    print("Loading site data...")
    site_df = load_csv_files(site_data_path, "*.csv")
    print(f"Loaded {len(site_df)} rows from site data.")

    print("Loading weather data...")
    weather_df = load_csv_files(weather_data_path, "*.csv")
    print(f"Loaded {len(weather_df)} rows from weather data.")

    # --- Merge Data ---
    # TODO: User - Implement the merging logic based on your data structure.
    # This is a placeholder merge.
    print("Merging data...")
    # Assuming 'timestamp' and 'site_name' are common columns for merging
    merged_df = pd.merge(inverter_df, site_df, on=['timestamp', 'site_name'], how='left')
    merged_df = pd.merge(merged_df, weather_df, on=['timestamp'], how='left')
    print("Data merged.")

    # --- Calculate Indicators ---
    if not merged_df.empty:
        print("Calculating indicators...")
        indicators_df = calculate_indicators(merged_df)
        print("Indicators calculated.")

        # --- Export Results ---
        print(f"Exporting results to {output_path}...")
        # Define the required columns
        output_columns = [
            "site_name",
            "dc_voltage_stability",
            "ac_voltage_balance",
            "ac_current_harmony",
            "ac_frequency_stability",
            "power_factor",
            "generation_irradiance_ratio",
            "temporal_variability"
        ]
        # Ensure all columns are present, fill with NaN if not
        for col in output_columns:
            if col not in indicators_df.columns:
                indicators_df[col] = np.nan

        indicators_df = indicators_df[output_columns]
        indicators_df.to_csv(output_path, index=False)
        print("Results exported successfully.")
    else:
        print("Merged DataFrame is empty. Cannot calculate indicators.")

if __name__ == "__main__":
    main()
