import pandas as pd
from pathlib import Path
import numpy as np

def load_and_process_data(base_path, relative_path_pattern, site_name_extractor=None):
    """
    Loads CSV files, adds a site_name column, and concatenates them.
    """
    path = Path(base_path)
    files = list(path.glob(relative_path_pattern))
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        if site_name_extractor:
            site_name = site_name_extractor(file.name)
            df['site_name'] = site_name
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def main():
    """
    Main function to perform data cleaning and preparation.
    """
    # Path to the original raw dataset, relative to the 'certificat' directory
    base_path = Path("../Dataset")
    # Output path for the processed data within the new project structure
    output_path = Path("data/processed/cleaned_data.csv")

    # 1. Load Data
    print("Loading data...")
    inverter_df = load_and_process_data(base_path, "Time series dataset/PV generation dataset/PV stations with panel level optimizer/Inverter level dataset/*.csv", lambda x: x.replace('_Inverter.csv', ''))
    site_df_1 = load_and_process_data(base_path, "Time series dataset/PV generation dataset/PV stations with panel level optimizer/Site level dataset/*.csv", lambda x: x.replace('.csv', ''))
    site_df_2 = load_and_process_data(base_path, "Time series dataset/PV generation dataset/PV stations without panel level optimizer/Site level dataset/*.csv", lambda x: x.replace('.csv', ''))
    weather_df = load_and_process_data(base_path, "Time series dataset/Meteorological dataset/Irradiance/*.csv")

    site_df = pd.concat([site_df_1, site_df_2], ignore_index=True)
    print("Data loaded.")

    # Prepare for merge
    inverter_df['Time'] = pd.to_datetime(inverter_df['Time'], format='mixed')
    site_df['Time'] = pd.to_datetime(site_df['Time'], format='mixed')
    weather_df['Time'] = pd.to_datetime(weather_df['Time'], format='%Y/%m/%d %H:%M')

    cleaned_dfs = []
    all_site_names = pd.concat([inverter_df['site_name'], site_df['site_name']]).unique()

    for site_name in all_site_names:
        print(f"Processing site: {site_name}...")
        site_inverter_df = inverter_df[inverter_df['site_name'] == site_name].copy()
        site_site_df = site_df[site_df['site_name'] == site_name].copy()

        site_merged_df = pd.merge(site_inverter_df, site_site_df, on=['site_name', 'Time'], how='outer')
        site_merged_df = pd.merge(site_merged_df, weather_df, on='Time', how='left')

        site_merged_df = site_merged_df.set_index('Time')
        numeric_cols = site_merged_df.select_dtypes(include=np.number).columns
        site_merged_df[numeric_cols] = site_merged_df[numeric_cols].resample('1min').mean()
        site_merged_df = site_merged_df.interpolate(method='linear')

        voltage_cols = [col for col in site_merged_df.columns if 'Voltage' in col]
        for col in voltage_cols:
            site_merged_df[col] = site_merged_df[col].clip(0, 1000)
        current_cols = [col for col in site_merged_df.columns if 'Current' in col]
        for col in current_cols:
            site_merged_df[col] = site_merged_df[col].clip(0, 100)
        freq_cols = [col for col in site_merged_df.columns if 'Frequency' in col]
        for col in freq_cols:
            site_merged_df[col] = site_merged_df[col].clip(45, 55)

        site_merged_df['site_name'] = site_name
        cleaned_dfs.append(site_merged_df)

    print("Concatenating all cleaned site data...")
    final_cleaned_df = pd.concat(cleaned_dfs)
    print("All site data concatenated.")

    # 5. Export cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting cleaned data to {output_path}...")
    final_cleaned_df.to_csv(output_path)
    print("Cleaned data exported successfully.")

if __name__ == "__main__":
    main()