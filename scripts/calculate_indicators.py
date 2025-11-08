
import pandas as pd
import numpy as np

def calculate_indicators():
    # Load inverter data
    inverter_df = pd.read_csv(r"C:\Users\Flipper\data-cleaner-frontend\temp_dataset\Dataset\Time series dataset\PV generation dataset\PV stations with panel level optimizer\Inverter level dataset\Indoor Sports Centre_Inverter.csv")

    # Load and merge irradiance data
    irradiance_2021_df = pd.read_csv(r"C:\Users\Flipper\data-cleaner-frontend\temp_dataset\Dataset\Time series dataset\Meteorological dataset\Irradiance\Irradiance_2021.csv")
    irradiance_2022_df = pd.read_csv(r"C:\Users\Flipper\data-cleaner-frontend\temp_dataset\Dataset\Time series dataset\Meteorological dataset\Irradiance\Irradiance_2022.csv")
    irradiance_2023_df = pd.read_csv(r"C:\Users\Flipper\data-cleaner-frontend\temp_dataset\Dataset\Time series dataset\Meteorological dataset\Irradiance\Irradiance_2023.csv")
    irradiance_df = pd.concat([irradiance_2021_df, irradiance_2022_df, irradiance_2023_df])

    # Convert 'Time' columns to datetime objects
    inverter_df['Time'] = pd.to_datetime(inverter_df['Time'])
    irradiance_df['Time'] = pd.to_datetime(irradiance_df['Time'])

    # Set 'Time' as index
    inverter_df.set_index('Time', inplace=True)
    irradiance_df.set_index('Time', inplace=True)

    # Resample irradiance data and merge
    irradiance_resampled_df = irradiance_df.resample('5T').mean()
    merged_df = pd.merge(inverter_df, irradiance_resampled_df, left_index=True, right_index=True, how='inner')

    # --- Indicator Calculations ---
    indicators = {}

    # 1. DC Voltage Stability (Standard Deviation)
    indicators['dc_voltage_stability'] = merged_df['dcVoltage(V)'].std()

    # 2. AC Voltage Balance (Mean of Max Differences)
    voltage_phases = ['L1_acVoltage(V)', 'L2_acVoltage(V)', 'L3_acVoltage(V)']
    merged_df['voltage_diff'] = merged_df[voltage_phases].max(axis=1) - merged_df[voltage_phases].min(axis=1)
    indicators['ac_voltage_balance'] = merged_df['voltage_diff'].mean()

    # 3. AC Current Harmony (Z-score outlier count)
    current_phases = ['L1_acCurrent(A)', 'L2_acCurrent(A)', 'L3_acCurrent(A)']
    outlier_count = 0
    for phase in current_phases:
        z_scores = np.abs((merged_df[phase] - merged_df[phase].mean()) / merged_df[phase].std())
        outlier_count += (z_scores > 3).sum()
    indicators['ac_current_harmony'] = outlier_count

    # 4. AC Frequency Stability (Count outside range)
    freq_phases = ['L1_acFrequency(Hz)', 'L2_acFrequency(Hz)', 'L3_acFrequency(Hz)']
    freq_outside_range = 0
    for phase in freq_phases:
        freq_outside_range += ((merged_df[phase] < 49.5) | (merged_df[phase] > 50.5)).sum()
    indicators['ac_frequency_stability'] = freq_outside_range

    # 5. Power Factor
    total_reactive_power = merged_df['L1_reactivePower(W)'] + merged_df['L2_reactivePower(W)'] + merged_df['L3_reactivePower(W)']
    apparent_power = np.sqrt(merged_df['totalActivePower(W)']**2 + total_reactive_power**2)
    # Avoid division by zero
    power_factor = merged_df['totalActivePower(W)'] / apparent_power
    indicators['power_factor'] = power_factor.mean()

    # 6. Generation/Irradiance Ratio
    # Avoid division by zero
    merged_df['gen_irradiance_ratio'] = merged_df['totalActivePower(W)'] / merged_df['Irradiance (W/m2)']
    indicators['generation_irradiance_ratio'] = merged_df['gen_irradiance_ratio'].mean()

    # 7. Temporal Variability (Count of abrupt ramps)
    power_diff = merged_df['totalActivePower(W)'].diff()
    # Threshold: 50% of max power
    threshold = merged_df['totalActivePower(W)'].max() * 0.5
    indicators['temporal_variability'] = (power_diff.abs() > threshold).sum()

    # --- Save and Print Indicators ---
    indicators_df = pd.DataFrame([indicators])
    indicators_df.to_csv(r"C:\Users\Flipper\indoor_sports_centre_indicators.csv", index=False)

    print("Indicators for Indoor Sports Centre:")
    print(indicators_df.to_json(orient='records', lines=True))

if __name__ == "__main__":
    calculate_indicators()
