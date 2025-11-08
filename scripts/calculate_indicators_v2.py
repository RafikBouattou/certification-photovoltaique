
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_indicators_v2(inverter_csv_path_str, irradiance_2021_path_str, irradiance_2022_path_str, irradiance_2023_path_str, output_csv_path_str):
    # Convert string paths to Path objects
    inverter_csv_path = Path(inverter_csv_path_str)
    irradiance_2021_path = Path(irradiance_2021_path_str)
    irradiance_2022_path = Path(irradiance_2022_path_str)
    irradiance_2023_path = Path(irradiance_2023_path_str)
    output_csv_path = Path(output_csv_path_str)

    # Load inverter data
    inverter_df = pd.read_csv(inverter_csv_path)

    # Load irradiance data for all years
    irradiance_2021_df = pd.read_csv(irradiance_2021_path)
    irradiance_2022_df = pd.read_csv(irradiance_2022_path)
    irradiance_2023_df = pd.read_csv(irradiance_2023_path)
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

    # --- Indicator Calculations (V2) ---
    indicators = {}

    # 1. DC Voltage Stability
    indicators['dc_voltage_stability'] = merged_df['dcVoltage(V)'].std()

    # 2. AC Voltage Balance
    voltage_phases = ['L1_acVoltage(V)', 'L2_acVoltage(V)', 'L3_acVoltage(V)']
    merged_df['voltage_diff'] = merged_df[voltage_phases].max(axis=1) - merged_df[voltage_phases].min(axis=1)
    indicators['ac_voltage_balance'] = merged_df['voltage_diff'].mean()

    # 3. AC Current Harmony
    current_phases = ['L1_acCurrent(A)', 'L2_acCurrent(A)', 'L3_acCurrent(A)']
    outlier_count = 0
    for phase in current_phases:
        z_scores = np.abs((merged_df[phase] - merged_df[phase].mean()) / merged_df[phase].std())
        outlier_count += (z_scores > 3).sum()
    indicators['ac_current_harmony'] = outlier_count

    # 4. AC Frequency Stability
    freq_phases = ['L1_acFrequency(Hz)', 'L2_acFrequency(Hz)', 'L3_acFrequency(Hz)']
    freq_outside_range = 0
    for phase in freq_phases:
        freq_outside_range += ((merged_df[phase] < 49.5) | (merged_df[phase] > 50.5)).sum()
    indicators['ac_frequency_stability'] = freq_outside_range

    # 5. Power Factor
    total_reactive_power = merged_df['L1_reactivePower(W)'] + merged_df['L2_reactivePower(W)'] + merged_df['L3_reactivePower(W)']
    apparent_power = np.sqrt(merged_df['totalActivePower(W)']**2 + total_reactive_power**2)
    power_factor = merged_df['totalActivePower(W)'] / apparent_power
    indicators['power_factor'] = power_factor.mean()

    # 6. Generation/Irradiance Ratio (Normalized)
    merged_df['gen_irradiance_ratio'] = merged_df['totalActivePower(W)'] / merged_df['Irradiance (W/m2)']
    # Benchmark: median ratio during top 10% irradiance
    high_irradiance_threshold = merged_df['Irradiance (W/m2)'].quantile(0.9)
    benchmark_ratio = merged_df[merged_df['Irradiance (W/m2)'] > high_irradiance_threshold]['gen_irradiance_ratio'].median()
    # Normalize: ratio of actual average to benchmark average
    indicators['generation_irradiance_ratio'] = merged_df['gen_irradiance_ratio'].mean() / benchmark_ratio if benchmark_ratio else 0

    # 7. Temporal Variability (Adjusted Threshold)
    power_diff = merged_df['totalActivePower(W)'].diff()
    threshold = merged_df['totalActivePower(W)'].max() * 0.3  # Adjusted to 30%
    indicators['temporal_variability'] = (power_diff.abs() > threshold).sum()

    # --- Save Indicators ---
    indicators_df = pd.DataFrame([indicators])
    output_csv_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    indicators_df.to_csv(output_csv_path, index=False)

    return indicators

if __name__ == "__main__":
    # This part will not be executed when imported as a module
    pass
