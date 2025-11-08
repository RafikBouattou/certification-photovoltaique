import pandas as pd

def process_and_merge_data():
    # Load generation data
    generation_df = pd.read_csv(r"C:\Users\Flipper\data-cleaner-frontend\temp_dataset\Dataset\Time series dataset\PV generation dataset\PV stations with panel level optimizer\Site level dataset\Indoor Sports Centre.csv")

    # Load irradiance data for all years
    irradiance_2021_df = pd.read_csv(r"C:\Users\Flipper\data-cleaner-frontend\temp_dataset\Dataset\Time series dataset\Meteorological dataset\Irradiance\Irradiance_2021.csv")
    irradiance_2022_df = pd.read_csv(r"C:\Users\Flipper\data-cleaner-frontend\temp_dataset\Dataset\Time series dataset\Meteorological dataset\Irradiance\Irradiance_2022.csv")
    irradiance_2023_df = pd.read_csv(r"C:\Users\Flipper\data-cleaner-frontend\temp_dataset\Dataset\Time series dataset\Meteorological dataset\Irradiance\Irradiance_2023.csv")
    irradiance_df = pd.concat([irradiance_2021_df, irradiance_2022_df, irradiance_2023_df])

    # Convert 'Time' columns to datetime objects
    generation_df['Time'] = pd.to_datetime(generation_df['Time'])
    irradiance_df['Time'] = pd.to_datetime(irradiance_df['Time'])

    # Set 'Time' as index
    generation_df.set_index('Time', inplace=True)
    irradiance_df.set_index('Time', inplace=True)

    # Resample irradiance data to 15-minute intervals and calculate the mean
    irradiance_resampled_df = irradiance_df.resample('15T').mean()

    # Merge the two dataframes
    merged_df = pd.merge(generation_df, irradiance_resampled_df, left_index=True, right_index=True, how='inner')

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(r"C:\Users\Flipper\Indoor_Sports_Centre_merged.csv")

    print("Data processing and merging complete. Merged file saved as Indoor_Sports_Centre_merged.csv")
    print(merged_df.head())

if __name__ == "__main__":
    process_and_merge_data()