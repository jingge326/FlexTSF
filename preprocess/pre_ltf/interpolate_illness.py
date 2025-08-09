import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from pathlib import Path


def process_spline_data(input_file, output_file,
                        noise_factor=0.01,
                        random_seed=None):
    """
    Process weekly data into daily data using cubic spline interpolation with noise
    """
    # Configuration
    COUNT_COLS = ['AGE 0-4', 'AGE 5-24', 'ILITOTAL', 'NUM. OF PROVIDERS', 'OT']
    PCT_COLS = ['% WEIGHTED ILI', '%UNWEIGHTED ILI']

    if random_seed:
        np.random.seed(random_seed)

    # Read and prepare data
    df = pd.read_csv(input_file, parse_dates=['date'])
    orig_dates = df['date'].copy()
    start_date = df['date'].min()

    # Create complete daily timeline
    full_dates = pd.date_range(
        start=start_date, end=orig_dates.max(), freq='D')
    df_full = pd.DataFrame({'date': full_dates})

    # Merge with original data
    df_merged = pd.merge(df_full, df, on='date', how='left')

    # Convert dates to numeric values (days since start)
    x_orig = (df['date'] - start_date).dt.days.values
    x_new = (df_full['date'] - start_date).dt.days.values

    # Perform cubic spline interpolation for each column
    numeric_cols = PCT_COLS + COUNT_COLS
    for col in numeric_cols:
        # Existing values
        y_orig = df[col].values

        # Create cubic spline interpolator
        cs = CubicSpline(x_orig, y_orig)

        # Interpolate values
        df_full[col] = cs(x_new)

        # Preserve original values
        df_full.loc[df_merged[col].notna(), col] = df_merged[col].dropna()

    # Add noise to interpolated points
    for col in numeric_cols:
        orig_std = df[col].std()
        mask = df_full[col].isna() | df_full[col].notna()  # All points
        noise = np.random.normal(
            scale=orig_std*noise_factor, size=len(df_full))
        df_full[col] += noise

    # Post-processing
    df_full[COUNT_COLS] = df_full[COUNT_COLS].round().clip(lower=0)
    df_full[PCT_COLS] = df_full[PCT_COLS].clip(lower=0)

    # Final output
    df_full.to_csv(output_file, index=False)


# Usage
folder_data = Path("/home/xiao/project/UpFlexTSF/data/ltf/illness")
process_spline_data(
    input_file=folder_data/'national_illness.csv',
    output_file=folder_data/'national_illness_ext.csv',
    noise_factor=0.01,
    random_seed=42
)
