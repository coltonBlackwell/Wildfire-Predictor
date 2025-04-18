import pandas as pd

def preprocess_wildfire_data(input_csv='../data/CANADA_WILDFIRES.csv', output_csv='../data/wildfires_preprocessed.csv'):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Filter for fires only in BC
    df = df[df['SRC_AGENCY'] == 'BC']

    # Convert REP_DATE to datetime
    df['REP_DATE'] = pd.to_datetime(df['REP_DATE'], errors='coerce')

    # Drop rows with invalid dates
    df = df.dropna(subset=['REP_DATE'])

    # Extract date features
    df['YEAR'] = df['REP_DATE'].dt.year
    df['MONTH'] = df['REP_DATE'].dt.month
    df['DAY'] = df['REP_DATE'].dt.day
    df['DAYOFYEAR'] = df['REP_DATE'].dt.dayofyear

    # Create a "season" column (summer: May–September)
    df['SEASON'] = df['MONTH'].apply(lambda x: 'summer' if 5 <= x <= 9 else 'other')

    # Filter out non-summer fires
    df = df[df['SEASON'] == 'summer']

    # Normalize coordinates (if needed, scaling can be applied here)
    df['LATITUDE'] = df['LATITUDE']
    df['LONGITUDE'] = df['LONGITUDE']

    # One-hot encode CAUSE
    cause_dummies = pd.get_dummies(df['CAUSE'], prefix='CAUSE')

    # One-hot encode ECOZ_NAME and PROTZONE
    ecoz_dummies = pd.get_dummies(df['ECOZ_NAME'], prefix='ECOZ')
    protzone_dummies = pd.get_dummies(df['PROTZONE'], prefix='PROTZONE')

    # Concatenate encoded columns
    df_encoded = pd.concat([df, cause_dummies, ecoz_dummies, protzone_dummies], axis=1)

    # Drop original categorical columns
    df_encoded = df_encoded.drop(['CAUSE', 'ECOZ_NAME', 'PROTZONE', 'SEASON'], axis=1)

    # Save preprocessed data
    df_encoded.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

if __name__ == '__main__':
    preprocess_wildfire_data()
