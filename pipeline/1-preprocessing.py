import pandas as pd

def preprocess_wildfire_data(input_csv='../data/CANADA_WILDFIRES.csv', output_csv='../data/wildfires_preprocessed.csv'):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Filter for fires only in BC
    df = df[df['SRC_AGENCY'] == 'BC'].copy()

    # Drop SRC_AGENCY since it's now redundant
    df.drop(columns=['SRC_AGENCY'], inplace=True)

    # Convert REP_DATE to datetime
    df['REP_DATE'] = pd.to_datetime(df['REP_DATE'], errors='coerce')

    # Drop rows with invalid or missing REP_DATE
    df = df.dropna(subset=['REP_DATE'])

    # Extract date features
    df['YEAR'] = df['REP_DATE'].dt.year
    df['MONTH'] = df['REP_DATE'].dt.month
    df['DAY'] = df['REP_DATE'].dt.day
    df['DAYOFYEAR'] = df['REP_DATE'].dt.dayofyear

    # Filter to summer fires only (May to September)
    df = df[df['MONTH'].between(5, 9)]

    # Remove extreme or invalid fire sizes
    df = df[df['SIZE_HA'] > 0]              # Remove 0 or negative sizes
    df['SIZE_HA'] = df['SIZE_HA'].clip(upper=10000)  # Cap extremely large values

    # Round SIZE_HA to the nearest 0.001
    df['SIZE_HA'] = df['SIZE_HA'].round(3)

    # Drop rows with nulls in important features
    important_cols = ['LATITUDE', 'LONGITUDE', 'CAUSE', 'ECOZ_NAME', 'PROTZONE']
    df.dropna(subset=important_cols, inplace=True)

    # One-hot encode categorical columns
    cause_dummies = pd.get_dummies(df['CAUSE'], prefix='CAUSE')
    ecoz_dummies = pd.get_dummies(df['ECOZ_NAME'], prefix='ECOZ')
    protzone_dummies = pd.get_dummies(df['PROTZONE'], prefix='PROTZONE')

    # Concatenate all encoded features
    df_encoded = pd.concat([df, cause_dummies, ecoz_dummies, protzone_dummies], axis=1)

    # Drop unused categorical/original columns
    df_encoded.drop(columns=['CAUSE', 'ECOZ_NAME', 'PROTZONE', 'REP_DATE'], inplace=True)

    # Save the cleaned and feature-enriched data
    df_encoded.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

if __name__ == '__main__':
    preprocess_wildfire_data()
