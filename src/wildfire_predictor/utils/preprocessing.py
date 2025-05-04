import pandas as pd

def preprocess_wildfire_data(input_csv, output_csv):
    """Preprocess the wildfire data by filtering, encoding categorical variables,"""

    df = pd.read_csv(input_csv)

    df = df[df['SRC_AGENCY'] == 'BC'].copy()
    df.drop(columns=['SRC_AGENCY'], inplace=True)
    df['REP_DATE'] = pd.to_datetime(df['REP_DATE'], errors='coerce')
    df = df.dropna(subset=['REP_DATE'])

    df['YEAR'] = df['REP_DATE'].dt.year
    df['MONTH'] = df['REP_DATE'].dt.month
    df['DAY'] = df['REP_DATE'].dt.day
    df['DAYOFYEAR'] = df['REP_DATE'].dt.dayofyear

    df = df[df['MONTH'].between(5, 9)]
    df = df[df['SIZE_HA'] > 0] 

    df['SIZE_HA'] = df['SIZE_HA'].clip(upper=5000) 
    df['SIZE_HA'] = df['SIZE_HA'].round(3)

    important_cols = ['LATITUDE', 'LONGITUDE', 'CAUSE', 'ECOZ_NAME', 'PROTZONE']
    df.dropna(subset=important_cols, inplace=True)
    cause_dummies = pd.get_dummies(df['CAUSE'], prefix='CAUSE')
    ecoz_dummies = pd.get_dummies(df['ECOZ_NAME'], prefix='ECOZ')
    protzone_dummies = pd.get_dummies(df['PROTZONE'], prefix='PROTZONE')

    df_encoded = pd.concat([df, cause_dummies, ecoz_dummies, protzone_dummies], axis=1)
    df_encoded.drop(columns=['CAUSE', 'ECOZ_NAME', 'PROTZONE', 'REP_DATE'], inplace=True)
    df_encoded.to_csv(output_csv, index=False)

    print(f"Preprocessed data saved to {output_csv}")
