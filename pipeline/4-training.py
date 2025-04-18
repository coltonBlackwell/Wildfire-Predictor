import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def train_test_split_and_predict(input_csv='../data/grid_cell_features.csv'):
    # Load the grid cell features
    df = pd.read_csv(input_csv)

    # Filter out the columns of interest (excluding the categorical one-hot encoded ones)
    feature_columns = ['num_fires', 'avg_size', 'dist_to_last_year']
    target_column = 'num_fires'  # Assuming we're predicting number of fires (you can change this)
    
    # Prepare the data
    X = df[feature_columns]
    y = df[target_column]

    # Split into train (1950-2019) and test (2020-2023)
    train_data = df[df['year'] <= 2019]
    test_data = df[(df['year'] >= 2020) & (df['year'] <= 2023)]
    future_data = df[df['year'] == 2025]  # Data for 2025 prediction (if available)

    # Features and target for training and validation sets
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    
    # Train a model (Decision Tree Regressor in this case)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Validate the model (2020-2023)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error for 2020-2023 validation: {mse}')

    # Predict for 2025 (based on 2023/2024 trends)
    if not future_data.empty:
        X_future = future_data[feature_columns]
        y_future_pred = model.predict(X_future)
        future_data['predicted_num_fires'] = y_future_pred
        print(f'Predictions for 2025: \n{future_data[["cell_id", "year", "predicted_num_fires"]]}')

    # Return the trained model (optional, in case further predictions are needed)
    return model

if __name__ == '__main__':
    train_test_split_and_predict()
