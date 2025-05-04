from utils.preprocessing import preprocess_wildfire_data
from utils.visualize.create_fire_grid import create_grid
from utils.visualize.visualize_grid_hotspot import visualize_fire_grid
from utils.train_model import train
from utils.test_model import test
from utils.map import create_html_map

def main():

    #Preprocessing
    preprocess_wildfire_data(input_csv='../../data/CANADA_WILDFIRES.csv', output_csv='../../data/processed/wildfires_preprocessed.csv')

    #Visualizing
    create_grid(input_csv='../../data/processed/wildfires_preprocessed.csv', output_csv='../../data/processed/fire_grid_counts.csv', grid_size_km=10)
    visualize_fire_grid(input_csv='../../data/processed/fire_grid_counts.csv')

    #train model
    X, large_regressor, y_pred, y_test_actual = train(input_csv='../../data/processed/wildfires_preprocessed.csv')

    #test model
    test(X, large_regressor, y_pred, y_test_actual)

    #create html map (index.html)
    create_html_map()


if __name__ == '__main__':
    main()
