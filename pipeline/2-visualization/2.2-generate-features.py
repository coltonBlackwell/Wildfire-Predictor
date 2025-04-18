import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def get_most_common(group, prefix):
    cols = [col for col in group.columns if col.startswith(prefix)]
    if not cols:
        return None
    values = group[cols].sum().values
    idx = np.argmax(values)
    return cols[idx].replace(prefix, '').strip('_')

def generate_features(input_csv='../data/wildfires_preprocessed.csv',
                      grid_counts_csv='../data/fire_grid_counts.csv',
                      output_csv='../data/grid_cell_features.csv'):

    # Load and prepare wildfire data
    fire_df = pd.read_csv(input_csv)
    fire_gdf = gpd.GeoDataFrame(
        fire_df,
        geometry=gpd.points_from_xy(fire_df['LONGITUDE'], fire_df['LATITUDE']),
        crs='EPSG:4326'
    ).to_crs(epsg=3005)

    # Load grid data
    grid_df = pd.read_csv(grid_counts_csv)
    grid_gdf = gpd.GeoDataFrame(
        grid_df,
        geometry=gpd.GeoSeries.from_wkt(grid_df['geometry']),
        crs='EPSG:3005'
    )

    # Spatial join
    fire_gdf = gpd.sjoin(
        fire_gdf,
        grid_gdf[['cell_id', 'geometry']],
        how='inner',
        predicate='within'
    )

    print("Available columns:", fire_gdf.columns.tolist())

    # Pre-grouped dictionary for fast lookup of previous year centroids
    grouped = dict(tuple(fire_gdf.groupby(['cell_id', 'YEAR'])))

    # Precompute centroids for each group
    centroids = {
        (cell_id, year): group.geometry.unary_union.centroid
        for (cell_id, year), group in grouped.items()
    }

    features = []

    # Initialize progress tracking
    total_combinations = len(grouped)
    print(f"Processing {total_combinations} (cell_id, year) combinations...")

    for idx, ((cell_id, year), group) in enumerate(grouped.items()):
        # Print progress every 100 iterations
        if idx % 100 == 0:
            print(f"Processing combination {idx + 1}/{total_combinations}...")

        avg_size = group['SIZE_HA'].mean()
        num_fires = len(group)
        common_cause = get_most_common(group, 'CAUSE_')
        common_ecoz = get_most_common(group, 'ECOZ_')
        common_protz = get_most_common(group, 'PROTZONE_')

        # Centroid distance to previous year
        prev_key = (cell_id, year - 1)
        curr_centroid = centroids[(cell_id, year)]
        prev_centroid = centroids.get(prev_key)
        dist_to_prev = curr_centroid.distance(prev_centroid) if prev_centroid else np.nan

        features.append({
            'cell_id': cell_id,
            'year': year,
            'num_fires': num_fires,
            'avg_size': avg_size,
            'common_cause': common_cause,
            'common_ecoz': common_ecoz,
            'common_protz': common_protz,
            'dist_to_last_year': dist_to_prev
        })

    # Final dataframe and one-hot encoding
    df_feat = pd.DataFrame(features)
    df_feat = pd.get_dummies(df_feat, columns=['common_cause', 'common_ecoz', 'common_protz'])

    # Save to CSV
    df_feat.to_csv(output_csv, index=False)
    print(f"Grid cell features saved to {output_csv}")

if __name__ == '__main__':
    generate_features()
