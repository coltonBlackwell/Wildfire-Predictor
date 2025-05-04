import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import numpy as np

def create_grid(input_csv, output_csv, grid_size_km):
    # Load preprocessed data
    df = pd.read_csv(input_csv)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']))
    gdf.set_crs(epsg=4326, inplace=True)

    # Project to a metric CRS (BC Albers - good for distance in meters)
    gdf = gdf.to_crs(epsg=3005)

    # Get bounds for BC region
    minx, miny, maxx, maxy = gdf.total_bounds

    # Create grid
    grid_size_m = grid_size_km * 1000
    cols = list(np.arange(minx, maxx + grid_size_m, grid_size_m))
    rows = list(np.arange(miny, maxy + grid_size_m, grid_size_m))
    grid_cells = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            cell = box(x, y, x + grid_size_m, y + grid_size_m)
            grid_cells.append(cell)
    grid = gpd.GeoDataFrame({'geometry': grid_cells})
    grid.set_crs(epsg=3005, inplace=True)

    # Spatial join: assign fires to grid cells
    joined = gpd.sjoin(gdf, grid.reset_index().rename(columns={'index': 'cell_id'}), how='inner', predicate='within')

    # Count number of fires per cell per year
    grouped = joined.groupby(['cell_id', 'YEAR']).size().reset_index(name='FIRE_COUNT')

    # Optional: join back with grid to get geometry for each cell_id
    grouped = pd.merge(grouped, grid.reset_index().rename(columns={'index': 'cell_id'}), on='cell_id')

    # Save result
    grouped_gdf = gpd.GeoDataFrame(grouped, geometry='geometry', crs='EPSG:3005')
    grouped_gdf.to_csv(output_csv, index=False)
    print(f"Fire counts per grid cell saved to {output_csv}")