import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkt

def visualize_fire_grid(input_csv):

    # Load and prepare data
    fire_grid = pd.read_csv(input_csv)
    fire_grid['geometry'] = fire_grid['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(fire_grid, geometry='geometry')
    gdf.crs = "EPSG:3005"

    # Aggregate fire counts
    fire_counts = gdf.groupby('cell_id').agg({'FIRE_COUNT': 'sum', 'geometry': 'first'})
    fire_counts = gpd.GeoDataFrame(fire_counts, geometry='geometry')
    fire_counts.crs = gdf.crs

    # Extract centroids and bounds
    fire_counts['centroid'] = fire_counts.geometry.centroid
    fire_counts['x'] = fire_counts.centroid.x
    fire_counts['y'] = fire_counts.centroid.y

    # Get bounds for sizing
    def get_bounds(poly):
        x, y = poly.exterior.xy
        return min(x), max(x), min(y), max(y)

    fire_counts['xmin'], fire_counts['xmax'], fire_counts['ymin'], fire_counts['ymax'] = \
        zip(*fire_counts.geometry.apply(get_bounds))

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate dimensions and scaling
    max_fires = fire_counts['FIRE_COUNT'].max()
    height_scale = 1000  # Adjust for visual prominence

    # Plot each cell
    for idx, row in fire_counts.iterrows():
        # Calculate bar dimensions
        dx = row.xmax - row.xmin
        dy = row.ymax - row.ymin
        dz = (row.FIRE_COUNT / max_fires) * height_scale
        
        # Create color based on fire count
        color = plt.cm.YlOrRd(row.FIRE_COUNT / max_fires)
        
        # Plot the 3D bar
        ax.bar3d(row.xmin,         # x coordinate of bottom corner
                row.ymin,         # y coordinate of bottom corner
                0,                # z coordinate of bottom corner (base)
                dx,               # width (x-direction)
                dy,              # depth (y-direction)
                dz,              # height (z-direction)
                color=color,
                alpha=0.8,
                edgecolor='k',
                linewidth=0.3)

    # Formatting
    ax.set_title('3D Fire Count Visualization in BC', fontsize=16, pad=20)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Fire Count (scaled)')

    # Adjust view
    ax.view_init(elev=40, azim=-60)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                            norm=plt.Normalize(vmin=0, vmax=max_fires))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Fire Count')

    plt.tight_layout()
    plt.show()