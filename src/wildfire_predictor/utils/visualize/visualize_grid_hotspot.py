import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkt

def visualize_fire_grid(input_csv):
    """Visualize the fire grid with 3D bar plots representing fire counts."""

    fire_grid = pd.read_csv(input_csv)
    fire_grid['geometry'] = fire_grid['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(fire_grid, geometry='geometry')
    gdf.crs = "EPSG:3005"

    fire_counts = gdf.groupby('cell_id').agg({'FIRE_COUNT': 'sum', 'geometry': 'first'})
    fire_counts = gpd.GeoDataFrame(fire_counts, geometry='geometry')
    fire_counts.crs = gdf.crs

    fire_counts['centroid'] = fire_counts.geometry.centroid
    fire_counts['x'] = fire_counts.centroid.x
    fire_counts['y'] = fire_counts.centroid.y

    def get_bounds(poly):
        x, y = poly.exterior.xy
        return min(x), max(x), min(y), max(y)

    fire_counts['xmin'], fire_counts['xmax'], fire_counts['ymin'], fire_counts['ymax'] = \
        zip(*fire_counts.geometry.apply(get_bounds))

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')

    max_fires = fire_counts['FIRE_COUNT'].max()
    height_scale = 1000

    for idx, row in fire_counts.iterrows():

        dx = row.xmax - row.xmin
        dy = row.ymax - row.ymin
        dz = (row.FIRE_COUNT / max_fires) * height_scale
        
        color = plt.cm.YlOrRd(row.FIRE_COUNT / max_fires)
        
        ax.bar3d(row.xmin,
                row.ymin,     
                0,           
                dx,             
                dy,   
                dz,   
                color=color,
                alpha=0.8,
                edgecolor='k',
                linewidth=0.3)

    ax.set_title('3D Fire Count Visualization in BC', fontsize=16, pad=20)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Fire Count (scaled)')

    ax.view_init(elev=40, azim=-60)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                            norm=plt.Normalize(vmin=0, vmax=max_fires))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Fire Count')

    plt.tight_layout()
    plt.show()