import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely import wkt
import numpy as np

# Read and prepare data
fire_grid = pd.read_csv('../data/fire_grid_counts.csv')
fire_grid['geometry'] = fire_grid['geometry'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(fire_grid, geometry='geometry')
gdf.crs = "EPSG:3005"

# Aggregate fire counts
fire_counts = gdf.groupby('cell_id').agg({'FIRE_COUNT': 'sum', 'geometry': 'first'})
fire_counts = gpd.GeoDataFrame(fire_counts, geometry='geometry')
fire_counts.crs = gdf.crs

# Create plot
fig, ax = plt.subplots(figsize=(15, 15))

# Plot with adjusted threshold (vmax=10 makes cells with 10+ fires appear red)
fire_counts.plot(column='FIRE_COUNT', 
                cmap='YlOrRd', 
                legend=True,
                ax=ax,
                vmin=0,
                vmax=100,  # Adjust this number to control sensitivity
                legend_kwds={'label': "Fire Count", 'shrink': 0.5},
                edgecolor='none',
                alpha=0.7)

# Add basemap
ctx.add_basemap(ax, crs=fire_counts.crs.to_string(), 
               source=ctx.providers.OpenStreetMap.Mapnik)

ax.set_title('Fire Count Heatmap in British Columbia (Adjusted Threshold)', fontsize=16)
ax.set_axis_off()
plt.tight_layout()
plt.show()