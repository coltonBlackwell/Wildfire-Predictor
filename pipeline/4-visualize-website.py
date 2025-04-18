import numpy as np
import folium
from folium import Marker, Circle, FeatureGroup, LayerControl, Element, Html
import joblib
import random
import json

# Load saved models and data
(X_test, y_pred_log, small_idx_test, y_reg_test,
 y_class_pred, small_regressor, large_regressor, scaler) = joblib.load('model_outputs.pkl')

# === Single Prediction Example ===
sample = X_test.iloc[0:1]
sample_pred_log = small_regressor.predict(sample)
sample_pred = np.expm1(sample_pred_log)

actual_log = y_pred_log[small_idx_test][0]
actual = np.expm1(actual_log)

print("\n--- Single Prediction Example ---")
print(f"Predicted log(SIZE_HA): {sample_pred_log[0]:.4f}")
print(f"Predicted SIZE_HA: {sample_pred[0]:.2f}")
print(f"Actual log(SIZE_HA): {actual_log:.4f}")
print(f"Actual SIZE_HA: {actual:.2f}")

# === Generate Folium Map ===
sample_count = 1000
sample_idxs = random.sample(range(len(X_test)), sample_count)

m = folium.Map(location=[53.5, -125], zoom_start=5.8, tiles='Esri.WorldImagery')

# Feature groups
predicted_layer = FeatureGroup(name="Predicted Radius")
actual_layer = FeatureGroup(name="Actual Radius")

for sample_idx in sample_idxs:
    sample = X_test.iloc[sample_idx:sample_idx+1]
    pred_log = small_regressor.predict(sample) if y_class_pred[sample_idx] == 0 else large_regressor.predict(sample)
    pred_ha = np.expm1(pred_log)[0]
    actual_ha = np.expm1(y_reg_test.iloc[sample_idx])

    lat = sample['LATITUDE'].values[0] * scaler.scale_[0] + scaler.mean_[0]
    lon = sample['LONGITUDE'].values[0] * scaler.scale_[1] + scaler.mean_[1]

    pred_radius_m = np.sqrt(pred_ha * 0.01 / np.pi) * 1000
    actual_radius_m = np.sqrt(actual_ha * 0.01 / np.pi) * 1000

    data_info = f"""
    <b>Cause (Human):</b> {sample['CAUSE_H'].values[0]}<br>
    <b>Cause (Lightning):</b> {sample['CAUSE_L'].values[0]}<br>
    <b>Cause (Unknown):</b> {sample['CAUSE_U'].values[0]}<br>
    <b>Boreal Cordillera:</b> {sample['ECOZ_Boreal Cordillera'].values[0]}<br>
    <b>Boreal Plain:</b> {sample['ECOZ_Boreal PLain'].values[0]}<br>
    <b>Montane Cordillera:</b> {sample['ECOZ_Montane Cordillera'].values[0]}<br>
    <b>Pacific Maritime:</b> {sample['ECOZ_Pacific Maritime'].values[0]}<br>
    <b>Prairie:</b> {sample['ECOZ_Prairie'].values[0]}<br>
    <b>Taiga Plain:</b> {sample['ECOZ_Taiga Plain'].values[0]}<br>
    """

    Marker([lat, lon], popup=folium.Popup(data_info, max_width=400)).add_to(m)

    # Add predicted radius (always visible)
    Circle([lat, lon], radius=pred_radius_m, color='red', fill=True,
           fill_opacity=0.3, popup=f'Predicted: {pred_ha:.1f} ha').add_to(predicted_layer)

    # Add actual radius to toggle group
    Circle([lat, lon], radius=actual_radius_m, color='blue', fill=True,
           fill_opacity=0.3, popup=f'Actual: {actual_ha:.1f} ha').add_to(actual_layer)

# Load GeoJSON data
with open("georef-canada-province@public.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# Get all property fields dynamically
fields = list(geojson_data['features'][0]['properties'].keys())



# Add GeoJson layer with all fields
folium.GeoJson(
    geojson_data,
    name="Provinces",
    style_function=lambda feature: {
        'fillColor': '#00000000',  # transparent
        'color': 'yellow',         # change for visibility
        'weight': 2,
    }
).add_to(m)

# Add feature groups to map
predicted_layer.add_to(m)
actual_layer.add_to(m)

# Add LayerControl (to toggle layers)
LayerControl().add_to(m)

# Save map
m.save('wildfire_prediction_map.html')
