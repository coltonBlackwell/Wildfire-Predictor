import numpy as np
from folium import Marker, Circle, FeatureGroup, LayerControl, Map, Popup, GeoJson
import joblib
import random
import json

def create_html_map():
    """Create an HTML map with predicted and actual wildfire sizes."""

    # Load model outputs (updated to include LAT_ORIG / LON_ORIG)
    # (X_test, y_pred_log, y_test_log, regressor, scaler)
    X_test, y_pred_log, y_test_log, regressor, scaler = joblib.load('model_outputs.pkl')

    # Quick sanity check
    sample = X_test.iloc[0:1]
    sample_pred_log = regressor.predict(sample)
    sample_pred = np.expm1(sample_pred_log)[0]
    sample_actual = np.expm1(y_test_log.iloc[0])
    print(f"Sample Prediction: {sample_pred:.2f} ha, Actual: {sample_actual:.2f} ha")

    # Limit points for plotting
    sample_count = min(1000, len(X_test))
    sample_idxs = random.sample(range(len(X_test)), sample_count)

    # Base map
    m = Map(location=[53.5, -125], zoom_start=5.8, tiles='Esri.WorldImagery')
    predicted_layer = FeatureGroup(name="Predicted Radius")
    actual_layer = FeatureGroup(name="Actual Radius")

    # Conversion from hectares to radius in meters
    def ha_to_radius_m(ha: float) -> float:
        ha = max(ha, 0.0)
        return float(np.sqrt(ha * 10000.0 / np.pi))

    for idx in sample_idxs:
        row = X_test.iloc[idx]

        # Use original LAT/LON for plotting
        lat = row.get('LAT_ORIG', row.get('LATITUDE'))
        lon = row.get('LON_ORIG', row.get('LONGITUDE'))

        pred_ha = float(np.expm1(y_pred_log[idx]))
        actual_ha = float(np.expm1(y_test_log.iloc[idx]))

        pred_radius_m = ha_to_radius_m(pred_ha)
        actual_radius_m = ha_to_radius_m(actual_ha)

        # Popup info
        data_info = f"""
        <b>Predicted:</b> {pred_ha:.1f} ha<br>
        <b>Actual:</b> {actual_ha:.1f} ha<br>
        <b>Latitude:</b> {lat:.4f}<br>
        <b>Longitude:</b> {lon:.4f}<br>
        """

        # Add marker and circles
        Marker([lat, lon], popup=Popup(data_info, max_width=400)).add_to(m)
        Circle([lat, lon], radius=pred_radius_m, color='red', fill=True,
               fill_opacity=0.3, popup=f'Predicted: {pred_ha:.1f} ha').add_to(predicted_layer)
        Circle([lat, lon], radius=actual_radius_m, color='blue', fill=True,
               fill_opacity=0.3, popup=f'Actual: {actual_ha:.1f} ha').add_to(actual_layer)

    # Provinces overlay
    with open("../../json/georef-canada-province@public.geojson", "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    GeoJson(
        geojson_data,
        name="Provinces",
        style_function=lambda feature: {
            'fillColor': '#00000000',
            'color': 'yellow',
            'weight': 2,
        }
    ).add_to(m)

    predicted_layer.add_to(m)
    actual_layer.add_to(m)
    LayerControl().add_to(m)

    m.save('index.html')
    print("Map saved to 'index.html'")
