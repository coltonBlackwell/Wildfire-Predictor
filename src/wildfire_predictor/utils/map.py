import numpy as np
from folium import Marker, Circle, FeatureGroup, LayerControl, Map, Popup, GeoJson
import joblib
import random
import json


def create_html_map():
    """Create an HTML map with predicted and actual fire sizes using single-regressor outputs."""

    # Matches what your new training code saves:
    # (X_test, y_pred_log, y_test_log, regressor, scaler)
    X_test, y_pred_log, y_test_log, regressor, scaler = joblib.load('model_outputs.pkl')

    # For a quick sanity check on one row
    sample = X_test.iloc[0:1]
    sample_pred_log = regressor.predict(sample)
    sample_pred = np.expm1(sample_pred_log)[0]
    sample_actual = np.expm1(y_test_log.iloc[0])

    print("\n--- Single Prediction Example ---")
    print(f"Predicted log(SIZE_HA): {sample_pred_log[0]:.4f}")
    print(f"Predicted SIZE_HA: {sample_pred:.2f}")
    print(f"Actual log(SIZE_HA): {y_test_log.iloc[0]:.4f}")
    print(f"Actual SIZE_HA: {sample_actual:.2f}")

    # Sample up to 1000 points safely
    sample_count = min(1000, len(X_test))
    sample_idxs = random.sample(range(len(X_test)), sample_count)

    m = Map(location=[53.5, -125], zoom_start=5.8, tiles='Esri.WorldImagery')

    predicted_layer = FeatureGroup(name="Predicted Radius")
    actual_layer = FeatureGroup(name="Actual Radius")

    for idx in sample_idxs:
        row = X_test.iloc[idx]   # Series
        pred_log = y_pred_log[idx]
        pred_ha = float(np.expm1(pred_log))
        actual_ha = float(np.expm1(y_test_log.iloc[idx]))

        # Invert StandardScaler for lat/lon (x_original = x_scaled * scale + mean)
        lat = row['LATITUDE'] * scaler.scale_[0] + scaler.mean_[0]
        lon = row['LONGITUDE'] * scaler.scale_[1] + scaler.mean_[1]

        # Correct circle radius from hectares:
        # 1 ha = 10,000 m^2; radius = sqrt(area / pi) = sqrt(ha * 10000 / pi)
        def ha_to_radius_m(ha: float) -> float:
            ha = max(ha, 0.0)
            return float(np.sqrt(ha * 10000.0 / np.pi))

        pred_radius_m = ha_to_radius_m(pred_ha)
        actual_radius_m = ha_to_radius_m(actual_ha)

        # Safely fetch one-hot/binary features (default to 0 if missing)
        def safe(row_, key):
            try:
                return int(row_.get(key, 0))
            except Exception:
                return 0

        data_info = f"""
        <b>Cause (Human):</b> {safe(row, 'CAUSE_H')}<br>
        <b>Cause (Lightning):</b> {safe(row, 'CAUSE_L')}<br>
        <b>Cause (Unknown):</b> {safe(row, 'CAUSE_U')}<br>
        <b>Boreal Cordillera:</b> {safe(row, 'ECOZ_Boreal Cordillera')}<br>
        <b>Boreal Plain:</b> {safe(row, 'ECOZ_Boreal PLain')}<br>
        <b>Montane Cordillera:</b> {safe(row, 'ECOZ_Montane Cordillera')}<br>
        <b>Pacific Maritime:</b> {safe(row, 'ECOZ_Pacific Maritime')}<br>
        <b>Prairie:</b> {safe(row, 'ECOZ_Prairie')}<br>
        <b>Taiga Plain:</b> {safe(row, 'ECOZ_Taiga Plain')}<br>
        """

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
