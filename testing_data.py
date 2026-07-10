import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

TEST_DATA_PATH = 'testing.csv'
REG_MODEL_PATH = 'sqa_supplier_score_regressor.joblib'
CLF_MODEL_PATH = 'sqa_risk_classifier.joblib'
OUTPUT_PATH = 'testing_predictions_output.csv'

EXPECTED_FEATURES = [
    'plant_name', 'vehicle_model', 'supplier_id', 'supplier_name', 'supplier_batch_id',
    'part_name', 'part_category', 'market_price_inr', 'open_market_price_range_inr',
    'qty_inspected', 'qty_defective', 'ppm', 'otd_pct', 'audit_score_pct', 'cpk',
    'belt_tension_dev_ratio', 'slip_ratio', 'crack_density_per_100mm',
    'pressure_retention_ratio', 'hose_leak_rate_ml_min', 'engine_temp_dev_ratio',
    'filter_pressure_drop_pa_ratio', 'filtration_efficiency', 'fuel_penalty_ratio',
    'brake_leak_rate_cc_min', 'pushrod_stroke_excess_ratio',
    'brake_response_lag_s_ratio', 'relay_brake_lag_s_ratio',
    'relay_pressure_delivery_ratio', 'relay_leak_rate_cc_min',
    'pulse_integrity_ratio', 'speed_deviation_ratio', 'dropout_count_ratio',
    'bearing_temp_rise_c_ratio', 'bearing_vibration_rms_ratio',
    'bearing_play_mm_ratio', 'ujoint_vibration_rms_ratio', 'ujoint_backlash_mm_ratio',
    'ujoint_temp_rise_c_ratio', 'downstream_water_ppm_ratio', 'separator_dp_kpa_ratio',
    'service_compliance_ratio', 'data_source_type', 'impact_raw',
    'criticality_weight_0_1', 'min_required_sqm_0_1', 'importance_band',
    'test_month', 'test_quarter', 'test_dayofweek'
]

# Load trained models
reg_model = joblib.load(REG_MODEL_PATH)
clf_model = joblib.load(CLF_MODEL_PATH)

# Load test data
raw_df = pd.read_csv(TEST_DATA_PATH)
result_df = raw_df.copy()

# Create date features if test_date is available
if 'test_date' in raw_df.columns:
    raw_df['test_date'] = pd.to_datetime(raw_df['test_date'], errors='coerce')
    raw_df['test_month'] = raw_df['test_date'].dt.month
    raw_df['test_quarter'] = raw_df['test_date'].dt.quarter
    raw_df['test_dayofweek'] = raw_df['test_date'].dt.dayofweek

# Add missing columns as NaN if your larger dataset does not have all training features
missing_features = [c for c in EXPECTED_FEATURES if c not in raw_df.columns]
for c in missing_features:
    raw_df[c] = np.nan

# Keep only the exact training features in the same order
X_test = raw_df[EXPECTED_FEATURES].copy()

# Predict supplier quality score
result_df['predicted_supplier_quality_score_0_1'] = reg_model.predict(X_test).round(4)

# Predict risk label
result_df['predicted_risk_label'] = clf_model.predict(X_test)

# Predict class probabilities
if hasattr(clf_model, 'predict_proba'):
    probas = clf_model.predict_proba(X_test)
    class_names = clf_model.named_steps['model'].classes_
    for idx, class_name in enumerate(class_names):
        safe = str(class_name).lower().replace(' ', '_').replace('/', '_')
        result_df[f'prob_{safe}'] = probas[:, idx].round(4)

# Optional business interpretation
if 'min_required_sqm_0_1' in result_df.columns:
    result_df['predicted_sqm_status'] = np.where(
        result_df['predicted_supplier_quality_score_0_1'] >= result_df['min_required_sqm_0_1'],
        'Target Met',
        'Below/Need Review'
    )

# Save predictions
result_df.to_csv(OUTPUT_PATH, index=False)

print('Testing complete!')
print(f'Input rows: {len(result_df)}')
print(f'Output file saved as: {OUTPUT_PATH}')