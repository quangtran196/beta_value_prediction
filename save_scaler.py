# save_scaler_properly.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load your CSV
df = pd.read_csv('merged_data_with_error.csv')

# Create scaler
scaler = MinMaxScaler()

# Fit only on the relevant columns
data_to_scale = df[['amplitude', 'frequency', 'beta_value']].values
scaler.fit(data_to_scale)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

print("Scaler saved with proper ranges:")
print(f"Min values: {scaler.data_min_}")
print(f"Max values: {scaler.data_max_}")

# Test the scaler
test_beta = 0.2
test_array = np.array([[0, 0, test_beta]])
normalized = scaler.transform(test_array)[0, 2]
print(f"\nTest: Beta {test_beta} → Normalized {normalized:.3f}")