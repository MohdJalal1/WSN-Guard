import pandas as pd
import joblib

# Load the saved model and preprocessing objects
data = joblib.load("best_ids_model.pkl")
model = data['model']
scaler = data['scaler']
selector = data['selector']
pca = data['pca']

# Example: new network traffic sample as a dictionary
# Make sure the keys match your dataset columns (except 'label' and 'difficulty_level')
new_sample = {
    'duration': 0,
    'protocol_type': 'tcp',
    'service': 'http',
    'flag': 'SF',
    'src_bytes': 181,
    'dst_bytes': 5450,
    'land': 0,
    'wrong_fragment': 0,
    'urgent': 0,
    'hot': 0,
    'num_failed_logins': 0,
    'logged_in': 1,
    'num_compromised': 0,
    'root_shell': 0,
    'su_attempted': 0,
    'num_root': 0,
    'num_file_creations': 0,
    'num_shells': 0,
    'num_access_files': 0,
    'num_outbound_cmds': 0,
    'is_host_login': 0,
    'is_guest_login': 0,
    'count': 2,
    'srv_count': 2,
    'serror_rate': 0.0,
    'srv_serror_rate': 0.0,
    'rerror_rate': 0.0,
    'srv_rerror_rate': 0.0,
    'same_srv_rate': 1.0,
    'diff_srv_rate': 0.0,
    'srv_diff_host_rate': 0.0,
    'dst_host_count': 150,
    'dst_host_srv_count': 100,
    'dst_host_same_srv_rate': 0.67,
    'dst_host_diff_srv_rate': 0.0,
    'dst_host_same_src_port_rate': 0.0,
    'dst_host_srv_diff_host_rate': 0.0,
    'dst_host_serror_rate': 0.0,
    'dst_host_srv_serror_rate': 0.0,
    'dst_host_rerror_rate': 0.0,
    'dst_host_srv_rerror_rate': 0.0
}

# Convert to DataFrame
df = pd.DataFrame([new_sample])

# Encode categorical columns same as training
for col in ['protocol_type', 'service', 'flag']:
    df[col] = df[col].astype('category').cat.codes  # simple encoding

# Scale features
X_scaled = scaler.transform(df)

# Select features (SelectKBest)
X_selected = selector.transform(X_scaled)

# Apply PCA
X_pca = pca.transform(X_selected)

# Predict
prediction = model.predict(X_pca)
prediction_prob = model.predict_proba(X_pca)

# Show results
print("Prediction:", "Attack" if prediction[0]==1 else "Normal")
print("Probability (Normal, Attack):", prediction_prob[0])
