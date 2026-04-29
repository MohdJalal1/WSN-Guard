import pandas as pd
import numpy as np

# Load NSL-KDD dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty_level'
]

# Read data
train_data = pd.read_csv('KDDTrain+.txt', names=column_names)
test_data = pd.read_csv('KDDTest+.txt', names=column_names)

# Explore data
print(train_data.head())
print(train_data['label'].value_counts())
print(train_data.info())


from sklearn.preprocessing import LabelEncoder

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le


# ✔ Check encoded values
print("Encoded training data sample:")
print(train_data[categorical_cols].head())

print("\nEncoded test data sample:")
print(test_data[categorical_cols].head())

# ✔ Check category → number mapping
for col, le in label_encoders.items():
    print(f"\nMapping for {col}:")
    print(list(le.classes_))


# Convert multi-class labels to binary (normal vs attack)
train_data['label'] = train_data['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_data['label'] = test_data['label'].apply(lambda x: 0 if x == 'normal' else 1)


from sklearn.preprocessing import StandardScaler

# Separate features and labels
X_train = train_data.drop(['label', 'difficulty_level'], axis=1)
y_train = train_data['label']
X_test = test_data.drop(['label', 'difficulty_level'], axis=1)
y_test = test_data['label']

# Normalize features (important for SGD)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"Original dataset shape: {X_train_scaled.shape}")
print(f"Balanced dataset shape: {X_train_balanced.shape}")
