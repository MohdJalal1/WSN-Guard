from sklearn.preprocessing import LabelEncoder

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le
from wsn_ids_data_explore import train_data
from wsn_ids_data_explore import test_data