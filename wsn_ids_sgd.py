import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

# ===== FILE PATHS =====
TRAIN_PATH = "KDDTrain+.TXT"
TEST_PATH = "KDDTest+.TXT"
MODEL_PATH = "best_ids_model_enhanced.pkl"
# =====================

# NSL-KDD columns
COLUMNS = [
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

# -------- LOAD DATA ----------
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, names=COLUMNS)
    test_df  = pd.read_csv(test_path, names=COLUMNS)
    return train_df, test_df

# -------- PREPROCESS DATA ----------
def preprocess_data(train_df, test_df):
    # Encode categorical columns
    cat_cols = ['protocol_type', 'service', 'flag']
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])

    # Labels: 0=Normal, 1=Attack
    train_df['label'] = train_df['label'].apply(lambda x: 0 if x=='normal' else 1)
    test_df['label']  = test_df['label'].apply(lambda x: 0 if x=='normal' else 1)

    # Split features and labels
    X_train = train_df.drop(['label', 'difficulty_level'], axis=1)
    y_train = train_df['label']
    X_test  = test_df.drop(['label', 'difficulty_level'], axis=1)
    y_test  = test_df['label']

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # SMOTE balancing
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print("Balanced class distribution:\n", pd.Series(y_train_bal).value_counts())

    return X_train_bal, X_test_scaled, y_train_bal, y_test, scaler

# -------- FEATURE SELECTION + PCA ----------
def feature_selection_pca(X_train, X_test, y_train):
    selector = SelectKBest(score_func=mutual_info_classif)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected  = selector.transform(X_test)
    print(f"\nSelected features: {X_train_selected.shape[1]} / {X_train.shape[1]}")

    pca = PCA(n_components=0.90, random_state=42)
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca  = pca.transform(X_test_selected)
    print(f"PCA components: {X_train_pca.shape[1]}, Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    return X_train_pca, X_test_pca, selector, pca

# -------- TRAIN SGD ----------
def train_sgd(X_train, y_train):
    sgd = SGDClassifier(
        loss='log_loss',
        max_iter=5000,
        tol=1e-4,
        class_weight={0:1, 1:3},  # increase attack weight
        random_state=42
    )
    sgd.fit(X_train, y_train)
    return sgd

# -------- TRAIN Logistic Regression ----------
def train_logistic(X_train, y_train):
    clf = LogisticRegression(
        max_iter=5000,
        class_weight={0:1,1:3},
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

# -------- EVALUATION ----------
def evaluate(model, X_test, y_test, threshold=0.5):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
        y_pred = (y_prob >= threshold).astype(int)
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
    else:
        y_pred = model.predict(X_test)
        auc = None
        fpr = tpr = None

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Normal','Attack']))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    if auc is not None:
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (threshold={threshold})")
        plt.legend()
        plt.show()

# -------- ENSEMBLE PREDICTION ----------
def ensemble_predict(models, X_test, threshold=0.5):
    # Average probabilities
    avg_prob = np.mean([m.predict_proba(X_test)[:,1] for m in models], axis=0)
    return (avg_prob >= threshold).astype(int), avg_prob

# -------- MAIN ----------
def main():
    print("Loading data...")
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    print("\nPreprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(train_df, test_df)

    print("\nFeature selection + PCA...")
    X_train_pca, X_test_pca, selector, pca = feature_selection_pca(X_train, X_test, y_train)

    print("\nTraining SGD model...")
    sgd_model = train_sgd(X_train_pca, y_train)

    print("\nTraining Logistic Regression model...")
    lr_model = train_logistic(X_train_pca, y_train)

    print("\nEvaluating SGD (threshold=0.4)...")
    evaluate(sgd_model, X_test_pca, y_test, threshold=0.4)  # lower threshold to detect more attacks

    print("\nEvaluating Logistic Regression (threshold=0.4)...")
    evaluate(lr_model, X_test_pca, y_test, threshold=0.4)

    print("\nEvaluating Ensemble (average of SGD+LR, threshold=0.4)...")
    y_pred_ensemble, _ = ensemble_predict([sgd_model, lr_model], X_test_pca, threshold=0.4)
    acc = accuracy_score(y_test, y_pred_ensemble)
    print(f"\nEnsemble Accuracy: {acc:.4f}")
    print("\nClassification Report (Ensemble):\n", classification_report(y_test, y_pred_ensemble, target_names=['Normal','Attack']))
    print("\nConfusion Matrix (Ensemble):\n", confusion_matrix(y_test, y_pred_ensemble))

    # Save ensemble + preprocessing
    joblib.dump({
        'sgd_model': sgd_model,
        'lr_model': lr_model,
        'scaler': scaler,
        'selector': selector,
        'pca': pca
    }, MODEL_PATH)
    print(f"\nEnhanced model saved as {MODEL_PATH}")

if __name__ == "__main__":
    main()
