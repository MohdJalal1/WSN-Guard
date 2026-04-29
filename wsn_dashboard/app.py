from flask import Flask, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import plotly.express as px
import os

app = Flask(__name__)

# ---------------- COLUMNS ----------------
COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label','difficulty_level'
]

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_PATH  = os.path.join(DATA_DIR, "KDDTest+.txt")

# ---------------- DATA LOADER ----------------
def load_and_prepare():

    # Safety check
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Missing file: {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Missing file: {TEST_PATH}")

    # Load data
    train_df = pd.read_csv(TRAIN_PATH, names=COLUMNS)
    test_df  = pd.read_csv(TEST_PATH, names=COLUMNS)

    # Encode categorical columns
    cat_cols = ['protocol_type', 'service', 'flag']
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])

    # Binary labels
    train_df['label_binary'] = train_df['label'].apply(
        lambda x: 'normal' if x == 'normal' else 'attack'
    )

    # Original class distribution
    class_counts = (
        train_df['label_binary']
        .value_counts()
        .reset_index()
    )
    class_counts.columns = ['class', 'count']

    # Prepare features
    drop_cols = ['label', 'label_binary']
    if 'difficulty_level' in train_df.columns:
        drop_cols.append('difficulty_level')

    X = train_df.drop(columns=drop_cols)
    y = train_df['label_binary']

    # SMOTE (only to compute balanced counts)
    smote = SMOTE(random_state=42)
    _, y_bal = smote.fit_resample(X, y)

    balanced_counts = (
        pd.Series(y_bal)
        .value_counts()
        .reset_index()
    )
    balanced_counts.columns = ['class', 'count']

    return train_df, class_counts, balanced_counts

# ---------------- ROUTE ----------------
@app.route("/")
def index():

    train_df, class_counts, balanced_counts = load_and_prepare()

    # Table sample
    sample_df_html = train_df.head(50).to_html(
        classes="table table-striped table-bordered table-sm",
        index=False
    )

    # Stats
    total_rows = len(train_df)
    num_features = train_df.shape[1] - 2
    num_normal = int(class_counts[class_counts['class'] == 'normal']['count'].values[0])
    num_attack = int(class_counts[class_counts['class'] == 'attack']['count'].values[0])

    # Plot 1
    fig = px.bar(
        class_counts,
        x="class", y="count",
        title="Class Distribution (Original)",
        color="class", text="count"
    )
    fig.update_layout(template="plotly_dark")
    graph_json = fig.to_json()

    # Plot 2
    fig_bal = px.bar(
        balanced_counts,
        x="class", y="count",
        title="Class Distribution (After SMOTE)",
        color="class", text="count"
    )
    fig_bal.update_layout(template="plotly_dark")
    graph_bal_json = fig_bal.to_json()

    return render_template(
        "index.html",
        table_html=sample_df_html,
        total_rows=total_rows,
        num_features=num_features,
        num_normal=num_normal,
        num_attack=num_attack,
        graphJSON=graph_json,
        graphBalJSON=graph_bal_json
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
