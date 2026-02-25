# nsl_kdd.py
# ML Classification on NSL-KDD Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# -----------------------------
# 1. COLUMN NAMES
# -----------------------------
columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','label','difficulty'
]

# -----------------------------
# 2. LOAD DATASET
# -----------------------------
train_df = pd.read_csv("KDDTrain+.txt", names=columns)
test_df = pd.read_csv("KDDTest+.txt", names=columns)

print("Datasets Loaded Successfully")

# -----------------------------
# 3. ENCODE CATEGORICAL DATA
# -----------------------------
encoder = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

# Binary Classification
train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# -----------------------------
# 4. SPLIT FEATURES & LABEL
# -----------------------------
X_train = train_df.drop(['label', 'difficulty'], axis=1)
y_train = train_df['label']

X_test = test_df.drop(['label', 'difficulty'], axis=1)
y_test = test_df['label']

# -----------------------------
# 5. FEATURE SCALING
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 6. MODELS
# -----------------------------
models = {
    "Logistic_Regression": LogisticRegression(max_iter=2000),
    "SVM": SVC(kernel='rbf'),
    "Random_Forest": RandomForestClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=100)
}

accuracies = {}

# -----------------------------
# 7. TRAIN, EVALUATE & SAVE CONFUSION MATRIX
# -----------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

    print(f"{name} Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal','Attack'],
                yticklabels=['Normal','Attack'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close()

# -----------------------------
# 8. ACCURACY COMPARISON GRAPH
# -----------------------------
plt.figure(figsize=(8,5))
plt.bar(accuracies.keys(), accuracies.values())
plt.ylabel("Accuracy")
plt.xlabel("Algorithms")
plt.title("ML Algorithm Performance Comparison (NSL-KDD)")
plt.ylim(0.9, 1.0)
plt.tight_layout()
plt.savefig("algorithm_accuracy_comparison.png")
plt.show()
