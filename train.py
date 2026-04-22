import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Dataset.csv')
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.replace(r'\s+', '', regex=True)
    .str.replace(r'[^\w]', '', regex=True)
)

# print(df.columns)
df = df.sort_values(['Patient_ID', 'Hour']).reset_index(drop=True)

# FIX: Save Patient_ID before groupby-apply, restore if pandas drops it (pandas 2.x bug)
patient_ids = df['Patient_ID'].copy()
df = df.groupby('Patient_ID', group_keys=False).apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
if 'Patient_ID' not in df.columns:
    df['Patient_ID'] = patient_ids

df = df.drop(columns=[
    'EtCO2',
    'Bilirubin_direct',
    'TroponinI',
    'Fibrinogen',
    'BaseExcess',
    'HCO3',
    'FiO2',
    'pH',
    'PaCO2',
    'SaO2',
    'AST',
    'Alkalinephos',
    'Bilirubin_total'
])

df = df.fillna(df.median(numeric_only=True))

X = df.drop(columns=['SepsisLabel'])
y = df['SepsisLabel']
groups = df['Patient_ID']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

cols_to_drop = ['Patient_ID', 'Unnamed: 0']
X_train = X_train_raw.drop(columns=[c for c in cols_to_drop if c in X_train_raw.columns])
X_test = X_test_raw.drop(columns=[c for c in cols_to_drop if c in X_test_raw.columns])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- MODEL 1: Logistic Regression ---
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# --- MODEL 2: Random Forest ---
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train_scaled, y_train)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

# --- MODEL 3: XGBoost ---
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(scale_pos_weight=scale_weight, max_depth=6, learning_rate=0.1, n_estimators=100, random_state=42)
xgb.fit(X_train_scaled, y_train)
y_prob_xgb = xgb.predict_proba(X_test_scaled)[:, 1]

models = {
    "Logistic Regression": y_prob_lr,
    "Random Forest": y_prob_rf,
    "XGBoost": y_prob_xgb
}

print(f"{'Model':<20} | {'ROC-AUC':<10} | {'Accuracy':<10}")
print("-" * 45)
for name, probs in models.items():
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, probs > 0.5)
    print(f"{name:<20} | {auc:<10.4f} | {acc:<10.4f}")

importances = xgb.feature_importances_
feature_names = X_train.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis')
plt.title('Top 15 Clinical Predictors of Sepsis (XGBoost)', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Medical Feature', fontsize=12)
plt.show()

print("Top 5 Features that affect the output most:")
print(feature_importance_df.head(5) * 100)
