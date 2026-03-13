"""
train_model.py
--------------
Generates a synthetic fraud dataset, trains a Random Forest classifier,
evaluates it, and saves the model + scaler as .pkl files.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── 1. Generate synthetic dataset ────────────────────────────────────────────
N_LEGIT  = 9500
N_FRAUD  = 500
N_TOTAL  = N_LEGIT + N_FRAUD

print("=" * 60)
print("  FRAUD DETECTION MODEL TRAINER")
print("=" * 60)
print(f"\n[1/5] Generating synthetic dataset ({N_TOTAL:,} transactions)…")

def make_legit(n):
    return pd.DataFrame({
        "amount":           np.random.lognormal(mean=7.5, sigma=1.2, size=n),  # ~₹1,800 median
        "hour_of_day":      np.random.randint(6, 23, size=n),
        "day_of_week":      np.random.randint(0, 7,  size=n),
        "transaction_type": np.random.choice([0, 1, 2, 3], size=n, p=[0.4, 0.3, 0.2, 0.1]),
        "merchant_category":np.random.choice(range(10), size=n),
        "distance_from_home": np.random.exponential(scale=15, size=n),
        "num_prev_transactions": np.random.randint(0, 50, size=n),
        "account_age_days": np.random.randint(90, 3650, size=n),
        "failed_attempts":  np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        "is_international": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
        "label": 0,
    })

def make_fraud(n):
    return pd.DataFrame({
        "amount":           np.random.lognormal(mean=9.0, sigma=1.8, size=n),  # ~₹8,000+ median
        "hour_of_day":      np.random.choice(list(range(0, 6)) + list(range(22, 24)), size=n),
        "day_of_week":      np.random.randint(0, 7, size=n),
        "transaction_type": np.random.choice([0, 1, 2, 3], size=n, p=[0.1, 0.2, 0.3, 0.4]),
        "merchant_category":np.random.choice(range(10), size=n),
        "distance_from_home": np.random.exponential(scale=80, size=n),
        "num_prev_transactions": np.random.randint(0, 10, size=n),
        "account_age_days": np.random.randint(1, 180, size=n),
        "failed_attempts":  np.random.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3]),
        "is_international": np.random.choice([0, 1], size=n, p=[0.4, 0.6]),
        "label": 1,
    })

df = pd.concat([make_legit(N_LEGIT), make_fraud(N_FRAUD)], ignore_index=True).sample(frac=1, random_state=SEED)
print(f"   Legitimate: {N_LEGIT:,}  |  Fraud: {N_FRAUD:,}")

# ── 2. Pre-process ────────────────────────────────────────────────────────────
print("\n[2/5] Pre-processing…")
FEATURES = [
    "amount", "hour_of_day", "day_of_week", "transaction_type",
    "merchant_category", "distance_from_home", "num_prev_transactions",
    "account_age_days", "failed_attempts", "is_international",
]

X = df[FEATURES]
y = df["label"]

# Clip extreme amounts
X = X.copy()
X["amount"] = X["amount"].clip(upper=40_00_000)  # ₹40 lakh cap

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 3. Train ──────────────────────────────────────────────────────────────────
print("\n[3/5] Training Random Forest classifier…")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
model.fit(X_train_sc, y_train)
print("   Training complete.")

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
print("\n[4/5] Evaluating…")
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print(f"\n   Accuracy : {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall   : {rec:.4f}")
print(f"   F1-Score : {f1:.4f}")
print(f"\n   Confusion Matrix:\n{cm}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Fraud'])}")

# Feature importance
importance = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
print("   Top feature importances:")
for feat, imp in importance[:5]:
    print(f"     {feat:<30} {imp:.4f}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
print("\n[5/5] Saving model artefacts…")
os.makedirs("model_data", exist_ok=True)

with open("model_data/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model_data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature list so app.py always uses the same order
with open("model_data/features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

print("   model_data/fraud_model.pkl ✓")
print("   model_data/scaler.pkl      ✓")
print("   model_data/features.pkl    ✓")
print("\n" + "=" * 60)
print("  Training finished successfully!")
print("=" * 60 + "\n")