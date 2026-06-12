import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from config import DISEASE_CONFIG

os.makedirs("models", exist_ok=True)

results = []

for disease_name, cfg in DISEASE_CONFIG.items():
    print(f"\n{'='*50}\nTraining model for: {disease_name}\n{'='*50}", flush=True)
    sys.stdout.flush()

    try:
        df = pd.read_csv(cfg["file"])
    except FileNotFoundError:
        print(f"  [SKIPPED] File not found: {cfg['file']}")
        continue

    # Drop unwanted columns
    for col in cfg.get("drop_cols", []):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop rows with missing target
    df = df.dropna(subset=[cfg["target"]])

    # Fix medically invalid zeros (e.g. zero glucose/BMI)
    for col in cfg.get("zero_invalid_cols", []):
        if col in df.columns:
            df[col] = df[col].replace(0, df[col].median())

    # Encode categorical columns (including target if categorical)
    encoders = {}
    for col in cfg.get("categorical_cols", []):
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # Fill remaining missing numeric values with median
    for col in df.columns:
        if df[col].dtype != object and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Drop any leftover non-numeric columns Claude/you didn't account for
    non_numeric = df.select_dtypes(include="object").columns.tolist()
    if non_numeric:
        print(f"  [WARNING] Dropping unhandled non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    # Split features/target
    X = df.drop(columns=[cfg["target"]])
    y = df[cfg["target"]]

    # If target is still non-binary/non-numeric somehow, encode it
    if y.dtype == object:
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
        encoders["__target__"] = target_encoder

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"  Accuracy: {acc:.3f}")
    results.append((disease_name, acc))

    # Save everything needed for this disease
    safe_name = disease_name.lower().replace(" ", "_")
    joblib.dump(model, f"models/{safe_name}_model.pkl")
    joblib.dump(scaler, f"models/{safe_name}_scaler.pkl")
    joblib.dump(list(X.columns), f"models/{safe_name}_features.pkl")
    joblib.dump(encoders, f"models/{safe_name}_encoders.pkl")

    print(f"  Saved -> models/{safe_name}_model.pkl, scaler, features, encoders")

print("\n" + "="*50)
print("Training Summary")
print("="*50)
for name, acc in results:
    print(f"{name:20s} -> Accuracy: {acc:.3f}")