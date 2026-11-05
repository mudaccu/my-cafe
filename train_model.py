"""TMUC Unit 25 - synthetic campus cafe queue regression teaching demo.

Run: python train_model.py
Outputs: campus_cafe_practice.csv, cafe_pipeline.joblib, model_results.json
All dataset rows are invented, not observations of actual customers.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(25)
N = 360
queue = RNG.integers(0, 35, size=N)
counters = RNG.integers(1, 5, size=N)
items = RNG.integers(1, 6, size=N)
periods = RNG.choice(["Normal", "Rush"], size=N, p=[0.62, 0.38])
# Synthetic mathematical relationship plus random noise, not measured cafe behavior.
minutes = (2.7 + 1.04 * queue / counters + 1.32 * items
           + 3.1 * (periods == "Rush")
           + RNG.normal(0, 1.35, size=N))
minutes = np.round(np.clip(minutes, 1, 60), 1)
df = pd.DataFrame({
    "queue_people": queue,
    "open_counters": counters,
    "order_items": items,
    "service_period": periods,
    "wait_minutes": minutes,
})
df.to_csv(HERE / "campus_cafe_practice.csv", index=False)

FEATURES = ["queue_people", "open_counters", "order_items", "service_period"]
X = df[FEATURES]
y = df["wait_minutes"]
# Hold out the test set. Do not use it for model selection.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=25)

# Encode a category inside the pipeline, fitted only on X_train.
preprocess = ColumnTransformer(
    transformers=[
        ("period", OneHotEncoder(handle_unknown="ignore"), ["service_period"]),
        ("numbers", "passthrough", ["queue_people", "open_counters", "order_items"]),
    ]
)
baseline = DummyRegressor(strategy="median")
baseline.fit(X_train, y_train)
model = Pipeline([
    ("prepare", preprocess),
    ("regressor", DecisionTreeRegressor(max_depth=6, min_samples_leaf=4, random_state=25)),
])
model.fit(X_train, y_train)

base_pred = baseline.predict(X_test)
pred = model.predict(X_test)
results = {
    "dataset_type": "synthetic classroom illustration; not field measurements",
    "rows": len(df),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "target": "wait_minutes",
    "baseline_test_mae_min": round(float(mean_absolute_error(y_test, base_pred)), 3),
    "tree_test_mae_min": round(float(mean_absolute_error(y_test, pred)), 3),
    "feature_columns": FEATURES,
}
# Only load model files created and verified by yourself or a trusted instructor.
joblib.dump(model, HERE / "cafe_pipeline.joblib")
(HERE / "model_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("Created:", (HERE / "campus_cafe_practice.csv").name)
print("Saved model + preprocessing:", (HERE / "cafe_pipeline.joblib").name)
print("Train/test rows:", len(X_train), "/", len(X_test))
print("Baseline test MAE (minutes):", results["baseline_test_mae_min"])
print("Decision-tree test MAE (minutes):", results["tree_test_mae_min"])
print("These scores describe the synthetic classroom dataset ONLY.")
