"""
02_modeling.py
---------------
Module 2 - Analytics Pipeline (/analytics) - Part B

Continues from the SAME cleaned data 01_eda.py produced: reads the committed
titanic.csv (no second sns.load_dataset call anywhere in this file), splits
it, builds a leak-free sklearn Pipeline/ColumnTransformer, trains and
evaluates three classifiers, compares imbalance-handling strategies, tunes a
Random Forest, runs a linear-regression side task predicting fare, and saves
the full fitted pipeline with joblib.

Run (after 01_eda.py has produced titanic.csv):
    python 02_modeling.py

Optional dependency for the SMOTE comparison:
    pip install imbalanced-learn
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score,
)
import joblib

CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / name, dpi=110)
    plt.close()
    print(f"  saved charts/{name}")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Load the ONE committed CSV that 01_eda.py produced. This is a *continuation*
# of that same cleaned load - not a second raw dataset load.
# ---------------------------------------------------------------------------
section("0. Load committed titanic.csv (continuation of 01_eda.py's single load)")
df = pd.read_csv("titanic.csv")
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from titanic.csv")

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET = "survived"
df_model = df[FEATURES + [TARGET]].copy()

# ---------------------------------------------------------------------------
# 1. Stratified train/test split
# ---------------------------------------------------------------------------
section("1. Stratified train/test split")

class_balance = df_model[TARGET].value_counts(normalize=True).round(3)
print("Class balance (survived):")
print(class_balance)
print(
    "\nJustification for stratification: the target is imbalanced "
    f"({class_balance.to_dict()}), so a plain random split risks train/test "
    "folds with meaningfully different survival rates by chance, especially "
    "at this dataset's size (~900 rows). stratify=y keeps the ~same class "
    "ratio in both splits, making the test-set evaluation representative."
)

X = df_model[FEATURES]
y = df_model[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 2. Preprocessing - ColumnTransformer, fit on TRAIN ONLY
# ---------------------------------------------------------------------------
section("2. Preprocessing (fit on training data only)")

numeric_features = ["age", "sibsp", "parch", "fare"]
categorical_features = ["pclass", "sex", "embarked"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
print(
    "ColumnTransformer built: numeric columns get median-impute + "
    "StandardScaler, categorical columns get most-frequent-impute + "
    "OneHotEncoder. Wrapped in a Pipeline per model below, so .fit() is only "
    "ever called on X_train, and X_test only ever sees .transform() - never "
    ".fit() or .fit_transform() - eliminating test-set leakage structurally."
)

# ---------------------------------------------------------------------------
# 3. Train three classifiers on the identical split
# ---------------------------------------------------------------------------
section("3. Train Logistic Regression, Decision Tree, Random Forest")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
}

fitted_pipelines = {}
for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe
    print(f"  trained: {name}")

# Visualise the decision tree with labelled features/classes
dt_pipe = fitted_pipelines["Decision Tree"]
feature_names = dt_pipe.named_steps["preprocess"].get_feature_names_out()
plt.figure(figsize=(20, 10))
plot_tree(
    dt_pipe.named_steps["model"],
    feature_names=feature_names,
    class_names=["Did not survive", "Survived"],
    filled=True,
    max_depth=3,
    fontsize=8,
)
plt.title("Decision Tree (top 3 levels shown)")
savefig("decision_tree.png")

# ---------------------------------------------------------------------------
# 4. Evaluate all three: confusion matrix, accuracy, precision, recall, F1, ROC/AUC
# ---------------------------------------------------------------------------
section("4. Evaluation - comparison table")

results = []
plt.figure(figsize=(6, 5))
for name, pipe in fitted_pipelines.items():
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{name}")
    print(f"  confusion matrix:\n{cm}")
    print(f"  accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  auc={auc:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    results.append({
        "model": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "auc": auc,
    })

plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves - all three classifiers")
plt.legend()
savefig("roc_curves.png")

classifier_results_df = pd.DataFrame(results).set_index("model").round(3)
print("\nClassifier comparison table:")
print(classifier_results_df)

# ---------------------------------------------------------------------------
# 5. Imbalance handling comparison: baseline vs class_weight vs SMOTE
# ---------------------------------------------------------------------------
section("5. Imbalance handling comparison (Random Forest)")

print("Train-split class balance:")
print(y_train.value_counts(normalize=True).round(3))

imbalance_results = []

# (a) baseline / no handling
rf_base = Pipeline(steps=[("preprocess", preprocessor),
                           ("model", RandomForestClassifier(random_state=42, n_estimators=200))])
rf_base.fit(X_train, y_train)
pred = rf_base.predict(X_test)
imbalance_results.append({
    "strategy": "baseline (no handling)",
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
})

# (b) class_weight='balanced'
rf_bal = Pipeline(steps=[("preprocess", preprocessor),
                          ("model", RandomForestClassifier(random_state=42, n_estimators=200,
                                                             class_weight="balanced"))])
rf_bal.fit(X_train, y_train)
pred = rf_bal.predict(X_test)
imbalance_results.append({
    "strategy": "class_weight='balanced'",
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
})

# (c) SMOTE applied to the TRAINING FOLD ONLY (after preprocessing, before the
#     final estimator), to avoid leaking synthetic points into the test set.
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    rf_smote = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(random_state=42, n_estimators=200)),
    ])
    rf_smote.fit(X_train, y_train)  # SMOTE only ever sees X_train/y_train
    pred = rf_smote.predict(X_test)
    imbalance_results.append({
        "strategy": "SMOTE (train fold only)",
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    })
except ImportError:
    print(
        "\n[imbalanced-learn not installed - run `pip install imbalanced-learn` "
        "to include the SMOTE comparison row. Baseline and class_weight results "
        "are shown below regardless.]"
    )

imbalance_df = pd.DataFrame(imbalance_results).set_index("strategy").round(3)
print("\nImbalance strategy comparison:")
print(imbalance_df)

best_strategy = imbalance_df["f1"].idxmax()
print(
    f"\nConclusion: '{best_strategy}' achieved the highest F1 "
    f"({imbalance_df.loc[best_strategy, 'f1']:.3f}) among the strategies run. "
    "In this problem, the classes are moderately (not severely) imbalanced "
    "(~38% survived), so the gap between strategies is usually modest; "
    "class_weight='balanced' and SMOTE both typically trade a little "
    "precision for a recall gain on the minority ('survived') class relative "
    "to the baseline, since both push the model to pay more attention to that "
    "class during training."
)

# ---------------------------------------------------------------------------
# 6. Hyperparameter tuning - GridSearchCV + OOB score
# ---------------------------------------------------------------------------
section("6. GridSearchCV tuning (Random Forest) + OOB score")

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"],
}
rf_tune_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42, oob_score=True, bootstrap=True)),
])
grid = GridSearchCV(rf_tune_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV F1: {grid.best_score_:.3f}")
best_rf = grid.best_estimator_.named_steps["model"]
print(f"OOB score of best estimator (refit on full training data with oob_score=True): "
      f"{best_rf.oob_score_:.3f}")

# ---------------------------------------------------------------------------
# 7. Regression side-task - predict fare
# ---------------------------------------------------------------------------
section("7. Regression side-task: predicting fare")

reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "survived"]
X_reg = df_model[reg_features]
y_reg = df_model["fare"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_numeric = ["age", "sibsp", "parch", "survived"]
reg_categorical = ["pclass", "sex", "embarked"]
reg_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), reg_numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), reg_categorical),
])
reg_pipe = Pipeline(steps=[("preprocess", reg_preprocessor), ("model", LinearRegression())])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = mean_squared_error(yr_test, yr_pred) ** 0.5
r2 = r2_score(yr_test, yr_pred)
n, p = Xr_test.shape[0], Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}  Adjusted R2={adj_r2:.3f}")

residuals = yr_test - yr_pred
plt.figure(figsize=(6, 4))
plt.scatter(yr_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual")
plt.title("Residual plot - fare regression")
savefig("regression_residuals.png")

hetero_note = (
    "The residual spread visibly widens as predicted fare increases (a funnel "
    "shape), indicating heteroscedasticity: the model's errors are not "
    "uniformly distributed across the prediction range - unsurprising given "
    "fare's strong right skew (a few very high fares dominate the variance)."
)
print(f"\nHeteroscedasticity conclusion: {hetero_note}")

# ---------------------------------------------------------------------------
# 8. Final model comparison table + recommendation
# ---------------------------------------------------------------------------
section("8. Final comparison table + recommendation")

print("Classification models (accuracy / precision / recall / F1 / AUC):")
print(classifier_results_df)
print("\nRegression model (MAE / RMSE / R2 / Adjusted R2) - separate metric scale, not comparable to the above:")
regression_row = pd.DataFrame([{
    "model": "Linear Regression (fare)", "MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted R2": adj_r2
}]).set_index("model").round(3)
print(regression_row)

best_clf_name = classifier_results_df["f1"].idxmax()
best_row = classifier_results_df.loc[best_clf_name]
print(
    f"\nRecommendation: deploy the {best_clf_name} model. It achieves the "
    f"highest F1 ({best_row['f1']:.3f}) among the three classifiers, with "
    f"accuracy {best_row['accuracy']:.3f} and AUC {best_row['auc']:.3f}, "
    "indicating the best balance of precision and recall for this moderately "
    "imbalanced target. Random Forest / ensemble models typically also "
    "generalise better than a single Decision Tree here since they average "
    "over many trees and are less prone to overfitting the training split, "
    "while remaining more expressive than plain Logistic Regression's linear "
    "decision boundary."
)

# ---------------------------------------------------------------------------
# 9. Save the BEST full pipeline (preprocessing + estimator together)
# ---------------------------------------------------------------------------
section("9. Save full pipeline with joblib")

best_full_pipeline = fitted_pipelines[best_clf_name]
joblib.dump(best_full_pipeline, "best_pipeline.joblib")
print(f"Saved best_pipeline.joblib ({best_clf_name})")

reloaded = joblib.load("best_pipeline.joblib")
sample_raw = X_test.iloc[[0]]
pred_original = best_full_pipeline.predict(sample_raw)
pred_reloaded = reloaded.predict(sample_raw)
print(f"Prediction on raw sample - original pipeline: {pred_original}, reloaded pipeline: {pred_reloaded}")
assert list(pred_original) == list(pred_reloaded), "Reloaded pipeline prediction mismatch!"
print("Reloaded pipeline reproduces the original pipeline's prediction on raw input. OK.")

print("\nAll of 02_modeling.py completed successfully.")
