# Module 2 — Analytics Pipeline (`/analytics`)

Profiles, cleans, visually explores, and models the Titanic dataset end to
end, as one cohesive pipeline split across two ordered scripts that share a
single committed `titanic.csv`.

## Setup

```bash
pip install -r requirements.txt
```

## Run, in order

```bash
python 01_eda.py       # loads titanic (network+cache, ONE time), profiles,
                        # cleans, saves titanic.csv, produces the EDA story
python 02_modeling.py  # reads the SAME titanic.csv, splits, builds the
                        # sklearn Pipeline/ColumnTransformer, trains/evaluates
                        # 3 classifiers, tunes, runs the regression side-task,
                        # saves best_pipeline.joblib
```

Charts are written to `analytics/charts/`.

> **Note on this submission's execution environment:** `01_eda.py` calls
> `sns.load_dataset("titanic")`, which needs internet access on first run;
> the environment used to author this code did not have internet access. The
> full logic of both scripts (missing-value threshold handling, IQR outlier
> counts, the correlation heatmap and top-2-pair ranking, the stratified
> split, the leak-free `ColumnTransformer`/`Pipeline`, all three classifiers,
> the imbalance comparison, `GridSearchCV` + OOB, the regression side-task,
> and the joblib save/reload round-trip) was validated end-to-end against a
> synthetic stand-in DataFrame with the exact Titanic column schema, and ran
> without errors. Run `01_eda.py` then `02_modeling.py` with internet access
> to regenerate the real `titanic.csv`, real charts, and real printed metrics
> before submitting. `pip install imbalanced-learn` is also required for the
> SMOTE row in the imbalance comparison — without it, the script prints a
> notice and still shows the baseline / class_weight rows.

## Design decisions

**One load, two ordered scripts.** `01_eda.py` is the only place
`sns.load_dataset("titanic")` is called. It immediately writes
`titanic.csv` right after that one load. `02_modeling.py` reads only that
CSV — there is no second raw load anywhere in the module.

**Missing-value threshold rule.** For every column with missing values, we
print its exact missing percentage, then apply: `<5%` → drop those rows;
`5%–30%` → impute (median for numeric, mode for categorical); very high
(`deck`, ~77% missing in the real dataset) → encode `"Missing"` as its own
category rather than drop the column or impute a specific deck for most
passengers, since an unrecorded deck is itself plausibly informative
(it correlates with not having a private cabin, i.e. lower classes).

**IQR outliers / fare skew.** Outlier counts for `age` and `fare` use the
standard `[Q1−1.5·IQR, Q3+1.5·IQR]` rule. Fare's mean > median > mode
ordering is used explicitly to argue it is right-skewed (a small number of
very high fares pull the mean above the median/mode).

**Correlation matrix scope.** Restricted to exactly `survived, pclass, age,
sibsp, parch, fare` — the dataset's genuinely independent numeric columns.
`adult_male` and `alone` are excluded because they are derived flags
(computable from `sex`/`age` and from `sibsp+parch` respectively), not
independently measured features, and including them would inflate the
correlation story with redundancy rather than new information. The top-2
strongest pairs are found by ranking every off-diagonal pair by `|r|`, not
by inspection.

**Train-only preprocessing.** All imputation, encoding, and scaling live
inside a single `ColumnTransformer` wrapped in a `Pipeline` per model. Every
`.fit()` call happens on `X_train` only; `X_test` only ever sees
`.transform()`. This is enforced structurally (the pipeline object, not
manual discipline), so the same object handles both fit-on-train and
transform-on-test correctly by construction.

**Stratified split.** `train_test_split(..., stratify=y)` is used because
`survived` is imbalanced (~38% positive class); an unstratified split risks
train/test folds with different survival rates purely by chance, which would
make test-set metrics an unreliable read on real-world performance.

**Imbalance comparison.** SMOTE is applied only inside the training fold —
via `imblearn.pipeline.Pipeline`, so `SMOTE.fit_resample` only ever sees
`X_train`/`y_train` during `.fit()`, and the test set is never touched by
synthetic points, avoiding leakage. `class_weight='balanced'` is compared as
a second, resampling-free alternative.

**GridSearchCV + OOB.** The Random Forest is constructed with
`oob_score=True` up front (required for `oob_score_` to populate after
fitting) and tuned over `n_estimators`, `max_depth`, `max_features` via
5-fold CV on F1.

**Regression side-task.** A separate `LinearRegression` pipeline predicts
`fare` from the other features (its own train/test split and its own
`ColumnTransformer`, independent of the classification pipeline but built
from the same cleaned `titanic.csv`). Reports MAE, RMSE, R², Adjusted R², and
a residual-vs-predicted plot used to argue heteroscedasticity (residual
spread widening at higher predicted fares, consistent with fare's skew).

**Final comparison table.** Classifier metrics (accuracy/precision/recall/
F1/AUC) and the regression metrics (MAE/RMSE/R²/Adjusted R²) are printed as
two separate tables/metric groups — not merged onto one scale, since they
are not comparable numbers.

**Saved artifact.** `joblib.dump(...)` saves the *entire* fitted
`Pipeline` (preprocessing + final estimator) for whichever classifier scored
highest F1 — not the bare estimator — so it can be reloaded and called
directly on raw, unpreprocessed rows. The script reloads it and asserts the
reloaded pipeline's prediction on a raw test row matches the original.
