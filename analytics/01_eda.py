"""
01_eda.py
---------
Module 2 - Analytics Pipeline (/analytics) - Part A

Loads the Titanic dataset via seaborn (network+cache on first run), profiles
it, cleans missing values per an explicit percentage-based threshold rule,
tells a visual data story, and saves the cleaned data as titanic.csv - the
single committed offline fallback that 02_modeling.py reads from, so the raw
dataset is loaded from network/cache exactly ONCE across the whole module.

Outputs (written to ./charts/):
  age_hist.png, age_box.png, fare_hist.png, fare_box.png,
  correlation_heatmap.png, chart_survival_by_sex.png,
  chart_survival_by_pclass.png, chart_age_dist_by_survival.png,
  chart_fare_by_class_survival.png
and titanic.csv in the current directory.

Run:
    python 01_eda.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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
# 1. LOAD (the one and only network/cache load of the raw dataset) + PROFILE
# ---------------------------------------------------------------------------
section("1. Load + profile")

df = sns.load_dataset("titanic")

# Save the offline fallback immediately, right after the one raw load, so the
# rest of this module (and 02_modeling.py) never needs a second network call.
df.to_csv("titanic.csv", index=False)
print("Saved committed offline fallback -> titanic.csv")

print("\ndf.shape:", df.shape)
print("\ndf.info():")
df.info()
print("\ndf.describe():")
print(df.describe(include="all").T)

missing_pct = (df.isna().mean() * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
print("\nPercentage missing per column (only columns with missing values):")
print(missing_pct)

# ---------------------------------------------------------------------------
# 2. MISSING VALUE HANDLING - explicit threshold rule
#    < 5% missing  -> drop those rows
#    5%-30% missing -> impute
#    very high missing (unreliable to impute) -> drop column or "missing" category
# ---------------------------------------------------------------------------
section("2. Missing value handling (threshold rule)")

df_clean = df.copy()

for col, pct in missing_pct.items():
    if pct < 5:
        n_before = len(df_clean)
        df_clean = df_clean[df_clean[col].notna()]
        print(f"- {col}: {pct}% missing (<5%)  -> dropped {n_before - len(df_clean)} rows")
    elif pct <= 30:
        if df_clean[col].dtype in (np.float64, np.int64):
            fill_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(fill_val)
            print(f"- {col}: {pct}% missing (5-30%) -> imputed with median ({fill_val:.2f})")
        else:
            fill_val = df_clean[col].mode(dropna=True)[0]
            df_clean[col] = df_clean[col].fillna(fill_val)
            print(f"- {col}: {pct}% missing (5-30%) -> imputed with mode ('{fill_val}')")
    else:
        # very high missing rate: imputation would be unreliable.
        if col == "deck":
            # deck is ~77% missing in the standard titanic dataset. Rather than
            # invent a deck for most passengers, encode "missing" as its own
            # category - the absence of a recorded deck is itself informative
            # (unrecorded decks correlate strongly with lower classes / no cabin).
            df_clean[col] = df_clean[col].astype(object).fillna("Missing")
            print(f"- {col}: {pct}% missing (very high) -> encoded 'Missing' as its own category "
                  f"(dropping the column would lose a plausibly informative signal; "
                  f"imputing a specific deck for most passengers would be unreliable)")
        else:
            df_clean = df_clean.drop(columns=[col])
            print(f"- {col}: {pct}% missing (very high) -> column dropped (imputation unreliable)")

print(f"\nShape after cleaning: {df_clean.shape}")

# ---------------------------------------------------------------------------
# 3. UNIVARIATE ANALYSIS - age & fare
# ---------------------------------------------------------------------------
section("3. Univariate analysis: age & fare")

for col in ["age", "fare"]:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_clean[col], kde=True)
    plt.title(f"Histogram of {col}")
    savefig(f"{col}_hist.png")

    plt.figure(figsize=(4, 5))
    sns.boxplot(y=df_clean[col])
    plt.title(f"Box plot of {col}")
    savefig(f"{col}_box.png")

    q1, q3 = df_clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    print(f"{col}: Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} bounds=[{lower:.2f}, {upper:.2f}] "
          f"-> {n_outliers} outlier(s)")

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]
print(f"\nfare: mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
skew_note = (
    "right-skewed (mean > median > mode, a long tail of high fares pulls the mean up)"
    if fare_mean > fare_median > fare_mode
    else "not clearly right-skewed by the mean>median>mode ordering - see printed values"
)
print(f"Skewness interpretation: fare is {skew_note}.")

# ---------------------------------------------------------------------------
# 4. BIVARIATE ANALYSIS - survival rate breakdowns + correlation heatmap
# ---------------------------------------------------------------------------
section("4. Bivariate analysis")

surv_by_sex = df_clean.groupby("sex")["survived"].mean().round(3)
print("\nSurvival rate by sex (boolean masking):")
for sex_val in df_clean["sex"].unique():
    mask = df_clean["sex"] == sex_val
    rate = df_clean.loc[mask, "survived"].mean()
    print(f"  sex == '{sex_val}': {rate:.3f}")

print("\nSurvival rate by pclass (boolean masking):")
for pclass_val in sorted(df_clean["pclass"].unique()):
    mask = df_clean["pclass"] == pclass_val
    rate = df_clean.loc[mask, "survived"].mean()
    print(f"  pclass == {pclass_val}: {rate:.3f}")

print("\nSurvival rate by sex & pclass together (boolean masking with &):")
for sex_val in df_clean["sex"].unique():
    for pclass_val in sorted(df_clean["pclass"].unique()):
        mask = (df_clean["sex"] == sex_val) & (df_clean["pclass"] == pclass_val)
        rate = df_clean.loc[mask, "survived"].mean()
        print(f"  sex=='{sex_val}' & pclass=={pclass_val}: {rate:.3f}")

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df_clean[corr_cols].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix (survived, pclass, age, sibsp, parch, fare)")
savefig("correlation_heatmap.png")

# find top-2 strongest off-diagonal absolute correlations
pairs = []
for i, c1 in enumerate(corr_cols):
    for j, c2 in enumerate(corr_cols):
        if i < j:
            pairs.append((c1, c2, abs(corr.loc[c1, c2]), corr.loc[c1, c2]))
pairs.sort(key=lambda x: x[2], reverse=True)
print("\nTop 2 strongest correlation pairs (by |r|):")
for c1, c2, absr, r in pairs[:2]:
    print(f"  {c1} vs {c2}: r = {r:.3f}")
print(
    f"\nInterpretation: {pairs[0][0]}-{pairs[0][1]} (r={pairs[0][3]:.2f}) and "
    f"{pairs[1][0]}-{pairs[1][1]} (r={pairs[1][3]:.2f}) are the strongest linear "
    f"relationships. A negative pclass-fare-style or pclass-survived relationship "
    f"reflects that lower pclass numbers (1st class) paid higher fares and had "
    f"higher survival odds; sibsp-parch typically correlate positively since both "
    f"describe family-group size on board."
)

# ---------------------------------------------------------------------------
# 5. MULTIVARIATE "DATA STORY" - >= 4 charts, each with written interpretation
# ---------------------------------------------------------------------------
section("5. Multivariate data story")

plt.figure(figsize=(5, 4))
sns.barplot(data=df_clean, x="sex", y="survived", errorbar=None)
plt.title("Survival rate by sex")
savefig("chart_survival_by_sex.png")
print(
    "Chart 1 (survival by sex): Women survived at a much higher rate than men "
    "(consistent with 'women and children first' evacuation priority). Sex is "
    "the single strongest categorical predictor of survival in this dataset."
)

plt.figure(figsize=(5, 4))
sns.barplot(data=df_clean, x="pclass", y="survived", errorbar=None)
plt.title("Survival rate by passenger class")
savefig("chart_survival_by_pclass.png")
print(
    "Chart 2 (survival by pclass): Survival rate falls steadily from 1st to 3rd "
    "class. 1st class cabins sat closer to the boat deck and 1st class "
    "passengers likely had faster access to lifeboats, compounding the fare/"
    "wealth advantage already visible in the correlation heatmap."
)

plt.figure(figsize=(6, 4))
sns.kdeplot(data=df_clean, x="age", hue="survived", common_norm=False, fill=True, alpha=0.4)
plt.title("Age distribution by survival outcome")
savefig("chart_age_dist_by_survival.png")
print(
    "Chart 3 (age distribution by survival): Survivors skew slightly younger, "
    "with a visible bump among young children - consistent with children being "
    "prioritised during evacuation, though the effect is much weaker than sex "
    "or class."
)

plt.figure(figsize=(6, 4))
sns.boxplot(data=df_clean, x="pclass", y="fare", hue="survived")
plt.title("Fare by class, split by survival")
savefig("chart_fare_by_class_survival.png")
print(
    "Chart 4 (fare by class & survival): Within every class, survivors tended "
    "to have paid somewhat higher fares than non-survivors of the same class - "
    "suggesting fare captures within-class advantages (e.g. cabin location) "
    "beyond what pclass alone explains."
)

# ---------------------------------------------------------------------------
# 6. EXPLORATORY z-score standardization check (EDA-stage only; NOT used by
#    the modeling pipeline, which does its own train-only scaling in Task 8)
# ---------------------------------------------------------------------------
section("6. Exploratory z-score standardization check (age, fare)")

print("Before standardization:")
print(df_clean[["age", "fare"]].agg(["mean", "std"]))

standardized = df_clean[["age", "fare"]].copy()
for col in ["age", "fare"]:
    standardized[col] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()

print("\nAfter z-score standardization (z = (x - mean) / std):")
print(standardized.agg(["mean", "std"]).round(6))
print(
    "\nAs expected, standardized age and fare have (approximately) mean 0 and "
    "std 1. This is an EDA sanity check only; the modeling pipeline in "
    "02_modeling.py fits its own StandardScaler on the training split alone."
)

print("\nDone. titanic.csv and charts/ are ready for 02_modeling.py.")
