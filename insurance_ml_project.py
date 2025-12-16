# =========================================================
# 1. IMPORTS
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 2. LOAD DATA
# =========================================================
df = pd.read_csv("insurance.csv")

print("Dataset shape:", df.shape)
print(df.head())
print(df.info())

# =========================================================
# 3. FEATURES & TARGET
# =========================================================
X = df.drop("charges", axis=1)
y = df["charges"]

categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]

# =========================================================
# 4. PREPROCESSING
# =========================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

# =========================================================
# 5. TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# 6. BASELINE MODEL — LINEAR REGRESSION
# =========================================================
lr_model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regressor", LinearRegression())
    ]
)

lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\nLINEAR REGRESSION (Baseline)")
print(f"MAE  : {lr_mae:.2f}")
print(f"RMSE : {lr_rmse:.2f}")
print(f"R²   : {lr_r2:.3f}")

# =========================================================
# 7. IMPROVED MODEL — RANDOM FOREST
# =========================================================
rf_model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRANDOM FOREST (Improved)")
print(f"MAE  : {rf_mae:.2f}")
print(f"RMSE : {rf_rmse:.2f}")
print(f"R²   : {rf_r2:.3f}")

# =========================================================
# 8. FEATURE IMPORTANCE (RANDOM FOREST)
# =========================================================
ohe = rf_model.named_steps["preprocessing"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)

feature_names = np.concatenate([cat_feature_names, numerical_features])
importances = rf_model.named_steps["regressor"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTOP FEATURE IMPORTANCE")
print(importance_df.head(10))

# =========================================================
# 9. ALL VISUALIZATIONS IN ONE WINDOW
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1️. Distribution of charges
sns.histplot(df["charges"], bins=40, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Insurance Charges")

# 2️. Charges vs smoker

sns.boxplot(x="smoker", y="charges", data=df, ax=axes[0, 1])
axes[0, 1].set_title("Charges by Smoking Status")

# 3️. Actual vs Predicted (Random Forest)
sns.scatterplot(x=y_test, y=rf_pred, ax=axes[1, 0])
axes[1, 0].set_xlabel("Actual charges")
axes[1, 0].set_ylabel("Predicted charges")
axes[1, 0].set_title("Random Forest: Actual vs Predicted")

# 4️. Feature importance (Top 10)
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df.head(10),
    ax=axes[1, 1]
)
axes[1, 1].set_title("Top 10 Feature Importance")

plt.tight_layout()
plt.show()