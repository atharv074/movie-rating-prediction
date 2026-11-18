
# Movie Rating Prediction with Python

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. Load Dataset
DATA_PATH = Path(__file__).resolve().parent / "data" / "Movie dataset.csv"

df = pd.read_csv(DATA_PATH, encoding="latin1")

print("\n========== DATASET INFORMATION ==========")
print("Original dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


# 3. Data Cleaning
df = df.drop_duplicates().copy()

# Convert numerical columns
df["Year"] = pd.to_numeric(
    df["Year"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce"
)

df["Duration"] = pd.to_numeric(
    df["Duration"].astype(str).str.extract(r"(\d+)")[0],
    errors="coerce"
)

df["Votes"] = pd.to_numeric(
    df["Votes"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

# Remove rows without a valid target rating
df = df.dropna(subset=["Rating"])
df = df[df["Rating"].between(0, 10)]

print("\n========== CLEANED DATA ==========")
print("Cleaned dataset shape:", df.shape)


# 4. Select Features and Target
numeric_features = ["Year", "Duration", "Votes"]

categorical_features = [
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3"
]

feature_columns = numeric_features + categorical_features

X = df[feature_columns]
y = df["Rating"]


# 5. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# 6. Create Random Forest Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    ))
])


# 7. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 8. Train Model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

print("Model training completed!")


# 9. Model Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.3f}")


# 10. Baseline Model Comparison
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print("\n========== BASELINE COMPARISON ==========")
print(f"Baseline MAE: {baseline_mae:.3f}")
print(f"Random Forest MAE: {mae:.3f}")

results = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2 Score", "Baseline MAE"],
    "Value": [mae, rmse, r2, baseline_mae]
})

results.to_csv("model_results.csv", index=False)

print("\nResults saved to model_results.csv")

# 11. Feature Importance
trained_model = model.named_steps["regressor"]
trained_preprocessor = model.named_steps["preprocessor"]

feature_names = trained_preprocessor.get_feature_names_out()
importances = trained_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 15 IMPORTANT FEATURES ==========")
print(importance_df.head(15).to_string(index=False))


# 12. Feature Importance Visualization
top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 15 Features Influencing Movie Rating Predictions")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()


# 13. Example Movie Rating Prediction
new_movie = pd.DataFrame([{
    "Year": 2026,
    "Duration": 120,
    "Votes": 100,
    "Genre": "Drama, Thriller",
    "Director": "Unknown Director",
    "Actor 1": "Unknown Actor 1",
    "Actor 2": "Unknown Actor 2",
    "Actor 3": "Unknown Actor 3"
}])

predicted_rating = model.predict(new_movie)[0]

print("\n========== MOVIE RATING PREDICTION ==========")
print("Example Movie Details:")
print(new_movie.to_string(index=False))
print(f"\nPredicted IMDb Rating: {predicted_rating:.2f}/10")

print("\n========== PROJECT COMPLETED ==========")
