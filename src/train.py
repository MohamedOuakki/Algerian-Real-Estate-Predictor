import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os


def load_data(path="data/processed/listings_clean.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["product_price"])
    y = np.log1p(df["product_price"])  # log transform price
    return X, y


def evaluate(model, X_test, y_test, name):
    log_preds = model.predict(X_test)
    preds     = np.expm1(log_preds)   # convert back to DA
    actuals   = np.expm1(y_test)

    mae = mean_absolute_error(actuals, preds)
    r2  = r2_score(y_test, log_preds) # R² on log scale is more meaningful
    print(f"  {name}:")
    print(f"    MAE : {mae:,.0f} DA")
    print(f"    R²  : {r2:.3f}")
    return mae, r2


def train():
    print("Loading data...")
    X, y = load_data()
    print(f"  → {X.shape[0]} rows, {X.shape[1]} features")

    print("\nSplitting into train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost":           XGBRegressor(n_estimators=100, random_state=42, 
                                      n_jobs=-1, verbosity=0),
}

    print("\nTraining and evaluating models...\n")
    results = {}
    trained = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        mae, r2 = evaluate(model, X_test, y_test, name)
        results[name] = {"mae": mae, "r2": r2}
        trained[name] = model

    # pick best model by R²
    best_name  = max(results, key=lambda k: results[k]["r2"])
    best_model = trained[best_name]
    print(f"\nBest model: {best_name} (R² = {results[best_name]['r2']:.3f})")

    # save best model and feature names
    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    print("Model saved to models/best_model.pkl")
    return best_model, X.columns.tolist()


if __name__ == "__main__":
    train()