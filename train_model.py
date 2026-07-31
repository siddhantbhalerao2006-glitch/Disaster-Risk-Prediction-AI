import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from data_loader import generate_maharashtra_dataset

RANDOM_STATE = 42

def train():
    print("Step 1: Generating/Loading Maharashtra climate-disaster dataset...")
    base_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else "."
    csv_path = os.path.join(base_dir, "dataset_maharashtra.csv")
    
    if os.path.exists(csv_path):
        print(f"Loading existing dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        print("Dataset file not found. Generating synthetic dataset...")
        df = generate_maharashtra_dataset(n_samples=5000)
        df.to_csv(csv_path, index=False)
        print(f"Dataset generated and saved to {csv_path}.")
        
    print(f"Total rows in dataset: {len(df)}")
    
    # Step 2: Preprocessing and Encoding
    print("\nStep 2: Preprocessing and encoding categorical features...")
    district_encoder = LabelEncoder()
    df["district_enc"] = district_encoder.fit_transform(df["district"])
    
    disaster_encoder = LabelEncoder()
    df["disaster_type_enc"] = disaster_encoder.fit_transform(df["disaster_type"])
    
    # Map target risk_level to logical integers (ordered from Low to Severe)
    risk_mapping = {"Low": 0, "Medium": 1, "High": 2, "Severe": 3}
    risk_reverse_mapping = {0: "Low", 1: "Medium", 2: "High", 3: "Severe"}
    df["risk_level_enc"] = df["risk_level"].map(risk_mapping)
    
    # Define features and target
    feature_cols = [
        "district_enc", "pop_density", "slope", "coastal", "vulnerability_index",
        "rainfall", "wind_speed", "temperature", "humidity", "soil_moisture",
        "river_level"
    ]
    X = df[feature_cols]
    y = df["risk_level_enc"]
    
    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Step 3: Hyperparameter Tuning via GridSearchCV
    print("\nStep 3: Training Random Forest model with hyperparameter tuning...")
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [10, 15, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [2, 4]
    }
    
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Optimal Parameters: {grid_search.best_params_}")
    
    # Step 4: Model Evaluation
    print("\nStep 4: Evaluating the optimized model...")
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    target_names = [risk_reverse_mapping[i] for i in range(4)]
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=target_names)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("\n=== Test Set Classification Report ===")
    print(report_str)
    print("=== Confusion Matrix ===")
    print(conf_matrix)
    
    print("\nRunning 5-Fold Cross Validation on full dataset...")
    cv_scores = cross_val_score(best_model, X, y, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"5-Fold CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    # Step 5: Feature Importances
    print("\nStep 5: Logging Feature Importances...")
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    importance_path = os.path.join(base_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print("Feature importances saved to feature_importance.csv:")
    print(importance_df.to_string(index=False))
    
    # Step 6: Serialize model bundle
    print("\nStep 6: Serializing model and metadata...")
    model_bundle = {
        "model": best_model,
        "district_encoder": district_encoder,
        "disaster_encoder": disaster_encoder,
        "risk_mapping": risk_mapping,
        "risk_reverse_mapping": risk_reverse_mapping,
        "feature_cols": feature_cols,
        "metrics": {
            "test_accuracy": float(test_accuracy),
            "cv_mean": float(cv_mean),
            "cv_std": float(cv_std),
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist()
        }
    }
    
    model_path = os.path.join(base_dir, "disaster_risk_model.pkl")
    joblib.dump(model_bundle, model_path)
    print(f"Saved trained model bundle successfully to {model_path}")

if __name__ == "__main__":
    train()
