#!/usr/bin/env python3
"""
ML Training Script for FIRE_UdeA Classification
Random Forest vs Gradient Boosting
"""

import sys
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from io import StringIO

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

RANDOM_STATE = 42

NUM_COLS = [
    "ingresos_totales",
    "gastos_personal",
    "liquidez",
    "dias_efectivo",
    "cfo",
    "participacion_ley30",
    "participacion_regalias",
    "participacion_servicios",
    "participacion_matriculas",
    "hhi_fuentes",
    "endeudamiento",
    "tendencia_ingresos",
    "gp_ratio"
]

CAT_COLS = ["unidad", "anio"]


def load_and_prepare_data(csv_content: str):
    """Load CSV and prepare data"""
    data = pd.read_csv(StringIO(csv_content))
    
    # Clean data
    data.drop_duplicates(inplace=True)
    
    # Get actual columns
    num_features = [col for col in NUM_COLS if col in data.columns]
    cat_features = [col for col in CAT_COLS if col in data.columns]
    
    # Fill numeric nulls with median
    for col in num_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())
    
    # Fill categorical nulls
    for col in cat_features:
        data[col] = data[col].fillna("Unknown")
    
    return data, num_features, cat_features


def get_eda_stats(data, num_features):
    """Get EDA statistics"""
    stats = {
        "total_samples": len(data),
        "features_count": len(data.columns) - 1,
        "label_distribution": data["label"].value_counts().to_dict(),
        "numeric_stats": {}
    }
    
    for col in num_features:
        if col in data.columns:
            stats["numeric_stats"][col] = {
                "mean": float(data[col].mean()),
                "median": float(data[col].median()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max())
            }
    
    # Correlation matrix
    corr = data[num_features].corr().fillna(0)
    stats["correlation_matrix"] = corr.to_dict()
    
    return stats


def train_models(data, num_features, cat_features, use_grid_search=False):
    """Train RF and GB models"""
    
    X = data.drop(columns="label")
    y = data["label"]
    
    # Split 60/20/20
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    
    # Preprocessor
    num_transformer = Pipeline([("scaler", StandardScaler())])
    
    transformers = [("num", num_transformer, num_features)]
    if cat_features:
        cat_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(("cat", cat_transformer, cat_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Pipelines
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    
    gb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])
    
    if use_grid_search:
        # Grid Search for RF
        rf_param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__class_weight": [None, "balanced"]
        }
        
        rf_grid = GridSearchCV(
            rf_pipeline, rf_param_grid, cv=3, scoring="accuracy", n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        rf_model = rf_grid.best_estimator_
        rf_best_params = rf_grid.best_params_
        rf_cv_score = rf_grid.best_score_
        
        # Grid Search for GB
        gb_param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 5]
        }
        
        gb_grid = GridSearchCV(
            gb_pipeline, gb_param_grid, cv=3, scoring="accuracy", n_jobs=-1
        )
        gb_grid.fit(X_train, y_train)
        gb_model = gb_grid.best_estimator_
        gb_best_params = gb_grid.best_params_
        gb_cv_score = gb_grid.best_score_
    else:
        # Quick training without grid search
        rf_model = rf_pipeline.fit(X_train, y_train)
        gb_model = gb_pipeline.fit(X_train, y_train)
        rf_best_params = {"n_estimators": 100, "max_depth": None}
        gb_best_params = {"n_estimators": 100, "learning_rate": 0.1}
        rf_cv_score = 0.0
        gb_cv_score = 0.0
    
    # Evaluate both models
    results = {
        "random_forest": evaluate_model(rf_model, X_train, y_train, X_val, y_val, X_test, y_test, num_features),
        "gradient_boosting": evaluate_model(gb_model, X_train, y_train, X_val, y_val, X_test, y_test, num_features),
        "best_params": {
            "random_forest": {k.replace("classifier__", ""): v for k, v in rf_best_params.items()},
            "gradient_boosting": {k.replace("classifier__", ""): v for k, v in gb_best_params.items()}
        },
        "cv_scores": {
            "random_forest": float(rf_cv_score),
            "gradient_boosting": float(gb_cv_score)
        },
        "data_splits": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test)
        }
    }
    
    return results


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, num_features):
    """Evaluate model on train, val, and test sets"""
    
    def get_metrics(model, X, y, set_name):
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        cm = confusion_matrix(y, y_pred)
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y, y_prob)),
            "confusion_matrix": cm.tolist(),
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            }
        }
    
    # Get feature importances
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except:
        feature_names = num_features
    
    importances = classifier.feature_importances_.tolist()
    
    # Sort by importance
    feature_importance = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:15]  # Top 15
    
    return {
        "train": get_metrics(model, X_train, y_train, "train"),
        "validation": get_metrics(model, X_val, y_val, "validation"),
        "test": get_metrics(model, X_test, y_test, "test"),
        "feature_importance": [
            {"feature": f.replace("num__", "").replace("cat__", ""), "importance": round(i, 4)}
            for f, i in feature_importance
        ]
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No CSV file path provided"}))
        sys.exit(1)
    
    csv_path = sys.argv[1]
    use_grid_search = len(sys.argv) > 2 and sys.argv[2] == "--grid-search"
    
    try:
        with open(csv_path, 'r') as f:
            csv_content = f.read()
        
        # Load and prepare data
        data, num_features, cat_features = load_and_prepare_data(csv_content)
        
        # Get EDA stats
        eda_stats = get_eda_stats(data, num_features)
        
        # Train models
        training_results = train_models(data, num_features, cat_features, use_grid_search)
        
        # Combine results
        result = {
            "success": True,
            "eda": eda_stats,
            "training": training_results
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e), "success": False}))
        sys.exit(1)


if __name__ == "__main__":
    main()
