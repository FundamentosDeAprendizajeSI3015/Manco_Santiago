"""
Machine Learning Service for FIRE_UdeA
Handles training, prediction, and evaluation of ML models
"""
import os
import json
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from django.conf import settings


# Feature columns configuration
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

RANDOM_STATE = 42


class MLService:
    """Service class for ML operations"""
    
    def __init__(self):
        self.models_dir = settings.ML_MODELS_DIR
        
    def load_and_preprocess_data(self, file_path: str) -> tuple:
        """Load and preprocess dataset"""
        # Load data
        data = pd.read_csv(file_path)
        
        # Remove duplicates
        data.drop_duplicates(inplace=True)
        
        # Get numeric and categorical columns
        num_features = [col for col in NUM_COLS if col in data.columns]
        cat_features = [col for col in CAT_COLS if col in data.columns]
        
        # Fill missing values
        for col in num_features:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col] = data[col].fillna(data[col].median())
        
        for col in cat_features:
            data[col] = data[col].fillna("Unknown")
        
        # Prepare features and target
        X = data.drop(columns="label")
        y = data["label"]
        
        return data, X, y, num_features, cat_features
    
    def split_data(self, X, y, test_size=0.4, val_size=0.5):
        """Split data into train, validation, and test sets (60/20/20)"""
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=RANDOM_STATE
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_preprocessor(self, num_features: list, cat_features: list):
        """Create preprocessing pipeline"""
        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        
        if cat_features:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, num_features),
                    ("cat", categorical_transformer, cat_features)
                ]
            )
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, num_features)
                ]
            )
        
        return preprocessor
    
    def train_random_forest(self, X_train, y_train, preprocessor, use_gridsearch=True):
        """Train Random Forest model"""
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))
        ])
        
        if use_gridsearch:
            param_grid = {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__class_weight": [None, "balanced"]
            }
            
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_, grid_search.best_params_
        else:
            pipeline.fit(X_train, y_train)
            return pipeline, {}
    
    def train_gradient_boosting(self, X_train, y_train, preprocessor, use_gridsearch=True):
        """Train Gradient Boosting model"""
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ])
        
        if use_gridsearch:
            param_grid = {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
                "classifier__min_samples_split": [2, 5],
                "classifier__subsample": [0.8, 1.0]
            }
            
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_, grid_search.best_params_
        else:
            pipeline.fit(X_train, y_train)
            return pipeline, {}
    
    def evaluate_model(self, model, X, y, set_name="test"):
        """Evaluate model and return metrics"""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred)),
            "recall": float(recall_score(y, y_pred)),
            "f1_score": float(f1_score(y, y_pred)),
            "auc_roc": float(roc_auc_score(y, y_prob)),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, output_dict=True)
        }
        
        return metrics
    
    def get_feature_importance(self, model, feature_names=None):
        """Get feature importance from model"""
        classifier = model.named_steps["classifier"]
        
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            
            if feature_names is None:
                feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            
            importance_dict = {
                str(name): float(imp) 
                for name, imp in zip(feature_names, importances)
            }
            
            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            return importance_dict
        
        return {}
    
    def save_model(self, model, model_type: str, dataset_id: str):
        """Save trained model to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_{dataset_id}_{timestamp}.joblib"
        filepath = self.models_dir / filename
        
        joblib.dump(model, filepath)
        return str(filepath)
    
    def load_model(self, model_path: str):
        """Load model from disk"""
        return joblib.load(model_path)
    
    def predict(self, model, input_data: dict):
        """Make prediction with model"""
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure all required columns exist
        for col in NUM_COLS:
            if col not in df.columns:
                df[col] = 0.0
        
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1]
        
        return int(prediction), float(probability)
    
    def compute_eda(self, file_path: str):
        """Compute Exploratory Data Analysis"""
        data = pd.read_csv(file_path)
        
        # Basic statistics
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        stats = data[num_cols].describe().to_dict()
        
        # Correlation matrix
        corr_matrix = data[num_cols].corr().to_dict()
        
        # Target distribution
        if "label" in data.columns:
            target_dist = data["label"].value_counts(normalize=True).to_dict()
        else:
            target_dist = {}
        
        # Missing values
        missing = data.isnull().sum().to_dict()
        
        return {
            "statistics": stats,
            "correlation_matrix": corr_matrix,
            "target_distribution": target_dist,
            "missing_values": missing,
            "rows": len(data),
            "columns": len(data.columns),
            "column_names": data.columns.tolist()
        }


# Singleton instance
ml_service = MLService()
