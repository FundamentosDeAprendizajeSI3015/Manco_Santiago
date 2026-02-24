# ==========================================================
# PROYECTO: REGRESIÓN LINEAL Y LOGÍSTICA - MOVIES DATASET
# ==========================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import reciprocal
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

random_state = 42
np.random.seed(random_state)
plt.rc('font', family='serif', size=12)

os.makedirs("output", exist_ok=True)

# ==========================================================
# 1️⃣ CARGA Y LIMPIEZA
# ==========================================================

df = pd.read_csv("movies.csv")

print("Columnas detectadas:", df.columns)

# ---------- LIMPIEZA ----------

# YEAR -> extraer solo el primer año numérico
df["YEAR"] = df["YEAR"].astype(str).str.extract(r'(\d{4})')
df["YEAR"] = pd.to_numeric(df["YEAR"], errors='coerce')

# RUNTIME -> extraer número
df["RunTime"] = df["RunTime"].astype(str).str.extract(r'(\d+)')
df["RunTime"] = pd.to_numeric(df["RunTime"], errors='coerce')

# VOTES -> quitar comas
df["VOTES"] = df["VOTES"].astype(str).str.replace(",", "")
df["VOTES"] = pd.to_numeric(df["VOTES"], errors='coerce')

# GROSS -> quitar $ y comas
df["Gross"] = df["Gross"].astype(str).str.replace(r"[^\d.]", "", regex=True)
df["Gross"] = pd.to_numeric(df["Gross"], errors='coerce')

# RATING -> numérico
df["RATING"] = pd.to_numeric(df["RATING"], errors='coerce')

# Eliminar nulos
df = df.dropna()

# ==========================================================
# 2️⃣ REGRESIÓN LINEAL
# Objetivo: predecir Gross
# ==========================================================

print("\n================ REGRESIÓN LINEAL ================\n")

features = ["RunTime", "RATING", "VOTES", "YEAR"]

X = df[features]
y = df["Gross"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# ----------- Gráfica Train/Test -----------

plt.figure()
plt.scatter(X_train["RATING"], y_train, c='c', label="Train")
plt.scatter(X_test["RATING"], y_test, c='m', label="Test")
plt.xlabel("Rating")
plt.ylabel("Gross")
plt.legend()
plt.title("Train vs Test (Gross vs Rating)")
plt.savefig("output/reg_lineal_train_test.png")
plt.close()

# ----------- Pipelines -----------

ridge_base = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Ridge())
])

lasso_base = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Lasso(max_iter=10000))
])

param_distributions = {
    'poly__degree': list(range(1, 4)),
    'regressor__alpha': reciprocal(1e-4, 1e2)
}

ridge = RandomizedSearchCV(ridge_base, param_distributions=param_distributions,
                           cv=4, n_iter=40, random_state=random_state)

lasso = RandomizedSearchCV(lasso_base, param_distributions=param_distributions,
                           cv=4, n_iter=40, random_state=random_state)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

print("Mejores parámetros Ridge:", ridge.best_params_)
print("Mejores parámetros Lasso:", lasso.best_params_)

print("\nModelo Ridge")
print("R2:", ridge.score(X_test, y_test))
print("MAE:", mean_absolute_error(y_test, ridge.predict(X_test)))

print("\nModelo Lasso")
print("R2:", lasso.score(X_test, y_test))
print("MAE:", mean_absolute_error(y_test, lasso.predict(X_test)))

# ----------- Gráficas predicción -----------

plt.figure()
plt.scatter(y_test, ridge.predict(X_test))
plt.xlabel("Real Gross")
plt.ylabel("Predicted Gross (Ridge)")
plt.title("Ridge Prediction")
plt.savefig("output/ridge_prediction.png")
plt.close()

plt.figure()
plt.scatter(y_test, lasso.predict(X_test))
plt.xlabel("Real Gross")
plt.ylabel("Predicted Gross (Lasso)")
plt.title("Lasso Prediction")
plt.savefig("output/lasso_prediction.png")
plt.close()

# ==========================================================
# 3️⃣ REGRESIÓN LOGÍSTICA
# Clasificar si Gross es alto
# ==========================================================

print("\n================ REGRESIÓN LOGÍSTICA ================\n")

median_gross = df["Gross"].median()
df["High_Gross"] = (df["Gross"] > median_gross).astype(int)

X_class = df[features]
y_class = df["High_Gross"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=random_state
)

lr_base = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=10000))
])

param_distributions_lr = {
    'poly__degree': list(range(1, 4)),
    'classifier__C': reciprocal(1e-4, 1e2)
}

lr = RandomizedSearchCV(lr_base, param_distributions=param_distributions_lr,
                        cv=4, n_iter=40, random_state=random_state)

lr.fit(X_train_c, y_train_c)

print("Mejores parámetros:", lr.best_params_)
print("Accuracy:", lr.score(X_test_c, y_test_c))
print("F1-score:", f1_score(y_test_c, lr.predict(X_test_c)))

# ----------- Matriz de Confusión -----------

cm = confusion_matrix(y_test_c, lr.predict(X_test_c))
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("output/confusion_matrix.png")
plt.close()

print("\n✔️ TODO EJECUTADO CORRECTAMENTE.")
print("📂 Las imágenes están en la carpeta: output/")