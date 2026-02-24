# ==========================================================
# PROYECTO: REGRESIÓN LINEAL Y LOGÍSTICA - MOVIES DATASET
# ==========================================================

import os
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

# Crear carpeta output
os.makedirs("output", exist_ok=True)

# ==========================================================
# 1️⃣ CARGA Y LIMPIEZA DEL DATASET
# ==========================================================

df = pd.read_csv("movies.csv")

# Eliminamos columnas irrelevantes
df = df.drop(columns=["Title", "Director", "Actors", "Genre"], errors='ignore')

# Eliminamos filas con valores nulos
df = df.dropna()

# ==========================================================
# 2️⃣ REGRESIÓN LINEAL
# Objetivo: predecir Revenue (Millions)
# ==========================================================

print("\n================ REGRESIÓN LINEAL ================\n")

# Variables predictoras
features = [
    "Runtime (Minutes)",
    "Rating",
    "Votes",
    "Metascore",
    "Year"
]

X = df[features]
y = df["Revenue (Millions)"]

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# ----------- Gráfica Exploratoria -----------

plt.figure()
plt.scatter(X_train["Rating"], y_train, c='c', label="Train")
plt.scatter(X_test["Rating"], y_test, c='m', label="Test")
plt.xlabel("Rating")
plt.ylabel("Revenue (Millions)")
plt.legend()
plt.title("Training vs Test Data (Revenue vs Rating)")
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

ridge = RandomizedSearchCV(
    ridge_base,
    cv=4,
    param_distributions=param_distributions,
    n_iter=50,
    random_state=random_state
)

lasso = RandomizedSearchCV(
    lasso_base,
    cv=4,
    param_distributions=param_distributions,
    n_iter=50,
    random_state=random_state
)

# Entrenamiento
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# Mejores parámetros
print("Mejores parámetros Ridge:", ridge.best_params_)
print("Mejores parámetros Lasso:", lasso.best_params_)

# Métricas
print("\nModelo Ridge")
print("R2:", ridge.score(X_test, y_test))
print("MAE:", mean_absolute_error(y_test, ridge.predict(X_test)))

print("\nModelo Lasso")
print("R2:", lasso.score(X_test, y_test))
print("MAE:", mean_absolute_error(y_test, lasso.predict(X_test)))

# ----------- Gráfica Predicción -----------

y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

plt.figure()
plt.scatter(y_test, y_pred_ridge)
plt.xlabel("Real Revenue")
plt.ylabel("Predicted Revenue (Ridge)")
plt.title("Ridge Prediction")
plt.savefig("output/ridge_prediction.png")
plt.close()

plt.figure()
plt.scatter(y_test, y_pred_lasso)
plt.xlabel("Real Revenue")
plt.ylabel("Predicted Revenue (Lasso)")
plt.title("Lasso Prediction")
plt.savefig("output/lasso_prediction.png")
plt.close()

# ==========================================================
# 3️⃣ REGRESIÓN LOGÍSTICA
# Objetivo: clasificar si una película tiene alto ingreso
# ==========================================================

print("\n================ REGRESIÓN LOGÍSTICA ================\n")

# Creamos variable binaria
median_revenue = df["Revenue (Millions)"].median()
df["High_Revenue"] = (df["Revenue (Millions)"] > median_revenue).astype(int)

X_class = df[features]
y_class = df["High_Revenue"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=random_state
)

# ----------- Pipeline -----------

lr_base = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=10000))
])

param_distributions_lr = {
    'poly__degree': list(range(1, 4)),
    'classifier__C': reciprocal(1e-4, 1e2)
}

lr = RandomizedSearchCV(
    lr_base,
    cv=4,
    param_distributions=param_distributions_lr,
    n_iter=50,
    random_state=random_state
)

# Entrenamiento
lr.fit(X_train_c, y_train_c)

print("Mejores parámetros:", lr.best_params_)

# Métricas
print("Accuracy:", lr.score(X_test_c, y_test_c))
print("F1-score:", f1_score(y_test_c, lr.predict(X_test_c)))

# ----------- Matriz de Confusión -----------

cm = confusion_matrix(y_test_c, lr.predict(X_test_c))
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("output/confusion_matrix.png")
plt.close()

# ----------- Gráfica de Clasificación (2 features) -----------

# Solo para visualización usamos Rating y Metascore
X_vis = df[["Rating", "Metascore"]]
y_vis = df["High_Revenue"]

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_vis, y_vis, test_size=0.2, random_state=random_state
)

lr_vis = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

lr_vis.fit(X_train_v, y_train_v)

# Frontera de decisión
x_min, x_max = X_vis.iloc[:, 0].min() - 1, X_vis.iloc[:, 0].max() + 1
y_min, y_max = X_vis.iloc[:, 1].min() - 1, X_vis.iloc[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.1),
    np.arange(y_min, y_max, 0.1)
)

Z = lr_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=y_vis)
plt.xlabel("Rating")
plt.ylabel("Metascore")
plt.title("Logistic Regression Decision Boundary")
plt.savefig("output/logistic_boundary.png")
plt.close()

print("\n✔️ TODAS LAS IMÁGENES FUERON EXPORTADAS EN LA CARPETA 'output/'")