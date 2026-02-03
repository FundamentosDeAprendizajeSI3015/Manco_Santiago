# iris_lifecycle.py
# -*- coding: utf-8 -*-

"""
Ciclo de vida de Machine Learning aplicado al dataset Iris.

Fases:
1. Definición del problema
2. Recolección de datos
3. Procesamiento (limpieza, normalización, codificación)
4. Entrenamiento
5. Evaluación
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ======================================================
# 1. DEFINICIÓN DEL PROBLEMA
# ======================================================
"""
Problema:
Clasificación supervisada multiclase.

Objetivo:
Dado un conjunto de características morfológicas de una flor,
predecir la especie de Iris (setosa, versicolor, virginica).
"""

problem_type = "Clasificación multiclase"
target_variable = "species"

print("Tipo de problema:", problem_type)
print("Variable objetivo:", target_variable)

# ======================================================
# 2. RECOLECCIÓN DE DATOS
# ======================================================
"""
Fuente:
Dataset Iris (scikit-learn)
"""

iris = datasets.load_iris()

X = pd.DataFrame(
    iris.data,
    columns=[c.replace(" (cm)", "").replace(" ", "_") for c in iris.feature_names]
)

y = pd.Series(iris.target, name="species")

class_names = iris.target_names

print("\nDatos recolectados:")
print("Características:", X.shape)
print("Clases:", np.unique(y))

# ======================================================
# 3. PROCESAMIENTO DE DATOS
# ======================================================
"""
Incluye:
- Limpieza
- Normalización
- Codificación de variables
"""

# 3.1 Limpieza
# El dataset Iris no tiene valores nulos, pero se valida
print("\nValores nulos por columna:")
print(X.isnull().sum())

# 3.2 Normalización
# Se hará dentro del pipeline con StandardScaler

# 3.3 Codificación
# La variable objetivo ya está codificada numéricamente (0,1,2)

# 3.4 Partición de datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nDatos particionados:")
print("Train:", X_train.shape)
print("Test:", X_test.shape)

# ======================================================
# 4. ENTRENAMIENTO DEL MODELO
# ======================================================
"""
Modelo seleccionado:
Máquina de Soporte Vectorial (SVM) con kernel RBF
"""

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
])

model.fit(X_train, y_train)

print("\nModelo entrenado correctamente.")

# ======================================================
# 5. EVALUACIÓN DEL MODELO
# ======================================================
"""
Métricas utilizadas:
- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión
"""

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted"
)

cm = confusion_matrix(y_test, y_pred)

print("\n=== Evaluación del modelo ===")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-score : {f1:.3f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Matriz de confusión:")
print(cm)
