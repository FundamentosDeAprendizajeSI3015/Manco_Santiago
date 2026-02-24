# ==========================================================
# PROYECTO: CLASIFICACIÓN - RANDOM FOREST vs GRADIENT BOOSTING
# Dataset: dataset_ingenieria_sistemas_ia_300_realista.csv
# Target: preparacion_laboral (0 = No preparado, 1 = Preparado)
# División: 60% Train / 20% Val / 20% Test (estratificado)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# ==========================================================
# 1️⃣ CONFIGURACIÓN
# ==========================================================

random_state = 42
plt.rc('font', family='serif', size=12)

# ==========================================================
# 2️⃣ CARGA DE DATOS
# ==========================================================

print("\nCargando dataset...")
data = pd.read_csv("dataset_ingenieria_sistemas_ia_300_realista.csv")

print("\nPrimeras filas:")
print(data.head())

print("\nInformación general:")
print(data.info())

print("\nDistribución del target:")
print(data["preparacion_laboral"].value_counts())

# ==========================================================
# 3️⃣ EXPLORACIÓN GRÁFICA
# ==========================================================

print("\nGenerando histogramas...")
data.hist(figsize=(15,10))
plt.tight_layout()
plt.show()

# Matriz de correlación (solo numéricas)
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()

# ==========================================================
# 4️⃣ LIMPIEZA
# ==========================================================

print("\nEliminando duplicados...")
data.drop_duplicates(inplace=True)

print("\nValores nulos:")
print(data.isnull().sum())

# Rellenar nulos numéricos con mediana
data.fillna(data.median(numeric_only=True), inplace=True)

# ==========================================================
# 5️⃣ FEATURES Y TARGET
# ==========================================================

X = data.drop(columns="preparacion_laboral")
y = data["preparacion_laboral"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(include=np.number).columns

print("\nColumnas numéricas:", list(num_cols))
print("Columnas categóricas:", list(cat_cols))

# ==========================================================
# 6️⃣ DIVISIÓN 60 / 20 / 20
# ==========================================================

print("\nDividiendo dataset 60/20/20...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    stratify=y,
    random_state=random_state
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=random_state
)

print(f"\nTrain: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")

# ==========================================================
# 7️⃣ PIPELINE
# ==========================================================

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# ==========================================================
# 8️⃣ MODELOS
# ==========================================================

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=random_state
    ))
])

gb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=200,
        random_state=random_state
    ))
])

# ==========================================================
# 9️⃣ ENTRENAMIENTO
# ==========================================================

print("\nEntrenando Random Forest...")
rf_model.fit(X_train, y_train)

print("Entrenando Gradient Boosting...")
gb_model.fit(X_train, y_train)

# ==========================================================
# 🔟 FUNCIÓN DE EVALUACIÓN
# ==========================================================

def evaluar_modelo(modelo, X, y, nombre):

    y_pred = modelo.predict(X)
    y_prob = modelo.predict_proba(X)[:,1]

    print(f"\n===== {nombre} =====")
    print("Accuracy :", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall   :", recall_score(y, y_pred))
    print("F1 Score :", f1_score(y, y_pred))
    print("AUC      :", roc_auc_score(y, y_prob))

    print("\nMatriz de Confusión:")
    print(confusion_matrix(y, y_pred))

    print("\nReporte completo:")
    print(classification_report(y, y_pred))

    # Matriz gráfica
    plt.figure()
    sns.heatmap(confusion_matrix(y, y_pred),
                annot=True, fmt='d')
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.ylabel("Real")
    plt.xlabel("Predicción")
    plt.show()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1])
    plt.title(f"ROC Curve - {nombre}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

# ==========================================================
# 1️⃣1️⃣ EVALUACIÓN TRAIN
# ==========================================================

evaluar_modelo(rf_model, X_train, y_train, "Random Forest (Train)")
evaluar_modelo(gb_model, X_train, y_train, "Gradient Boosting (Train)")

# ==========================================================
# 1️⃣2️⃣ EVALUACIÓN VALIDATION
# ==========================================================

evaluar_modelo(rf_model, X_val, y_val, "Random Forest (Validation)")
evaluar_modelo(gb_model, X_val, y_val, "Gradient Boosting (Validation)")

# ==========================================================
# 1️⃣3️⃣ EVALUACIÓN TEST
# ==========================================================

evaluar_modelo(rf_model, X_test, y_test, "Random Forest (Test)")
evaluar_modelo(gb_model, X_test, y_test, "Gradient Boosting (Test)")
