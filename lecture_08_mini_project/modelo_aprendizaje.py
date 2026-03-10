# ==========================================================
# PROYECTO: CLASIFICACIÓN - RANDOM FOREST vs GRADIENT BOOSTING
# Dataset: Finanzas de la Universidad de Antioquia (FIRE_UdeA)
# Target: label (0 = Situación financiera estable, 1 = Situación financiera crítica)
# División: 60% Train / 20% Val / 20% Test (estratificado)
# ==========================================================
# Análisis de indicadores financieros por unidad académica y año

import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Visualizaciones interactivas ---
import plotly.express as px
from plotly.io import write_html

import umap.umap_ as umap

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import plot_tree
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

# Directorio de salida para EDA
OUTDIR = Path("./data_output_FIRE_UdeA")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Definición de columnas por tipo
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

CAT_COLS = [
    "unidad",
    "anio"
]

# ==========================================================
# 2️⃣ CARGA DE DATOS
# ==========================================================

print("\nCargando dataset FIRE_UdeA...")
data = pd.read_csv("dataset_sintetico_FIRE_UdeA_realista.csv")

print("\nPrimeras filas:")
print(data.head())

print("\nInformación general:")
print(data.info())

print("\nDistribución del target:")
print(data["label"].value_counts())
print("\nProporción:")
print(data["label"].value_counts(normalize=True))

# ==========================================================
# 2️⃣A EDA BÁSICO - MEDIDAS DE TENDENCIA CENTRAL
# ==========================================================

print("\n=== EDA - Medidas de Tendencia Central ===")

# --- Variables Numéricas ---
print("\n--- Variables Numéricas ---")
media_num = data[NUM_COLS].mean()
mediana_num = data[NUM_COLS].median()
moda_num = data[NUM_COLS].mode().iloc[0] if len(data[NUM_COLS].mode()) > 0 else np.nan

tendencia_numericas = pd.DataFrame({
    "Media": media_num,
    "Mediana": mediana_num,
    "Moda": moda_num
})

print(tendencia_numericas)

# Guardar resultados
tendencia_numericas.to_csv(OUTDIR / "tendencia_central_numericas.csv")

# --- Variables Categóricas ---
print("\n--- Variables Categóricas ---")
moda_cat = {}
for col in CAT_COLS:
    moda_cat[col] = data[col].mode().tolist()

for col, moda in moda_cat.items():
    print(f"Moda de {col}: {moda}")

with open(OUTDIR / "moda_categoricas.json", "w", encoding="utf-8") as f:
    json.dump(moda_cat, f, indent=2, ensure_ascii=False)

print("[OK] Medidas de tendencia central calculadas y guardadas.")

# ==========================================================
# 2️⃣B VISUALIZACIONES INTERACTIVAS
# ==========================================================

print("\n=== Generando visualizaciones interactivas ===")

# a) Scatter Matrix (solo con variables numéricas)
print("Generando Scatter Matrix...")
fig_scatter = px.scatter_matrix(
    data[NUM_COLS + ["label"]],
    dimensions=NUM_COLS[:5],  # Primeras 5 para legibilidad
    color="label",
    title="FIRE_UdeA — Scatter Matrix (Variables Financieras)",
    height=900
)
write_html(fig_scatter, OUTDIR / "interactive_scatter_matrix.html")

# b) Coordenadas paralelas (normalizadas)
print("Generando Coordenadas Paralelas...")
df_parallel = data[NUM_COLS[:6]].copy()
df_parallel = (df_parallel - df_parallel.min()) / (df_parallel.max() - df_parallel.min())
df_parallel["label"] = data["label"]
df_parallel["unidad"] = data["unidad"]

fig_parallel = px.parallel_coordinates(
    df_parallel,
    dimensions=NUM_COLS[:6],
    color="label",
    color_continuous_scale=px.colors.diverging.Tealrose,
    title="FIRE_UdeA — Coordenadas Paralelas"
)
write_html(fig_parallel, OUTDIR / "interactive_parallel_coordinates.html")

# c) Scatter 3D
print("Generando Scatter 3D...")
fig_3d = px.scatter_3d(
    data,
    x="ingresos_totales",
    y="gastos_personal",
    z="liquidez",
    color="label",
    size="gp_ratio",
    hover_data=["unidad", "anio"],
    title="FIRE_UdeA — Dispersión 3D",
    height=750
)
write_html(fig_3d, OUTDIR / "interactive_scatter_3d.html")

# d) UMAP 2D
print("Generando UMAP 2D...")
X_umap = data[NUM_COLS].copy()

# Rellenar nulos con mediana antes de UMAP
for col in NUM_COLS:
    X_umap[col] = pd.to_numeric(X_umap[col], errors="coerce")
    X_umap[col] = X_umap[col].fillna(X_umap[col].median())

scaler_umap = StandardScaler()
X_umap_scaled = scaler_umap.fit_transform(X_umap)

umap_2d = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=random_state
)

emb_2d = umap_2d.fit_transform(X_umap_scaled)
df_umap = pd.DataFrame(emb_2d, columns=["UMAP_1", "UMAP_2"])
df_umap["label"] = data["label"].values
df_umap["unidad"] = data["unidad"].values

fig_umap = px.scatter(
    df_umap,
    x="UMAP_1",
    y="UMAP_2",
    color="label",
    hover_data=["unidad"],
    title="FIRE_UdeA — UMAP 2D",
    height=700
)
write_html(fig_umap, OUTDIR / "interactive_umap_2d.html")

# e) UMAP 3D
print("Generando UMAP 3D...")
umap_3d = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=3,
    random_state=random_state
)

emb_3d = umap_3d.fit_transform(X_umap_scaled)
df_umap3 = pd.DataFrame(emb_3d, columns=["UMAP_1", "UMAP_2", "UMAP_3"])
df_umap3["label"] = data["label"].values
df_umap3["unidad"] = data["unidad"].values

fig_umap3 = px.scatter_3d(
    df_umap3,
    x="UMAP_1",
    y="UMAP_2",
    z="UMAP_3",
    color="label",
    hover_data=["unidad"],
    title="FIRE_UdeA — UMAP 3D",
    height=750
)
write_html(fig_umap3, OUTDIR / "interactive_umap_3d.html")

print("[OK] Visualizaciones guardadas en:", OUTDIR)

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
plt.title("Matriz de Correlación - FIRE_UdeA")
plt.show()

# ==========================================================
# 4️⃣ LIMPIEZA Y PREPARACIÓN
# ==========================================================

print("\n=== Limpieza de datos ===")
print("Eliminando duplicados...")
data.drop_duplicates(inplace=True)

print("\nValores nulos por columna:")
print(data.isnull().sum())

# Rellenar nulos numéricos con mediana
for col in NUM_COLS:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = data[col].fillna(data[col].median())

# Rellenar nulos categóricos con 'Unknown'
for col in CAT_COLS:
    data[col] = data[col].fillna("Unknown")

# ==========================================================
# 5️⃣ FEATURES Y TARGET
# ==========================================================

X = data.drop(columns="label")
y = data["label"]

print("\nShape de X:", X.shape)
print("Shape de y:", y.shape)

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
# 6️⃣ PIPELINE
# ==========================================================

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Obtener columnas categóricas dinámicamente
cat_features = X.select_dtypes(include="object").columns.tolist()
num_features = X.select_dtypes(include=np.number).columns.tolist()

print("\nColumnas para preprocesamiento:")
print(f"  Numéricas ({len(num_features)}): {num_features[:3]}...")
print(f"  Categóricas ({len(cat_features)}): {cat_features}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ] if cat_features else [("num", numeric_transformer, num_features)]
)

# ==========================================================
# 7️⃣ MODELOS
# ==========================================================

# ==========================================================
# 7️⃣ MODELOS BASE
# ==========================================================

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=random_state
    ))
])

gb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        random_state=random_state
    ))
])

# ==========================================================
# 8️⃣ BÚSQUEDA DE HIPERPARÁMETROS CON GRIDSEARCH
# ==========================================================

print("\nBuscando mejores hiperparámetros para Random Forest...")

rf_param_grid = {
    "classifier__n_estimators": [100, 200, 400],
    "classifier__max_depth": [None, 5, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__class_weight": [None, "balanced"]
}

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

print("\nMejores parámetros Random Forest:")
print(rf_grid.best_params_)
print("Mejor score CV Random Forest:", rf_grid.best_score_)

rf_model = rf_grid.best_estimator_


print("\nBuscando mejores hiperparámetros para Gradient Boosting...")

gb_param_grid = {
    "classifier__n_estimators": [100, 200, 400],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__max_depth": [2, 3, 5],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__subsample": [0.8, 1.0]
}

gb_grid = GridSearchCV(
    estimator=gb_pipeline,
    param_grid=gb_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

gb_grid.fit(X_train, y_train)

print("\nMejores parámetros Gradient Boosting:")
print(gb_grid.best_params_)
print("Mejor score CV Gradient Boosting:", gb_grid.best_score_)

gb_model = gb_grid.best_estimator_

# ==========================================================
# 🌳 GRAFICAR UN ÁRBOL DEL RANDOM FOREST
# ==========================================================

# Obtener el modelo interno ya entrenado
rf_clf = rf_model.named_steps["classifier"]

# Obtener nombres reales de features después del preprocesamiento
feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()

# Seleccionamos el primer árbol del bosque
tree = rf_clf.estimators_[0]

plt.figure(figsize=(20,10))
plot_tree(
    tree,
    feature_names=feature_names,
    class_names=["Estable", "Crítico"],
    filled=True,
    max_depth=3   # Limita profundidad para que sea legible
)
plt.title("Árbol individual del Random Forest - Análisis Financiero FIRE_UdeA")
plt.show()

# ==========================================================
# 🌳 GRAFICAR UN ÁRBOL DEL GRADIENT BOOSTING
# ==========================================================

gb_clf = gb_model.named_steps["classifier"]

# GradientBoosting guarda los árboles en estimators_[n][0]
tree_gb = gb_clf.estimators_[0][0]

plt.figure(figsize=(20,10))
plot_tree(
    tree_gb,
    feature_names=feature_names,
    class_names=["Estable", "Crítico"],
    filled=True,
    max_depth=3
)
plt.title("Árbol individual del Gradient Boosting - Análisis Financiero FIRE_UdeA")
plt.show()

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
# 9️⃣ EVALUACIÓN TRAIN
# ==========================================================

evaluar_modelo(rf_model, X_train, y_train, "Random Forest (Train)")
evaluar_modelo(gb_model, X_train, y_train, "Gradient Boosting (Train)")

# ==========================================================
# 🔟 EVALUACIÓN VALIDATION
# ==========================================================

evaluar_modelo(rf_model, X_val, y_val, "Random Forest (Validation)")
evaluar_modelo(gb_model, X_val, y_val, "Gradient Boosting (Validation)")

# ==========================================================
# 1️⃣1️⃣ EVALUACIÓN TEST
# ==========================================================

evaluar_modelo(rf_model, X_test, y_test, "Random Forest (Test)")
evaluar_modelo(gb_model, X_test, y_test, "Gradient Boosting (Test)")
