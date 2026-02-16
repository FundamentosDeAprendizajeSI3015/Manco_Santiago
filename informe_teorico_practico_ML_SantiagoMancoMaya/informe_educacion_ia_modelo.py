# ==========================================================
# PROYECTO: Impacto del uso de IA en la preparación laboral
# ==========================================================

import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Visualizaciones
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.io import write_html
import umap.umap_ as umap

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

DATA_CSV = "dataset_ingenieria_sistemas_ia_300_realista.csv"
OUTDIR = Path("./data_output_educacion_ia_2")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET = "preparacion_laboral"

CAT_COLS = [
    "frecuencia_uso_ia",
    "dependencia_ia",
    "aprendizaje_activo"
]

BIN_COLS = [
    "uso_para_codigo",
    "uso_para_teoria",
    "proyectos_personales"
]

NUM_COLS = [
    "promedio_acumulado",
    "nota_algoritmos",
    "nota_bases_datos",
    "horas_estudio_semana"
]

# ==========================================================
# 1️⃣ DEFINICIÓN DEL PROBLEMA
# ==========================================================

problem_definition = {
    "objetivo": "Predecir si un estudiante está laboralmente preparado (0/1)",
    "impacto": "Evaluar cómo el uso de IA influye en la preparación laboral",
    "tipo_problema": "Clasificación binaria supervisada",
    "variables_numericas": NUM_COLS,
    "variables_categoricas": CAT_COLS,
    "variables_binarias": BIN_COLS
}

with open(OUTDIR / "definicion_problema.json", "w", encoding="utf-8") as f:
    json.dump(problem_definition, f, indent=2, ensure_ascii=False)

# ==========================================================
# 2️⃣ RECOLECCIÓN DE DATOS
# ==========================================================

print("\n=== Cargando dataset ===")
df = pd.read_csv(DATA_CSV)
print("Shape:", df.shape)
print(df.head(3))
print(df.info())
print("\nNulos:\n", df.isna().sum())
print("\nDistribución Target:\n", df[TARGET].value_counts(normalize=True))

# ==========================================================
# 3️⃣ ANÁLISIS EXPLORATORIO COMPLETO
# ==========================================================

print("\n=== EDA COMPLETO ===")

# ---------------------------
# Tendencia central
# ---------------------------
media_num = df[NUM_COLS].mean()
mediana_num = df[NUM_COLS].median()
moda_num = df[NUM_COLS].mode().iloc[0]

tendencia_numericas = pd.DataFrame({
    "Media": media_num,
    "Mediana": mediana_num,
    "Moda": moda_num
})
tendencia_numericas.to_csv(OUTDIR / "tendencia_central_numericas.csv")

media_bin = df[BIN_COLS].mean()
moda_bin = df[BIN_COLS].mode().iloc[0]

tendencia_binarias = pd.DataFrame({
    "Media": media_bin,
    "Moda": moda_bin
})
tendencia_binarias.to_csv(OUTDIR / "tendencia_central_binarias.csv")

moda_cat = {col: df[col].mode().tolist() for col in CAT_COLS}

with open(OUTDIR / "moda_categoricas.json", "w", encoding="utf-8") as f:
    json.dump(moda_cat, f, indent=2, ensure_ascii=False)

# ---------------------------
# Cuartiles e IQR
# ---------------------------
iqr_results = {}

for col in NUM_COLS:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    iqr_results[col] = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR
    }

with open(OUTDIR / "iqr_results.json", "w") as f:
    json.dump(iqr_results, f, indent=2)

# ---------------------------
# Percentiles
# ---------------------------
percentiles = {}

for col in NUM_COLS:
    percentiles[col] = {
        "P10": float(np.percentile(df[col], 10)),
        "P50": float(np.percentile(df[col], 50)),
        "P90": float(np.percentile(df[col], 90))
    }

with open(OUTDIR / "percentiles.json", "w") as f:
    json.dump(percentiles, f, indent=2)

# ---------------------------
# Correlación
# ---------------------------
corr_matrix = df[NUM_COLS + [TARGET]].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor - Correlaciones")
plt.tight_layout()
plt.savefig(OUTDIR / "heatmap_correlacion.png")
plt.close()

# Pearson vs Spearman
correlation_stats = {}

for col in NUM_COLS:
    correlation_stats[col] = {
        "pearson": df[col].corr(df[TARGET], method="pearson"),
        "spearman": df[col].corr(df[TARGET], method="spearman")
    }

with open(OUTDIR / "correlation_stats.json", "w") as f:
    json.dump(correlation_stats, f, indent=2)

# ---------------------------
# Pivot table
# ---------------------------
pivot = df.pivot_table(
    index="frecuencia_uso_ia",
    columns=TARGET,
    values="promedio_acumulado",
    aggfunc="mean"
)
pivot.to_csv(OUTDIR / "pivot_promedio_por_frecuencia.csv")

# ---------------------------
# Visualizaciones interactivas
# ---------------------------

fig_scatter = px.scatter_matrix(
    df,
    dimensions=NUM_COLS,
    color=TARGET,
    height=900
)
write_html(fig_scatter, OUTDIR / "interactive_scatter_matrix.html")

fig_3d = px.scatter_3d(
    df,
    x="nota_algoritmos",
    y="horas_estudio_semana",
    z="promedio_acumulado",
    color=TARGET,
    size="nota_bases_datos"
)
write_html(fig_3d, OUTDIR / "interactive_scatter_3d.html")

# UMAP 2D y 3D
scaler_umap = StandardScaler()
X_umap_scaled = scaler_umap.fit_transform(df[NUM_COLS])

umap_2d = umap.UMAP(n_components=2, random_state=42)
emb2d = umap_2d.fit_transform(X_umap_scaled)
df_umap2 = pd.DataFrame(emb2d, columns=["UMAP_1", "UMAP_2"])
df_umap2[TARGET] = df[TARGET]
write_html(px.scatter(df_umap2, x="UMAP_1", y="UMAP_2", color=TARGET),
           OUTDIR / "interactive_umap_2d.html")

umap_3d = umap.UMAP(n_components=3, random_state=42)
emb3d = umap_3d.fit_transform(X_umap_scaled)
df_umap3 = pd.DataFrame(emb3d, columns=["UMAP_1", "UMAP_2", "UMAP_3"])
df_umap3[TARGET] = df[TARGET]
write_html(px.scatter_3d(df_umap3, x="UMAP_1", y="UMAP_2", z="UMAP_3", color=TARGET),
           OUTDIR / "interactive_umap_3d.html")

# ==========================================================
# 4️⃣ PROCESAMIENTO
# ==========================================================

for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

for c in CAT_COLS:
    df[c] = df[c].fillna("__MISSING__")

X = df.drop(columns=[TARGET])
y = df[TARGET]

X = pd.get_dummies(X, columns=CAT_COLS, drop_first=True)

# ==========================================================
# 5️⃣ SPLIT 70 / 15 / 15
# ==========================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# Balancear TRAIN
train_df = pd.concat([X_train, y_train], axis=1)
class_0 = train_df[train_df[TARGET] == 0]
class_1 = train_df[train_df[TARGET] == 1]

min_class = min(len(class_0), len(class_1))

class_0_bal = resample(class_0, replace=False, n_samples=min_class, random_state=42)
class_1_bal = resample(class_1, replace=False, n_samples=min_class, random_state=42)

train_balanced = pd.concat([class_0_bal, class_1_bal])

X_train = train_balanced.drop(columns=[TARGET])
y_train = train_balanced[TARGET]

# Escalado
scaler = StandardScaler()
X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
X_val[NUM_COLS]   = scaler.transform(X_val[NUM_COLS])
X_test[NUM_COLS]  = scaler.transform(X_test[NUM_COLS])

# ==========================================================
# 6️⃣ EXPORTACIÓN
# ==========================================================

X_train.to_parquet(OUTDIR / "X_train.parquet", index=False)
X_val.to_parquet(OUTDIR / "X_val.parquet", index=False)
X_test.to_parquet(OUTDIR / "X_test.parquet", index=False)

y_train.to_frame(name=TARGET).to_parquet(OUTDIR / "y_train.parquet", index=False)
y_val.to_frame(name=TARGET).to_parquet(OUTDIR / "y_val.parquet", index=False)
y_test.to_frame(name=TARGET).to_parquet(OUTDIR / "y_test.parquet", index=False)

schema = {
    "split": "70 / 15 / 15",
    "train_balance": y_train.value_counts().to_dict()
}

with open(OUTDIR / "processed_schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print("\n✔ PIPELINE COMPLETO EJECUTADO CORRECTAMENTE.")

# ==========================================================
# 7️⃣ ENTRENAMIENTO — REGRESIÓN LOGÍSTICA
# ==========================================================

print("\n=== ENTRENAMIENTO: REGRESIÓN LOGÍSTICA ===")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# ----------------------------------------------------------
# 7.1 Entrenamiento del modelo
# ----------------------------------------------------------

model = LogisticRegression(
    solver="liblinear",
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------------------------------
# 7.2 Predicciones
# ----------------------------------------------------------

y_train_pred = model.predict(X_train)
y_val_pred   = model.predict(X_val)
y_test_pred  = model.predict(X_test)

y_train_proba = model.predict_proba(X_train)[:, 1]
y_val_proba   = model.predict_proba(X_val)[:, 1]
y_test_proba  = model.predict_proba(X_test)[:, 1]

# ----------------------------------------------------------
# 7.3 Métricas
# ----------------------------------------------------------

def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }

metrics_train = compute_metrics(y_train, y_train_pred, y_train_proba)
metrics_val   = compute_metrics(y_val, y_val_pred, y_val_proba)
metrics_test  = compute_metrics(y_test, y_test_pred, y_test_proba)

all_metrics = {
    "train": metrics_train,
    "validation": metrics_val,
    "test": metrics_test
}

with open(OUTDIR / "logistic_regression_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\n=== MÉTRICAS ===")

print("\nTRAIN:")
for k, v in metrics_train.items():
    print(f"{k}: {v:.4f}")

print("\nVALIDATION:")
for k, v in metrics_val.items():
    print(f"{k}: {v:.4f}")

print("\nTEST:")
for k, v in metrics_test.items():
    print(f"{k}: {v:.4f}")

# ----------------------------------------------------------
# 7.4 Matriz de Confusión
# ----------------------------------------------------------

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - Test")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(OUTDIR / "confusion_matrix_test.png")
plt.close()

# ----------------------------------------------------------
# 7.5 Curva ROC
# ----------------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Test")
plt.tight_layout()
plt.savefig(OUTDIR / "roc_curve_test.png")
plt.close()

# ----------------------------------------------------------
# 7.6 Curva Precision-Recall
# ----------------------------------------------------------

precision, recall, _ = precision_recall_curve(y_test, y_test_proba)

plt.figure(figsize=(6,5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall - Test")
plt.tight_layout()
plt.savefig(OUTDIR / "precision_recall_curve_test.png")
plt.close()

# ----------------------------------------------------------
# 7.7 Coeficientes del Modelo
# ----------------------------------------------------------

coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": model.coef_[0]
}).sort_values(by="coefficient", ascending=False)

coef_df.to_csv(OUTDIR / "logistic_regression_coefficients.csv", index=False)

# Gráfica de importancia (coeficientes)
plt.figure(figsize=(8,6))
sns.barplot(
    x="coefficient",
    y="feature",
    data=coef_df
)
plt.title("Coeficientes - Regresión Logística")
plt.tight_layout()
plt.savefig(OUTDIR / "logistic_coefficients.png")
plt.close()

print("\n✔ ENTRENAMIENTO COMPLETADO Y RESULTADOS GUARDADOS.")
