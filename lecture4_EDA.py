import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Visualizaciones ---
import plotly.express as px
from plotly.io import write_html

import umap.umap_ as umap

# ---------------------------
# Constantes
# ---------------------------
DATA_CSV = "dataset_ia_aprendizaje_ml.csv"
OUTDIR = Path("./data_output_educacion_ia")
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

# ---------------------------
# 1) Carga del dataset
# ---------------------------
print("\n=== 1) Cargando dataset ===")
df = pd.read_csv(DATA_CSV)
print("Shape:", df.shape)
print(df.head(3))

# ---------------------------
# 2) EDA básico
# ---------------------------
print("\n=== 2) EDA básico ===")
print(df.info())
print("\nNulos por columna:")
print(df.isna().sum())

print("\nDistribución de la variable objetivo:")
print(df[TARGET].value_counts(normalize=True))

# ---------------------------
# 2.2) Medidas de tendencia central
# ---------------------------
print("\n=== 2.2) Medidas de Tendencia Central ===")

# ---------------------------
# A) VARIABLES NUMÉRICAS
# ---------------------------
print("\n--- Variables Numéricas ---")

media_num = df[NUM_COLS].mean()
mediana_num = df[NUM_COLS].median()
moda_num = df[NUM_COLS].mode().iloc[0]  # primera moda si hay varias

tendencia_numericas = pd.DataFrame({
    "Media": media_num,
    "Mediana": mediana_num,
    "Moda": moda_num
})

print(tendencia_numericas)

# ---------------------------
# B) VARIABLES CATEGÓRICAS
# ---------------------------
print("\n--- Variables Categóricas ---")

moda_cat = {}
for col in CAT_COLS:
    moda_cat[col] = df[col].mode().tolist()

for col, moda in moda_cat.items():
    print(f"Moda de {col}: {moda}")

# ---------------------------
# C) VARIABLES BINARIAS
# ---------------------------
print("\n--- Variables Binarias ---")

media_bin = df[BIN_COLS].mean()
moda_bin = df[BIN_COLS].mode().iloc[0]

tendencia_binarias = pd.DataFrame({
    "Media": media_bin,
    "Moda": moda_bin
})

print(tendencia_binarias)

# ---------------------------
# D) Guardar resultados (opcional pero recomendado)
# ---------------------------
tendencia_numericas.to_csv(
    OUTDIR / "tendencia_central_numericas.csv"
)

tendencia_binarias.to_csv(
    OUTDIR / "tendencia_central_binarias.csv"
)

with open(OUTDIR / "moda_categoricas.json", "w", encoding="utf-8") as f:
    json.dump(moda_cat, f, indent=2, ensure_ascii=False)

print("✔ Medidas de tendencia central calculadas y guardadas.")

# ---------------------------
# 2.5) EDA visual interactivo
# ---------------------------
print("\n=== 2.5) EDA visual interactivo ===")
OUTDIR.mkdir(parents=True, exist_ok=True)

# a) Scatter Matrix
fig_scatter = px.scatter_matrix(
    df,
    dimensions=NUM_COLS,
    color=TARGET,
    title="Educación + IA — Scatter Matrix",
    height=900
)
write_html(fig_scatter, OUTDIR / "interactive_scatter_matrix.html")

# b) Coordenadas paralelas
df_parallel = df[NUM_COLS].copy()
df_parallel = (df_parallel - df_parallel.min()) / (df_parallel.max() - df_parallel.min())
df_parallel[TARGET] = df[TARGET]

fig_parallel = px.parallel_coordinates(
    df_parallel,
    dimensions=NUM_COLS,
    color=TARGET,
    color_continuous_scale=px.colors.diverging.Tealrose,
    title="Educación + IA — Coordenadas Paralelas"
)
write_html(fig_parallel, OUTDIR / "interactive_parallel_coordinates.html")

# c) Scatter 3D
fig_3d = px.scatter_3d(
    df,
    x="nota_algoritmos",
    y="horas_estudio_semana",
    z="promedio_acumulado",
    color=TARGET,
    size="nota_bases_datos",
    title="Educación + IA — Dispersión 3D",
    height=750
)
write_html(fig_3d, OUTDIR / "interactive_scatter_3d.html")

# d) UMAP 2D
print("Generando UMAP 2D...")
X_umap = df[NUM_COLS].copy()
scaler_umap = StandardScaler()
X_umap_scaled = scaler_umap.fit_transform(X_umap)

umap_2d = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)

emb_2d = umap_2d.fit_transform(X_umap_scaled)
df_umap = pd.DataFrame(emb_2d, columns=["UMAP_1", "UMAP_2"])
df_umap[TARGET] = df[TARGET]

fig_umap = px.scatter(
    df_umap,
    x="UMAP_1",
    y="UMAP_2",
    color=TARGET,
    title="Educación + IA — UMAP 2D",
    height=700
)
write_html(fig_umap, OUTDIR / "interactive_umap_2d.html")

# e) UMAP 3D
print("Generando UMAP 3D...")
umap_3d = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=3,
    random_state=42
)

emb_3d = umap_3d.fit_transform(X_umap_scaled)
df_umap3 = pd.DataFrame(emb_3d, columns=["UMAP_1", "UMAP_2", "UMAP_3"])
df_umap3[TARGET] = df[TARGET]

fig_umap3 = px.scatter_3d(
    df_umap3,
    x="UMAP_1",
    y="UMAP_2",
    z="UMAP_3",
    color=TARGET,
    title="Educación + IA — UMAP 3D",
    height=750
)
write_html(fig_umap3, OUTDIR / "interactive_umap_3d.html")

print("✔ Visualizaciones guardadas en", OUTDIR)

# ---------------------------
# 3) Limpieza
# ---------------------------
print("\n=== 3) Limpieza ===")

for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

for c in CAT_COLS:
    df[c] = df[c].fillna("__MISSING__")

# ---------------------------
# 4) Preparación X / y
# ---------------------------
print("\n=== 4) Preparación para ML ===")

X = df.drop(columns=[TARGET])
y = df[TARGET]

X = pd.get_dummies(X, columns=CAT_COLS, drop_first=True)

# ---------------------------
# 5) Split TRAIN / VAL / TEST
# ---------------------------
print("\n=== 5) Split Train / Validation / Test ===")

# 80% temp, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 60% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

# ---------------------------
# 6) Escalado (solo con TRAIN)
# ---------------------------
print("\n=== 6) Escalado ===")

scaler = StandardScaler()
X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
X_val[NUM_COLS]   = scaler.transform(X_val[NUM_COLS])
X_test[NUM_COLS]  = scaler.transform(X_test[NUM_COLS])

print("Shapes:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

# ---------------------------
# 7) Exportación
# ---------------------------
print("\n=== 7) Exportación ===")

X_train.to_parquet(OUTDIR / "X_train.parquet", index=False)
X_val.to_parquet(OUTDIR / "X_val.parquet", index=False)
X_test.to_parquet(OUTDIR / "X_test.parquet", index=False)

y_train.to_frame(name=TARGET).to_parquet(OUTDIR / "y_train.parquet", index=False)
y_val.to_frame(name=TARGET).to_parquet(OUTDIR / "y_val.parquet", index=False)
y_test.to_frame(name=TARGET).to_parquet(OUTDIR / "y_test.parquet", index=False)

schema = {
    "dataset": DATA_CSV,
    "target": TARGET,
    "num_features": NUM_COLS,
    "cat_features": CAT_COLS,
    "binary_features": BIN_COLS,
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
    "split_strategy": "Train / Validation / Test (60/20/20)",
    "notes": [
        "Dataset sintético con fines académicos",
        "Clasificación supervisada binaria",
        "Escalado entrenado solo con TRAIN",
        "Validación usada para selección de modelo"
    ]
}

with open(OUTDIR / "processed_schema.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2, ensure_ascii=False)

with open(OUTDIR / "features_columns.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(X_train.columns))

print("\n✔ Pipeline de EDA + Preprocesamiento + Split COMPLETO.")
