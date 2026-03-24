import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN, KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.impute import SimpleImputer

# =========================
# Configuración general
# =========================
random_state = 42
plt.rc('font', family='serif', size=12)

# =========================
# 1. Cargar dataset
# =========================
ruta = "dataset_sintetico_FIRE_UdeA_realista.csv"
df = pd.read_csv(ruta)

print("Primeras filas del dataset:")
print(df.head())
print("\nDimensiones:", df.shape)
print("\nColumnas:", df.columns.tolist())
print("\nTipos de datos:")
print(df.dtypes)

print("\nValores faltantes por columna:")
print(df.isna().sum())

# =========================
# 2. Separar variables
# =========================
if "label" in df.columns:
    y_real = df["label"]
    X = df.drop(columns=["label"])
else:
    y_real = None
    X = df.copy()

print("\nVariables usadas para clustering:")
print(X.columns.tolist())

# =========================
# 3. Identificar columnas numéricas y categóricas
# =========================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nVariables numéricas:", numeric_features)
print("Variables categóricas:", categorical_features)

# =========================
# 4. Preprocesamiento
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

# =========================
# 5. Transformar datos
# =========================
X_processed = preprocessor.fit_transform(X)

print("\nForma de la matriz transformada:", X_processed.shape)
print("¿Hay NaN después del preprocesamiento?:", np.isnan(X_processed).sum())

# =========================
# 6. Visualización 2D con PCA
# =========================
pca = PCA(n_components=2, random_state=random_state)
X_pca = pca.fit_transform(X_processed)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1])
ax.set_title("Datos proyectados en 2D con PCA")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
plt.show()

# =========================
# 6B. Visualización 3D con PCA
# =========================
pca_3d = PCA(n_components=3, random_state=random_state)
X_pca_3d = pca_3d.fit_transform(X_processed)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], s=30)
ax.set_title("Datos proyectados en 3D con PCA")
ax.set_xlabel("CP1")
ax.set_ylabel("CP2")
ax.set_zlabel("CP3")
plt.show()

# =========================
# 7. Método del codo + silhouette con KMeans
# =========================
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    modelo_kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels_k = modelo_kmeans.fit_predict(X_processed)
    inertias.append(modelo_kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_processed, labels_k))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(k_range), inertias, marker='o')
ax.set_title("Método del codo - KMeans")
ax.set_xlabel("Número de clústeres (k)")
ax.set_ylabel("Inercia")
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(k_range), silhouette_scores, marker='o')
ax.set_title("Silhouette Score - KMeans")
ax.set_xlabel("Número de clústeres (k)")
ax.set_ylabel("Silhouette")
plt.show()

# =========================
# 8. Elegir mejor k automáticamente
# =========================
k_optimo = 2
print(f"\nUsando k = {k_optimo} clusters")

# =========================
# 9. Entrenar KMeans final
# =========================
modelo_kmeans_final = KMeans(
    n_clusters=k_optimo,
    random_state=random_state,
    n_init=10
)

labels_kmeans = modelo_kmeans_final.fit_predict(X_processed)

print(f"\nKMeans con k = {k_optimo}")
print("Inercia:", modelo_kmeans_final.inertia_)
print("Silhouette:", silhouette_score(X_processed, labels_kmeans))

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans)
ax.set_title(f"KMeans con k = {k_optimo}")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
plt.show()

# =========================
# 9B. KMeans en 3D
# =========================
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=labels_kmeans,
    s=35
)
ax.set_title(f"KMeans en 3D con k = {k_optimo}")
ax.set_xlabel("CP1")
ax.set_ylabel("CP2")
ax.set_zlabel("CP3")
plt.show()

# Centroides de KMeans proyectados a 3D
centroides = modelo_kmeans_final.cluster_centers_
centroides_3d = pca_3d.transform(centroides)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=labels_kmeans,
    s=30
)
ax.scatter(
    centroides_3d[:, 0],
    centroides_3d[:, 1],
    centroides_3d[:, 2],
    s=200,
    marker='X'
)
ax.set_title(f"KMeans en 3D con centroides (k = {k_optimo})")
ax.set_xlabel("CP1")
ax.set_ylabel("CP2")
ax.set_zlabel("CP3")
plt.show()

# =========================
# 10. DBSCAN
# =========================
modelo_dbscan = DBSCAN(eps=1.2, min_samples=5)
labels_dbscan = modelo_dbscan.fit_predict(X_processed)

print("\nDBSCAN")
print("Etiquetas encontradas:", np.unique(labels_dbscan))
print("Conteo por etiqueta:", np.unique(labels_dbscan, return_counts=True))

labels_unicos = set(labels_dbscan)
if len(labels_unicos - {-1}) > 1:
    mask = labels_dbscan != -1
    sil_db = silhouette_score(X_processed[mask], labels_dbscan[mask])
    print("Silhouette DBSCAN (sin ruido):", sil_db)
else:
    print("DBSCAN no encontró suficientes clústeres válidos para calcular silhouette.")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan)
ax.set_title("DBSCAN")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
plt.show()

# =========================
# 10B. DBSCAN en 3D
# =========================
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=labels_dbscan,
    s=35
)
ax.set_title("DBSCAN en 3D")
ax.set_xlabel("CP1")
ax.set_ylabel("CP2")
ax.set_zlabel("CP3")
plt.show()

# =========================
# 11. Comparación con la etiqueta real
# =========================
if y_real is not None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_real)
    ax.set_title("Etiqueta real del dataset (solo comparación)")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    plt.show()

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        X_pca_3d[:, 0],
        X_pca_3d[:, 1],
        X_pca_3d[:, 2],
        c=y_real,
        s=35
    )
    ax.set_title("Etiquetas reales del dataset en 3D")
    ax.set_xlabel("CP1")
    ax.set_ylabel("CP2")
    ax.set_zlabel("CP3")
    plt.show()

    ari_score = adjusted_rand_score(y_real, labels_kmeans)
    print(f"\nAdjusted Rand Index entre etiquetas reales y KMeans: {ari_score:.4f}")

# =========================
# 12. Guardar resultados
# =========================
df_resultado = df.copy()
df_resultado["cluster_kmeans"] = labels_kmeans
df_resultado["cluster_dbscan"] = labels_dbscan

print("\nPrimeras filas con clusters asignados:")
print(df_resultado.head())
