import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# =========================
# Configuración general
# =========================
random_state = 42
plt.rc('font', family='serif', size=12)

# =========================
# 1. Cargar dataset
# =========================
ruta = "dataset_sintetico_FIRE_UdeA.csv"
df = pd.read_csv(ruta)

print("Primeras filas del dataset:")
print(df.head())
print("\nDimensiones:", df.shape)
print("\nColumnas:", df.columns.tolist())

# =========================
# 2. Separar variables
# =========================
# En aprendizaje no supervisado NO usamos la etiqueta para entrenar
if "label" in df.columns:
    y_real = df["label"]   # opcional: solo para comparar después
    X = df.drop(columns=["label"])
else:
    y_real = None
    X = df.copy()

print("\nVariables usadas para clustering:")
print(X.columns.tolist())

# =========================
# 3. Preprocesamiento
# =========================
numeric_features = X.columns.tolist()

numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
    ],
    remainder="drop"
)

# =========================
# 4. Visualización 2D con PCA
# =========================
X_scaled = preprocessor.fit_transform(X)

pca = PCA(n_components=2, random_state=random_state)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1])
ax.set_title("Datos proyectados en 2D con PCA")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
fig.set_size_inches(8, 5)
plt.show()

# =========================
# 5. Método del codo con KMeans
# =========================
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    clu_kmeans = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clustering", KMeans(n_clusters=k, random_state=random_state, n_init=10))
    ])
    
    clu_kmeans.fit(X)
    
    labels_k = clu_kmeans["clustering"].labels_
    inertia_k = clu_kmeans["clustering"].inertia_
    sil_k = silhouette_score(preprocessor.transform(X), labels_k)
    
    inertias.append(inertia_k)
    silhouette_scores.append(sil_k)

fig, ax = plt.subplots()
ax.plot(list(k_range), inertias, marker='o')
ax.set_title("Método del codo - KMeans")
ax.set_xlabel("Número de clústeres (k)")
ax.set_ylabel("Inercia")
fig.set_size_inches(8, 5)
plt.show()

fig, ax = plt.subplots()
ax.plot(list(k_range), silhouette_scores, marker='o')
ax.set_title("Silhouette Score - KMeans")
ax.set_xlabel("Número de clústeres (k)")
ax.set_ylabel("Silhouette")
fig.set_size_inches(8, 5)
plt.show()

# =========================
# 6. Entrenar KMeans con un k elegido
# =========================
# Forzar k=2 clusters
k_optimo = 2

clu_kmeans = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clustering", KMeans(n_clusters=k_optimo, random_state=random_state, n_init=10))
])

clu_kmeans.fit(X)
labels_kmeans = clu_kmeans["clustering"].labels_

print(f"\nKMeans con k = {k_optimo}")
print("Inercia:", clu_kmeans["clustering"].inertia_)
print("Silhouette:", silhouette_score(preprocessor.transform(X), labels_kmeans))

fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans)
ax.set_title(f"KMeans con k = {k_optimo}")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
fig.set_size_inches(8, 5)
plt.show()

# =========================
# 7. Entrenar DBSCAN
# =========================
# Aquí sí usamos los datos escalados directamente
clu_dbscan = DBSCAN(eps=0.8, min_samples=10)
labels_dbscan = clu_dbscan.fit_predict(X_scaled)

print("\nDBSCAN")
print("Etiquetas encontradas:", np.unique(labels_dbscan))
print("Conteo por etiqueta:", np.unique(labels_dbscan, return_counts=True))

# Silhouette solo si hay más de un cluster válido
labels_unicos = set(labels_dbscan)
if len(labels_unicos - {-1}) > 1:
    mask = labels_dbscan != -1
    sil_db = silhouette_score(X_scaled[mask], labels_dbscan[mask])
    print("Silhouette DBSCAN (sin ruido):", sil_db)
else:
    print("DBSCAN no encontró suficientes clústeres para calcular silhouette.")

fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan)
ax.set_title("DBSCAN")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
fig.set_size_inches(8, 5)
plt.show()

# =========================
# 8. Comparación opcional con label real
# =========================
if y_real is not None:
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_real)
    ax.set_title("Etiqueta real del dataset (solo comparación)")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    fig.set_size_inches(8, 5)
    plt.show()
    
    # Comparación cuantitativa
    ari_score = adjusted_rand_score(y_real, labels_kmeans)
    print(f"\nAdjusted Rand Index entre etiquetas reales y KMeans: {ari_score:.4f}")
