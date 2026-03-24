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
# Clases: Subtractive Clustering y Fuzzy C-Means
# =========================
class SubtractiveClustering:
    def __init__(self, ra=0.5, rb=0.75, eps_upper=0.5, eps_lower=0.15):
        self.ra = ra
        self.rb = rb
        self.eps_upper = eps_upper
        self.eps_lower = eps_lower
        self.centers_ = None
        self.n_clusters_ = 0
        self._x_min = None
        self._x_range = None

    @staticmethod
    def _normalise(X):
        x_min = X.min(axis=0)
        x_range = X.max(axis=0) - x_min
        x_range[x_range == 0] = 1.0
        X_norm = (X - x_min) / x_range
        return X_norm, x_min, x_range

    @staticmethod
    def _potential(X, center, radius):
        dist_sq = np.sum(((X - center) / (radius / 2.0)) ** 2, axis=1)
        return np.exp(-dist_sq)

    def fit(self, X):
        X_norm, self._x_min, self._x_range = self._normalise(X)

        D_a = np.zeros(len(X_norm))
        for xi in X_norm:
            D_a += self._potential(X_norm, xi, self.ra)

        centers_norm = []
        D = D_a.copy()
        D1_max = D.max()

        while True:
            idx = np.argmax(D)
            P_k = D[idx]
            c_k = X_norm[idx]
            ratio = P_k / D1_max if D1_max > 0 else 0

            if ratio > self.eps_upper:
                accept = True
            elif ratio < self.eps_lower:
                break
            else:
                if centers_norm:
                    d_min = min(np.linalg.norm(c_k - c) for c in centers_norm)
                else:
                    d_min = np.inf
                accept = (d_min / self.ra + ratio) >= 1.0

                if not accept:
                    D[idx] = 0.0
                    if D.max() == 0:
                        break
                    continue

            if accept:
                centers_norm.append(c_k)
                D -= P_k * self._potential(X_norm, c_k, self.rb)
                D = np.clip(D, 0, None)
                if D.max() == 0:
                    break
            else:
                break

        if len(centers_norm) == 0:
            self.centers_ = np.mean(X, axis=0, keepdims=True)
            self.n_clusters_ = 1
        else:
            self.centers_ = np.array(
                [c * self._x_range + self._x_min for c in centers_norm]
            )
            self.n_clusters_ = len(self.centers_)

        return self

    def predict(self, X):
        dists = np.array([np.linalg.norm(X - c, axis=1) for c in self.centers_])
        return np.argmin(dists, axis=0)


class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2.0, max_iter=300, tol=1e-6, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers_ = None
        self.U_ = None
        self.history_ = []
        self.n_iter_ = 0

    def _init_membership(self, n_samples):
        rng = np.random.default_rng(self.random_state)
        U = rng.random((n_samples, self.n_clusters))
        U /= U.sum(axis=1, keepdims=True)
        return U

    def _update_centers(self, X, U):
        um = U ** self.m
        return (um.T @ X) / um.sum(axis=0)[:, None]

    def _update_membership(self, X, centers):
        n = len(X)
        c = len(centers)

        dist = np.array([
            np.linalg.norm(X - centers[k], axis=1)
            for k in range(c)
        ]).T

        dist = np.fmax(dist, np.finfo(float).eps)
        exp = 2.0 / (self.m - 1)

        U = np.zeros((n, c))
        for k in range(c):
            ratio = dist[:, k:k+1] / dist
            U[:, k] = 1.0 / (ratio ** exp).sum(axis=1)

        return U

    def _objective(self, X, U, centers):
        dist_sq = np.array([
            np.sum((X - centers[k]) ** 2, axis=1)
            for k in range(len(centers))
        ]).T
        return np.sum((U ** self.m) * dist_sq)

    def fit(self, X, init_centers=None):
        if init_centers is not None and len(init_centers) == self.n_clusters:
            centers = init_centers.copy()
            U = self._update_membership(X, centers)
        else:
            U = self._init_membership(len(X))
            centers = self._update_centers(X, U)

        for it in range(1, self.max_iter + 1):
            U_old = U.copy()
            centers = self._update_centers(X, U)
            U = self._update_membership(X, centers)
            J = self._objective(X, U, centers)
            self.history_.append(J)

            delta = np.max(np.abs(U - U_old))
            if delta < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter

        self.centers_ = centers
        self.U_ = U
        return self

    def predict(self, X=None):
        if X is not None:
            U = self._update_membership(X, self.centers_)
        else:
            U = self.U_
        return np.argmax(U, axis=1)

    def fit_predict(self, X, init_centers=None):
        self.fit(X, init_centers=init_centers)
        return self.predict()


# =========================
# Función auxiliar de métricas
# =========================
def safe_silhouette(X, labels, nombre_modelo="modelo"):
    unique = np.unique(labels)
    if len(unique) < 2:
        print(f"{nombre_modelo}: no hay suficientes clústeres para silhouette.")
        return None
    try:
        return silhouette_score(X, labels)
    except Exception as e:
        print(f"{nombre_modelo}: no se pudo calcular silhouette. Error: {e}")
        return None


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
X_processed = np.asarray(X_processed, dtype=float)

print("\nForma de la matriz transformada:", X_processed.shape)
print("¿Hay NaN después del preprocesamiento?:", np.isnan(X_processed).sum())

# =========================
# 6. PCA 2D y 3D
# =========================
pca = PCA(n_components=2, random_state=random_state)
X_pca = pca.fit_transform(X_processed)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], s=30)
ax.set_title("Datos proyectados en 2D con PCA")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
plt.show()

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
# 8. KMeans final
# =========================
k_optimo = 2
print(f"\nUsando k = {k_optimo} clusters")

modelo_kmeans_final = KMeans(
    n_clusters=k_optimo,
    random_state=random_state,
    n_init=10
)

labels_kmeans = modelo_kmeans_final.fit_predict(X_processed)
sil_kmeans = safe_silhouette(X_processed, labels_kmeans, "KMeans")

print(f"\nKMeans con k = {k_optimo}")
print("Inercia:", modelo_kmeans_final.inertia_)
print("Silhouette:", sil_kmeans)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans)
ax.set_title(f"KMeans con k = {k_optimo}")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
plt.show()

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

centroides_kmeans = modelo_kmeans_final.cluster_centers_
centroides_kmeans_3d = pca_3d.transform(centroides_kmeans)

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
    centroides_kmeans_3d[:, 0],
    centroides_kmeans_3d[:, 1],
    centroides_kmeans_3d[:, 2],
    s=200,
    marker='X'
)
ax.set_title(f"KMeans en 3D con centroides (k = {k_optimo})")
ax.set_xlabel("CP1")
ax.set_ylabel("CP2")
ax.set_zlabel("CP3")
plt.show()

# =========================
# 9. DBSCAN
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
    sil_db = None
    print("DBSCAN no encontró suficientes clústeres válidos para calcular silhouette.")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan)
ax.set_title("DBSCAN")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
plt.show()

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
# 10. Subtractive Clustering
# =========================
modelo_sub = SubtractiveClustering(
    ra=0.5,
    rb=0.75,
    eps_upper=0.5,
    eps_lower=0.15
)
modelo_sub.fit(X_processed)
labels_sub = modelo_sub.predict(X_processed)
sil_sub = safe_silhouette(X_processed, labels_sub, "Subtractive")

print("\nSubtractive Clustering")
print("Número de clústeres encontrados:", modelo_sub.n_clusters_)
print("Silhouette:", sil_sub)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_sub)
ax.set_title(f"Subtractive Clustering (k = {modelo_sub.n_clusters_})")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
plt.show()

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=labels_sub,
    s=35
)
ax.set_title(f"Subtractive Clustering en 3D (k = {modelo_sub.n_clusters_})")
ax.set_xlabel("CP1")
ax.set_ylabel("CP2")
ax.set_zlabel("CP3")
plt.show()

if modelo_sub.centers_ is not None and len(modelo_sub.centers_) > 0:
    centroides_sub_3d = pca_3d.transform(modelo_sub.centers_)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        X_pca_3d[:, 0],
        X_pca_3d[:, 1],
        X_pca_3d[:, 2],
        c=labels_sub,
        s=30
    )
    ax.scatter(
        centroides_sub_3d[:, 0],
        centroides_sub_3d[:, 1],
        centroides_sub_3d[:, 2],
        s=200,
        marker='X'
    )
    ax.set_title("Subtractive Clustering en 3D con centros")
    ax.set_xlabel("CP1")
    ax.set_ylabel("CP2")
    ax.set_zlabel("CP3")
    plt.show()

# =========================
# 11. Fuzzy C-Means
# =========================
# Si subtractive encontró una cantidad razonable de centros, los usa para inicializar FCM.
# Si no, usa k_optimo.
if modelo_sub.n_clusters_ >= 2:
    n_clusters_fcm = modelo_sub.n_clusters_
    init_centers_fcm = modelo_sub.centers_
else:
    n_clusters_fcm = k_optimo
    init_centers_fcm = None

modelo_fcm = FuzzyCMeans(
    n_clusters=n_clusters_fcm,
    m=2.0,
    max_iter=300,
    tol=1e-6,
    random_state=random_state
)

labels_fcm = modelo_fcm.fit_predict(X_processed, init_centers=init_centers_fcm)
sil_fcm = safe_silhouette(X_processed, labels_fcm, "Fuzzy C-Means")

print("\nFuzzy C-Means")
print("Número de clústeres:", n_clusters_fcm)
print("Iteraciones:", modelo_fcm.n_iter_)
print("Silhouette:", sil_fcm)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_fcm)
ax.set_title(f"Fuzzy C-Means (k = {n_clusters_fcm})")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
plt.show()

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=labels_fcm,
    s=35
)
ax.set_title(f"Fuzzy C-Means en 3D (k = {n_clusters_fcm})")
ax.set_xlabel("CP1")
ax.set_ylabel("CP2")
ax.set_zlabel("CP3")
plt.show()

centroides_fcm_3d = pca_3d.transform(modelo_fcm.centers_)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=labels_fcm,
    s=30
)
ax.scatter(
    centroides_fcm_3d[:, 0],
    centroides_fcm_3d[:, 1],
    centroides_fcm_3d[:, 2],
    s=200,
    marker='X'
)
ax.set_title("Fuzzy C-Means en 3D con centros")
ax.set_xlabel("CP1")
ax.set_ylabel("CP2")
ax.set_zlabel("CP3")
plt.show()

# =========================
# 12. Comparación con etiqueta real
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

    ari_kmeans = adjusted_rand_score(y_real, labels_kmeans)
    print(f"\nAdjusted Rand Index entre etiquetas reales y KMeans: {ari_kmeans:.4f}")

    # ARI de subtractive
    try:
        ari_sub = adjusted_rand_score(y_real, labels_sub)
        print(f"Adjusted Rand Index entre etiquetas reales y Subtractive: {ari_sub:.4f}")
    except Exception as e:
        print(f"No se pudo calcular ARI para Subtractive: {e}")

    # ARI de FCM
    try:
        ari_fcm = adjusted_rand_score(y_real, labels_fcm)
        print(f"Adjusted Rand Index entre etiquetas reales y Fuzzy C-Means: {ari_fcm:.4f}")
    except Exception as e:
        print(f"No se pudo calcular ARI para Fuzzy C-Means: {e}")

# =========================
# 13. Guardar resultados
# =========================
df_resultado = df.copy()
df_resultado["cluster_kmeans"] = labels_kmeans
df_resultado["cluster_dbscan"] = labels_dbscan
df_resultado["cluster_subtractive"] = labels_sub
df_resultado["cluster_fcm"] = labels_fcm

if modelo_fcm.U_ is not None:
    for i in range(modelo_fcm.U_.shape[1]):
        df_resultado[f"fcm_mu_cluster_{i}"] = modelo_fcm.U_[:, i]

print("\nPrimeras filas con clusters asignados:")
print(df_resultado.head())

df_resultado.to_csv("resultado_clustering_completo.csv", index=False)
print("\nArchivo guardado: resultado_clustering_completo.csv")
