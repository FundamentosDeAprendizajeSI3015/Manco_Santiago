# ==============================================================
# INFORME TEÓRICO-PRÁCTICO 02 — Machine Learning
# Autor: Santiago Manco Maya
# Dataset: dataset_ingenieria_sistemas_ia_300_realista.csv
# Target: preparacion_laboral (0 = No preparado, 1 = Preparado)
# ==============================================================
# CONTENIDO:
#   PARTE 1 — Análisis NO supervisado
#             KMeans, Fuzzy C-Means, Subtractive Clustering,
#             DBSCAN, Agglomerative Clustering (familia)
#   PARTE 2 — Re-evaluación de etiquetas (~30% pueden estar mal)
#   PARTE 3 — Modelos supervisados con etiquetas re-evaluadas
#             Árbol de Decisión, Regresión Logística, Regresión Lineal
#   PARTE 4 — Comparación: dataset original vs dataset re-etiquetado
# ==============================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # sin ventana, guarda en disco
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Clustering
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
)
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, mean_squared_error, r2_score
)

# Supervisados
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression, LinearRegression

# ==============================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================

random_state = 42
OUTDIR = Path("./output")
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rc("font", family="serif", size=11)

def guardar(fig, nombre):
    ruta = OUTDIR / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [guardado] {ruta}")

# ==============================================================
# CARGA Y PREPROCESAMIENTO BASE
# ==============================================================

print("\n" + "="*60)
print("CARGANDO DATASET")
print("="*60)

data = pd.read_csv("dataset_ingenieria_sistemas_ia_300_realista.csv")
print(data.head())
print("\nShape:", data.shape)
print("\nDistribución del target (original):")
print(data["preparacion_laboral"].value_counts())

# --- Separar X e y ---
X_raw = data.drop(columns=["preparacion_laboral"])
y_original = data["preparacion_laboral"].values

cat_cols = X_raw.select_dtypes(include="object").columns.tolist()
num_cols = X_raw.select_dtypes(include=np.number).columns.tolist()

print(f"\nColumnas numéricas  ({len(num_cols)}): {num_cols}")
print(f"Columnas categóricas ({len(cat_cols)}): {cat_cols}")

# --- Pipeline de preprocesamiento (para clustering y modelos) ---
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

X_proc = preprocessor.fit_transform(X_raw)
print(f"\nMatriz preprocesada: {X_proc.shape}")

# --- PCA 2D para visualizaciones ---
pca2 = PCA(n_components=2, random_state=random_state)
X_pca = pca2.fit_transform(X_proc)

# ==============================================================
# PARTE 1 — ANÁLISIS NO SUPERVISADO
# ==============================================================

print("\n" + "="*60)
print("PARTE 1 — ANÁLISIS NO SUPERVISADO")
print("="*60)

# ----------------------------------------------------------
# 1A. KMeans — Método del codo + Silhouette
# ----------------------------------------------------------
print("\n[1A] KMeans — método del codo + silhouette...")
inertias, sil_scores = [], []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    lbl = km.fit_predict(X_proc)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_proc, lbl))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_title("KMeans — Método del Codo")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inercia")
axes[1].plot(list(k_range), sil_scores, marker="o", color="coral")
axes[1].set_title("KMeans — Silhouette Score")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
fig.tight_layout()
guardar(fig, "01_kmeans_codo_silhouette.png")

k_optimo = list(k_range)[np.argmax(sil_scores)]
print(f"  => Mejor k segun silhouette: {k_optimo}")

km_final = KMeans(n_clusters=k_optimo, random_state=random_state, n_init=10)
labels_kmeans = km_final.fit_predict(X_proc)
print(f"  Silhouette final: {silhouette_score(X_proc, labels_kmeans):.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap="tab10", s=40)
ax.set_title(f"KMeans — k={k_optimo} (PCA 2D)")
ax.set_xlabel("CP1"); ax.set_ylabel("CP2")
fig.colorbar(sc, ax=ax, label="Cluster")
guardar(fig, "02_kmeans_clusters_pca.png")

# ----------------------------------------------------------
# 1B. Fuzzy C-Means
# ----------------------------------------------------------
print("\n[1B] Fuzzy C-Means...")
try:
    import skfuzzy as fuzz
    n_clusters_fcm = k_optimo
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_proc.T, n_clusters_fcm, 2,
        error=0.005, maxiter=1000, init=None
    )
    labels_fcm = np.argmax(u, axis=0)
    sil_fcm = silhouette_score(X_proc, labels_fcm)
    print(f"  Silhouette FCM: {sil_fcm:.4f}  |  FPC: {fpc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_fcm, cmap="tab10", s=40)
    ax.set_title(f"Fuzzy C-Means — c={n_clusters_fcm} (PCA 2D)")
    ax.set_xlabel("CP1"); ax.set_ylabel("CP2")
    fig.colorbar(sc, ax=ax, label="Cluster")
    guardar(fig, "03_fuzzy_cmeans_pca.png")

except ImportError:
    print("  skfuzzy no instalado — implementando FCM manual...")

    def fuzzy_cmeans_manual(X, c, m=2, max_iter=200, tol=1e-5, seed=42):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        # Inicializar membresías aleatoriamente
        U = rng.random((n, c))
        U = U / U.sum(axis=1, keepdims=True)
        for _ in range(max_iter):
            Um = U ** m
            centers = (Um.T @ X) / Um.sum(axis=0)[:, None]
            dist = np.array([np.linalg.norm(X - centers[j], axis=1) for j in range(c)]).T
            dist = np.maximum(dist, 1e-10)
            U_new = np.zeros_like(U)
            for j in range(c):
                ratio = (dist[:, j:j+1] / dist) ** (2 / (m - 1))
                U_new[:, j] = 1.0 / ratio.sum(axis=1)
            if np.linalg.norm(U_new - U) < tol:
                break
            U = U_new
        return U, centers

    U_fcm, centers_fcm = fuzzy_cmeans_manual(X_proc, c=k_optimo, seed=random_state)
    labels_fcm = np.argmax(U_fcm, axis=1)
    sil_fcm = silhouette_score(X_proc, labels_fcm)
    print(f"  Silhouette FCM (manual): {sil_fcm:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_fcm, cmap="tab10", s=40)
    ax.set_title(f"Fuzzy C-Means — c={k_optimo} (PCA 2D)")
    ax.set_xlabel("CP1"); ax.set_ylabel("CP2")
    fig.colorbar(sc, ax=ax, label="Cluster")
    guardar(fig, "03_fuzzy_cmeans_pca.png")

# ----------------------------------------------------------
# 1C. Subtractive Clustering
# ----------------------------------------------------------
print("\n[1C] Subtractive Clustering...")

def subtractive_clustering(X, ra=0.5, rb_factor=1.5, eps_up=0.5, eps_down=0.15):
    """
    Implementación de Subtractive Clustering (Chiu, 1994).
    ra  : radio de vecindad para calcular potencial
    rb  : radio de rechazo = ra * rb_factor
    """
    rb = ra * rb_factor
    X_s = (X - X.min(axis=0)) / (np.ptp(X, axis=0) + 1e-10)   # normalizar [0,1]
    n, d = X_s.shape

    # Potencial inicial de cada punto
    alpha = 4 / (ra ** 2)
    beta  = 4 / (rb ** 2)

    pot = np.array([
        np.exp(-alpha * np.sum((X_s - X_s[i]) ** 2, axis=1)).sum()
        for i in range(n)
    ])

    centers_idx = []
    pot_max0 = pot.max()

    while True:
        i_best = np.argmax(pot)
        pot_best = pot[i_best]

        if not centers_idx:
            threshold_accept = eps_up * pot_max0
            threshold_reject = eps_down * pot_max0
        else:
            threshold_accept = eps_up * pot_max0
            threshold_reject = eps_down * pot_max0

        if pot_best >= threshold_accept:
            centers_idx.append(i_best)
        elif pot_best <= threshold_reject:
            break
        else:
            # Criterio mixto
            d_min = min(np.linalg.norm(X_s[i_best] - X_s[c]) for c in centers_idx)
            if (d_min / ra) + (pot_best / pot_max0) >= 1:
                centers_idx.append(i_best)
            else:
                pot[i_best] = 0
                continue

        # Reducir potencial de puntos cercanos al nuevo centro
        pot -= pot_best * np.exp(-beta * np.sum((X_s - X_s[i_best]) ** 2, axis=1))
        pot[i_best] = 0

        if len(centers_idx) > 30:    # límite de seguridad
            break

    centers_idx = list(dict.fromkeys(centers_idx))   # quitar duplicados
    if not centers_idx:
        centers_idx = [np.argmax(pot)]

    # Asignar cada punto al centro más cercano
    centers = X_s[centers_idx]
    labels_sub = np.argmin(
        np.array([np.linalg.norm(X_s - c, axis=1) for c in centers]),
        axis=0
    )
    return labels_sub, centers_idx

labels_sub, centers_idx_sub = subtractive_clustering(X_proc, ra=0.5)
n_clusters_sub = len(np.unique(labels_sub))
print(f"  Centros detectados: {n_clusters_sub}")
if n_clusters_sub > 1:
    sil_sub = silhouette_score(X_proc, labels_sub)
    print(f"  Silhouette Subtractive: {sil_sub:.4f}")
else:
    sil_sub = None
    print("  Solo 1 clúster — silhouette no aplica")

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_sub, cmap="tab10", s=40)
ax.set_title(f"Subtractive Clustering — {n_clusters_sub} centros (PCA 2D)")
ax.set_xlabel("CP1"); ax.set_ylabel("CP2")
fig.colorbar(sc, ax=ax, label="Cluster")
guardar(fig, "04_subtractive_clustering_pca.png")

# ----------------------------------------------------------
# 1D. DBSCAN
# ----------------------------------------------------------
print("\n[1D] DBSCAN...")
dbscan = DBSCAN(eps=1.2, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_proc)
n_clusters_db = len(set(labels_dbscan) - {-1})
n_noise_db    = (labels_dbscan == -1).sum()
print(f"  Clústeres: {n_clusters_db}  |  Ruido: {n_noise_db}")
if n_clusters_db > 1:
    mask_db = labels_dbscan != -1
    sil_db = silhouette_score(X_proc[mask_db], labels_dbscan[mask_db])
    print(f"  Silhouette DBSCAN (sin ruido): {sil_db:.4f}")
else:
    sil_db = None
    print("  DBSCAN no generó suficientes clústeres para silhouette.")

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan, cmap="tab10", s=40)
ax.set_title(f"DBSCAN — {n_clusters_db} clústeres (PCA 2D)")
ax.set_xlabel("CP1"); ax.set_ylabel("CP2")
fig.colorbar(sc, ax=ax, label="Cluster")
guardar(fig, "05_dbscan_pca.png")

# ----------------------------------------------------------
# 1E. Familia de clustering
# ----------------------------------------------------------
print("\n[1E] Familia de clustering (Agglomerative, Spectral, BIRCH)...")

cluster_family = {
    "Agglomerative (Ward)":    AgglomerativeClustering(n_clusters=k_optimo, linkage="ward"),
    "Agglomerative (Average)": AgglomerativeClustering(n_clusters=k_optimo, linkage="average"),
    "Agglomerative (Complete)":AgglomerativeClustering(n_clusters=k_optimo, linkage="complete"),
    "Spectral":                SpectralClustering(n_clusters=k_optimo, random_state=random_state, affinity="rbf"),
    "BIRCH":                   Birch(n_clusters=k_optimo),
}

sil_family = {}
fig_fam, axes_fam = plt.subplots(2, 3, figsize=(18, 10))
axes_fam = axes_fam.flatten()

for idx, (name, model) in enumerate(cluster_family.items()):
    lbl = model.fit_predict(X_proc)
    sil = silhouette_score(X_proc, lbl)
    sil_family[name] = sil
    print(f"  {name}: Silhouette={sil:.4f}")
    ax = axes_fam[idx]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=lbl, cmap="tab10", s=25)
    ax.set_title(f"{name}\nSil={sil:.3f}")
    ax.set_xlabel("CP1"); ax.set_ylabel("CP2")

# Último subplot: comparación silhouette
ax_bar = axes_fam[len(cluster_family)]
methods_bar = list(sil_family.keys())
vals_bar    = list(sil_family.values())
ax_bar.barh(methods_bar, vals_bar, color="steelblue")
ax_bar.set_xlabel("Silhouette Score")
ax_bar.set_title("Comparación — Familia de Clustering")
fig_fam.tight_layout()
guardar(fig_fam, "06_familia_clustering_pca.png")

# ----------------------------------------------------------
# 1F. Resumen comparativo de clustering
# ----------------------------------------------------------
print("\n[1F] Resumen clustering...")

resumen_cluster = {
    "KMeans":          silhouette_score(X_proc, labels_kmeans),
    "Fuzzy C-Means":   sil_fcm,
    "Subtractive":     sil_sub if sil_sub is not None else np.nan,
    "DBSCAN":          sil_db  if sil_db  is not None else np.nan,
}
resumen_cluster.update(sil_family)

df_sil = pd.DataFrame.from_dict(resumen_cluster, orient="index", columns=["Silhouette"])
df_sil = df_sil.sort_values("Silhouette", ascending=False)
print(df_sil.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
df_sil.dropna().plot.barh(ax=ax, legend=False, color="teal")
ax.set_xlabel("Silhouette Score")
ax.set_title("Comparación General — Todos los Métodos de Clustering")
fig.tight_layout()
guardar(fig, "07_comparacion_clustering.png")

# ==============================================================
# PARTE 2 — RE-EVALUACIÓN DE ETIQUETAS
# ==============================================================

print("\n" + "="*60)
print("PARTE 2 — RE-EVALUACIÓN DE ETIQUETAS")
print("="*60)

# Usamos KMeans (mejor silhouette o más interpretable) con k=2
# para detectar etiquetas potencialmente incorrectas.

km2 = KMeans(n_clusters=2, random_state=random_state, n_init=10)
labels_km2 = km2.fit_predict(X_proc)

# Alinear clusters KMeans con etiquetas originales
# (KMeans puede invertir los nombres)
from scipy.stats import mode

def alinear_clusters(y_true, y_pred, n_classes=2):
    """Mapea los IDs de cluster para maximizar coincidencia con y_true."""
    from itertools import permutations
    best_acc, best_perm = 0, None
    for perm in permutations(range(n_classes)):
        mapping = {old: new for old, new in enumerate(perm)}
        mapped = np.vectorize(mapping.get)(y_pred)
        acc = (y_true == mapped).mean()
        if acc > best_acc:
            best_acc, best_perm = acc, perm
    mapping = {old: new for old, new in enumerate(best_perm)}
    return np.vectorize(mapping.get)(y_pred), best_acc

labels_km2_aligned, acc_align = alinear_clusters(y_original, labels_km2)
print(f"\nAlineación KMeans2 vs etiquetas originales: {acc_align:.2%}")

ari_km2 = adjusted_rand_score(y_original, labels_km2_aligned)
print(f"Adjusted Rand Index: {ari_km2:.4f}")

# Identificar muestras con etiqueta diferente al cluster
desacuerdo = (y_original != labels_km2_aligned)
print(f"\nMuestras en desacuerdo (etiqueta ≠ cluster): {desacuerdo.sum()} / {len(y_original)}")
print(f"Porcentaje: {desacuerdo.mean():.1%}")

# Estrategia de re-etiquetado:
#   - Calculamos la "confianza" de cada muestra = distancia al centroide propio.
#   - Las que están en desacuerdo Y tienen alta distancia al centroide asignado
#     se consideran potencialmente mal etiquetadas y se cambia su etiqueta.
#   - Limitamos el cambio al 30% del total.

distancias = km2.transform(X_proc)   # (n, 2)
dist_al_propio = np.array([distancias[i, labels_km2[i]] for i in range(len(labels_km2))])

# Candidatos: en desacuerdo con el cluster alineado
candidatos_idx = np.where(desacuerdo)[0]

# Ordenar candidatos por distancia (mayor distancia = más probable error)
orden_candidatos = candidatos_idx[np.argsort(dist_al_propio[candidatos_idx])[::-1]]

# Máximo a re-etiquetar: 30% del total
max_cambios = int(0.30 * len(y_original))
n_cambios = min(len(candidatos_idx), max_cambios)

print(f"\nRe-etiquetando hasta {n_cambios} muestras ({n_cambios/len(y_original):.1%} del dataset)...")

y_relabeled = y_original.copy()
indices_cambiados = orden_candidatos[:n_cambios]
y_relabeled[indices_cambiados] = 1 - y_relabeled[indices_cambiados]

print(f"Cambios realizados: {n_cambios}")
print(f"\nDistribución original:    {pd.Series(y_original).value_counts().to_dict()}")
print(f"Distribución re-etiquetada: {pd.Series(y_relabeled).value_counts().to_dict()}")

# Visualización: muestras cambiadas en PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colores_orig = ["#e74c3c" if v == 1 else "#3498db" for v in y_original]
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=colores_orig, s=25, alpha=0.7)
axes[0].set_title("Etiquetas originales (PCA 2D)")
axes[0].set_xlabel("CP1"); axes[0].set_ylabel("CP2")

colores_new = ["#e74c3c" if v == 1 else "#3498db" for v in y_relabeled]
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=colores_new, s=25, alpha=0.7)
# Marcar muestras cambiadas
axes[1].scatter(X_pca[indices_cambiados, 0], X_pca[indices_cambiados, 1],
                facecolors="none", edgecolors="gold", s=70, linewidths=1.5,
                label=f"Re-etiquetadas ({n_cambios})")
axes[1].set_title("Etiquetas re-evaluadas (PCA 2D)")
axes[1].set_xlabel("CP1"); axes[1].set_ylabel("CP2")
axes[1].legend(fontsize=9)
fig.tight_layout()
guardar(fig, "08_reetiquetado.png")

# ==============================================================
# PARTE 3 — MODELOS SUPERVISADOS (RE-ETIQUETADO)
# ==============================================================

print("\n" + "="*60)
print("PARTE 3 — MODELOS SUPERVISADOS CON ETIQUETAS RE-EVALUADAS")
print("="*60)

def dividir(X, y, rs=random_state):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4,
                                                  stratify=y, random_state=rs)
    X_v, X_te, y_v, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                              stratify=y_tmp, random_state=rs)
    return X_tr, X_v, X_te, y_tr, y_v, y_te

def construir_pipeline(clasificador):
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols)
        ])),
        ("classifier", clasificador)
    ])

def evaluar(modelo, X, y, nombre, prefijo_fig="", guardar_roc=True):
    y_pred = modelo.predict(X)
    y_prob = modelo.predict_proba(X)[:, 1] if hasattr(modelo, "predict_proba") else None

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    auc  = roc_auc_score(y, y_prob) if y_prob is not None else np.nan

    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    cm = confusion_matrix(y, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[0], cmap="Blues")
    axes[0].set_title(f"Matriz Confusión — {nombre}")
    axes[0].set_ylabel("Real"); axes[0].set_xlabel("Predicción")

    if y_prob is not None and guardar_roc:
        fpr, tpr, _ = roc_curve(y, y_prob)
        axes[1].plot(fpr, tpr, label=f"AUC={auc:.3f}")
        axes[1].plot([0,1],[0,1],"--", color="gray")
        axes[1].set_title(f"Curva ROC — {nombre}")
        axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
        axes[1].legend()
    else:
        axes[1].axis("off")

    fig.tight_layout()
    guardar(fig, f"{prefijo_fig}_cm_roc.png")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}

# --- División con etiquetas re-evaluadas ---
X_tr_r, X_v_r, X_te_r, y_tr_r, y_v_r, y_te_r = dividir(X_raw, y_relabeled)

# --- División con etiquetas originales ---
X_tr_o, X_v_o, X_te_o, y_tr_o, y_v_o, y_te_o = dividir(X_raw, y_original)

print(f"\nTrain={len(X_tr_r)} | Val={len(X_v_r)} | Test={len(X_te_r)}")

# ---- Árbol de Decisión ----
print("\n--- Árbol de Decisión (re-etiquetado) ---")
dt_pipe = construir_pipeline(DecisionTreeClassifier(
    max_depth=6, min_samples_leaf=5, random_state=random_state
))
dt_pipe.fit(X_tr_r, y_tr_r)
print("  [Train]"); res_dt_train_r = evaluar(dt_pipe, X_tr_r, y_tr_r, "DT Train (relabeled)", "09_dt_train_relabeled")
print("  [Val]");   res_dt_val_r   = evaluar(dt_pipe, X_v_r,  y_v_r,  "DT Val (relabeled)",   "10_dt_val_relabeled")
print("  [Test]");  res_dt_test_r  = evaluar(dt_pipe, X_te_r, y_te_r, "DT Test (relabeled)",  "11_dt_test_relabeled")

# Visualizar el árbol
feature_names_out = (
    list(num_cols) +
    dt_pipe.named_steps["preprocessor"]
           .named_transformers_["cat"]
           .named_steps["ohe"]
           .get_feature_names_out(cat_cols).tolist()
)
fig_tree, ax_tree = plt.subplots(figsize=(22, 10))
plot_tree(
    dt_pipe.named_steps["classifier"],
    feature_names=feature_names_out,
    class_names=["No Preparado", "Preparado"],
    filled=True, max_depth=3, ax=ax_tree
)
ax_tree.set_title("Árbol de Decisión (re-etiquetado)")
guardar(fig_tree, "12_arbol_decision_relabeled.png")

# ---- Regresión Logística ----
print("\n--- Regresión Logística (re-etiquetado) ---")
lr_pipe = construir_pipeline(LogisticRegression(
    max_iter=1000, random_state=random_state, C=1.0
))
lr_pipe.fit(X_tr_r, y_tr_r)
print("  [Train]"); res_lr_train_r = evaluar(lr_pipe, X_tr_r, y_tr_r, "LR Train (relabeled)", "13_lr_train_relabeled")
print("  [Val]");   res_lr_val_r   = evaluar(lr_pipe, X_v_r,  y_v_r,  "LR Val (relabeled)",   "14_lr_val_relabeled")
print("  [Test]");  res_lr_test_r  = evaluar(lr_pipe, X_te_r, y_te_r, "LR Test (relabeled)",  "15_lr_test_relabeled")

# ---- Regresión Lineal (como clasificador umbral 0.5) ----
print("\n--- Regresión Lineal → Clasificación por umbral 0.5 (re-etiquetado) ---")

lin_pipe_base = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])),
    ("regressor", LinearRegression())
])

lin_pipe_base.fit(X_tr_r, y_tr_r)

def evaluar_lineal(pipeline, X, y, nombre, prefijo_fig=""):
    y_cont = pipeline.predict(X)
    y_pred = (y_cont >= 0.5).astype(int)
    # Para AUC usamos el valor continuo
    auc = roc_auc_score(y, y_cont) if len(np.unique(y)) == 2 else np.nan

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    mse  = mean_squared_error(y, y_cont)
    r2   = r2_score(y, y_cont)
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}  MSE={mse:.4f}  R²={r2:.4f}")

    cm = confusion_matrix(y, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[0], cmap="Blues")
    axes[0].set_title(f"Matriz Confusión — {nombre}")
    axes[0].set_ylabel("Real"); axes[0].set_xlabel("Predicción")

    fpr, tpr, _ = roc_curve(y, y_cont)
    axes[1].plot(fpr, tpr, label=f"AUC={auc:.3f}")
    axes[1].plot([0,1],[0,1],"--", color="gray")
    axes[1].set_title(f"Curva ROC — {nombre}")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].legend()
    fig.tight_layout()
    guardar(fig, f"{prefijo_fig}_cm_roc.png")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc, "MSE": mse, "R²": r2}

print("  [Train]"); res_lin_train_r = evaluar_lineal(lin_pipe_base, X_tr_r, y_tr_r, "LinReg Train (relabeled)", "16_linreg_train_relabeled")
print("  [Val]");   res_lin_val_r   = evaluar_lineal(lin_pipe_base, X_v_r,  y_v_r,  "LinReg Val (relabeled)",   "17_linreg_val_relabeled")
print("  [Test]");  res_lin_test_r  = evaluar_lineal(lin_pipe_base, X_te_r, y_te_r, "LinReg Test (relabeled)",  "18_linreg_test_relabeled")

# ==============================================================
# PARTE 4 — COMPARACIÓN: DATASET ORIGINAL vs RE-ETIQUETADO
# ==============================================================

print("\n" + "="*60)
print("PARTE 4 — COMPARACIÓN: ORIGINAL vs RE-ETIQUETADO")
print("="*60)

# Modelos sobre dataset ORIGINAL
print("\n--- Árbol de Decisión (original) ---")
dt_orig = construir_pipeline(DecisionTreeClassifier(
    max_depth=6, min_samples_leaf=5, random_state=random_state
))
dt_orig.fit(X_tr_o, y_tr_o)
print("  [Test]"); res_dt_test_o = evaluar(dt_orig, X_te_o, y_te_o, "DT Test (original)", "19_dt_test_original")

print("\n--- Regresión Logística (original) ---")
lr_orig = construir_pipeline(LogisticRegression(
    max_iter=1000, random_state=random_state, C=1.0
))
lr_orig.fit(X_tr_o, y_tr_o)
print("  [Test]"); res_lr_test_o = evaluar(lr_orig, X_te_o, y_te_o, "LR Test (original)", "20_lr_test_original")

print("\n--- Regresión Lineal (original) ---")
lin_orig = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])),
    ("regressor", LinearRegression())
])
lin_orig.fit(X_tr_o, y_tr_o)
print("  [Test]"); res_lin_test_o = evaluar_lineal(lin_orig, X_te_o, y_te_o, "LinReg Test (original)", "21_linreg_test_original")

# --- Tabla comparativa final ---
print("\n--- Tabla comparativa (Test set) ---")

df_comp = pd.DataFrame({
    "DT — Original":    res_dt_test_o,
    "DT — Relabeled":   res_dt_test_r,
    "LR — Original":    res_lr_test_o,
    "LR — Relabeled":   res_lr_test_r,
    "LinReg — Original": res_lin_test_o,
    "LinReg — Relabeled": res_lin_test_r,
}).T

print(df_comp.to_string())
df_comp.to_csv(OUTDIR / "comparacion_modelos.csv")
print(f"\n[guardado] {OUTDIR / 'comparacion_modelos.csv'}")

# Gráfico comparativo de métricas clave
metricas_plot = ["Accuracy", "F1", "AUC"]
fig_comp, axes_comp = plt.subplots(1, len(metricas_plot), figsize=(15, 5))

colores_comp = ["#2ecc71", "#27ae60", "#3498db", "#2980b9", "#e67e22", "#d35400"]

for ax_c, metrica in zip(axes_comp, metricas_plot):
    vals = df_comp[metrica].dropna()
    bars = ax_c.bar(range(len(vals)), vals.values, color=colores_comp[:len(vals)])
    ax_c.set_xticks(range(len(vals)))
    ax_c.set_xticklabels(vals.index, rotation=30, ha="right", fontsize=8)
    ax_c.set_ylim(0, 1)
    ax_c.set_title(f"Comparación — {metrica}")
    ax_c.set_ylabel(metrica)
    for bar, val in zip(bars, vals.values):
        ax_c.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
                  ha="center", va="bottom", fontsize=8)

fig_comp.tight_layout()
guardar(fig_comp, "22_comparacion_modelos.png")

# ==============================================================
# BONUS — Importancia de features (Árbol sobre re-etiquetado)
# ==============================================================

print("\n[BONUS] Importancia de features — Árbol de Decisión (re-etiquetado)...")

importancias = dt_pipe.named_steps["classifier"].feature_importances_
df_imp = pd.DataFrame({
    "feature":    feature_names_out,
    "importance": importancias
}).sort_values("importance", ascending=False).head(15)

fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
ax_imp.barh(df_imp["feature"][::-1], df_imp["importance"][::-1], color="steelblue")
ax_imp.set_title("Top 15 — Importancia de Features (Árbol, re-etiquetado)")
ax_imp.set_xlabel("Importancia")
fig_imp.tight_layout()
guardar(fig_imp, "23_importancia_features_dt.png")

# ==============================================================
# RESUMEN FINAL
# ==============================================================

print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)
print(f"  Dataset: 300 muestras, {X_raw.shape[1]} features")
print(f"  Muestras re-etiquetadas: {n_cambios} ({n_cambios/len(y_original):.1%})")
print(f"\n  Clustering — Silhouettes:")
for k, v in df_sil.dropna().iterrows():
    print(f"    {k:35s}: {v['Silhouette']:.4f}")
print(f"\n  Test set — Comparación de modelos:")
print(df_comp[["Accuracy","F1","AUC"]].to_string())
print(f"\n  Gráficas guardadas en: {OUTDIR.resolve()}")
print("\n[FIN]")
