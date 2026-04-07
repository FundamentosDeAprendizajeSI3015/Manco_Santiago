# Output — Informe 02

Resultados generados por `analisis_completo.py`.

## Gráficas de clustering

| Archivo | Descripción |
|---------|-------------|
| `01_kmeans_codo_silhouette.png` | Método del codo y silhouette para KMeans |
| `02_kmeans_clusters_pca.png` | Clusters KMeans proyectados con PCA |
| `03_fuzzy_cmeans_pca.png` | Clusters Fuzzy C-Means con PCA |
| `04_subtractive_clustering_pca.png` | Subtractive Clustering con PCA |
| `05_dbscan_pca.png` | Clusters DBSCAN con PCA |
| `06_familia_clustering_pca.png` | Agglomerative Clustering con PCA |
| `07_comparacion_clustering.png` | Comparación de todos los métodos |

## Re-etiquetado y modelos supervisados

| Archivo | Descripción |
|---------|-------------|
| `08_reetiquetado.png` | Visualización del proceso de re-etiquetado |
| `09–11_dt_*_relabeled_cm_roc.png` | Árbol de Decisión (train/val/test) con etiquetas corregidas |
| `12_arbol_decision_relabeled.png` | Árbol de decisión graficado |
| `13–15_lr_*_relabeled_cm_roc.png` | Regresión Logística (train/val/test) con etiquetas corregidas |
| `16–18_linreg_*_relabeled_cm_roc.png` | Regresión Lineal (train/val/test) con etiquetas corregidas |
| `19–21_*_test_original_cm_roc.png` | Modelos sobre dataset original (comparación) |
| `22_comparacion_modelos.png` | Comparación global de modelos |
| `23_importancia_features_dt.png` | Importancia de variables — Árbol de Decisión |
| `comparacion_modelos.csv` | Métricas numéricas comparativas |
