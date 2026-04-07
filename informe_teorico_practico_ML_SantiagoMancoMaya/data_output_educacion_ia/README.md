# data_output_educacion_ia — Salida EDA y Preprocesamiento

Artefactos generados por `informe_educacion_ia.py` (fase de exploración y preprocesamiento).

## Archivos estadísticos

| Archivo | Descripción |
|---------|-------------|
| `definicion_problema.json` | Definición formal del problema ML |
| `processed_schema.json` | Esquema del dataset preprocesado |
| `correlation_stats.json` | Estadísticas de correlación entre variables |
| `iqr_results.json` | Resultados de análisis de rango intercuartílico |
| `moda_categoricas.json` | Moda de variables categóricas |
| `percentiles.json` | Percentiles de variables numéricas |
| `tendencia_central_numericas.csv` | Media, mediana, desviación de variables numéricas |
| `tendencia_central_binarias.csv` | Tendencia central de variables binarias |
| `pivot_promedio_por_frecuencia.csv` | Tabla pivote de promedios por frecuencia |

## Splits del dataset

| Archivo | Descripción |
|---------|-------------|
| `X_train.parquet` / `y_train.parquet` | Conjunto de entrenamiento |
| `X_val.parquet` / `y_val.parquet` | Conjunto de validación |
| `X_test.parquet` / `y_test.parquet` | Conjunto de prueba |

## Visualizaciones

| Archivo | Descripción |
|---------|-------------|
| `heatmap_correlacion.png` | Mapa de calor de correlaciones |
| `interactive_scatter_3d.html` | Scatter 3D interactivo |
| `interactive_scatter_matrix.html` | Matriz de scatter interactiva |
| `interactive_umap_2d.html` | Proyección UMAP 2D interactiva |
| `interactive_umap_3d.html` | Proyección UMAP 3D interactiva |
