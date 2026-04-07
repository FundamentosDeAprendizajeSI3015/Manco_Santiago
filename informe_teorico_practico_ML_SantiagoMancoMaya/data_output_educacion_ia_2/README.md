# data_output_educacion_ia_2 — Salida Modelado Supervisado

Artefactos generados por `informe_educacion_ia_modelo.py` (fase de entrenamiento y evaluación del modelo de regresión logística).

## Archivos del modelo

| Archivo | Descripción |
|---------|-------------|
| `logistic_regression_metrics.json` | Métricas de evaluación (accuracy, F1, AUC, etc.) |
| `logistic_regression_coefficients.csv` | Coeficientes del modelo por variable |

## Visualizaciones del modelo

| Archivo | Descripción |
|---------|-------------|
| `logistic_coefficients.png` | Gráfica de coeficientes del modelo |
| `confusion_matrix_test.png` | Matriz de confusión en test |
| `roc_curve_test.png` | Curva ROC en test |
| `precision_recall_curve_test.png` | Curva Precisión-Recall en test |

## Splits del dataset (heredados del EDA)

| Archivo | Descripción |
|---------|-------------|
| `X_train.parquet` / `y_train.parquet` | Conjunto de entrenamiento |
| `X_val.parquet` / `y_val.parquet` | Conjunto de validación |
| `X_test.parquet` / `y_test.parquet` | Conjunto de prueba |

## Visualizaciones EDA

| Archivo | Descripción |
|---------|-------------|
| `heatmap_correlacion.png` | Mapa de calor de correlaciones |
| `interactive_scatter_3d.html` | Scatter 3D interactivo |
| `interactive_umap_2d.html` / `interactive_umap_3d.html` | Proyecciones UMAP interactivas |
