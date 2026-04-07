# Lecture 10 — Clustering Avanzado

## Descripción

Extensión de la lecture 09 con análisis de clustering más avanzado, incorporando visualización 3D, métricas adicionales (Adjusted Rand Score) y una variante de clustering sustractivo.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `lecture_10.py` | Script base — KMeans, DBSCAN, PCA, visualización 3D |
| `lecture_10_realista.py` | Variante con dataset realista e imputación de valores nulos |
| `lecture_10_substractive.py` | Implementación de clustering sustractivo |
| `dataset_sintetico_FIRE_UdeA.csv` | Dataset sintético original |
| `dataset_sintetico_FIRE_UdeA_realista.csv` | Dataset sintético realista |
| `errores_por_unidad.csv` | Análisis de errores por unidad |
| `resultado_clustering_completo.csv` | Resultados finales del clustering |

## Mejoras respecto a Lecture 09

- Visualización de clusters en 3D (`mpl_toolkits.mplot3d`)
- Métrica **Adjusted Rand Score** para evaluación externa
- Imputación de valores faltantes (`SimpleImputer`)
- Clustering sustractivo como método alternativo

## Cómo ejecutar

```bash
python lecture_10.py
# variante realista:
python lecture_10_realista.py
# clustering sustractivo:
python lecture_10_substractive.py
```
