# Lecture 09 — Clustering

## Descripción

Ejercicio de aprendizaje no supervisado con algoritmos de clustering aplicados al dataset sintético FIRE-UdeA.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `lecture_09.py` | Script base — KMeans y DBSCAN con PCA |
| `lecture_09_realista.py` | Variante con el dataset realista |
| `dataset_sintetico_FIRE_UdeA.csv` | Dataset sintético original |
| `dataset_sintetico_FIRE_UdeA_realista.csv` | Dataset sintético con valores más realistas |

## Algoritmos aplicados

- **KMeans** — con método del codo y silhouette para selección de k
- **DBSCAN** — clustering basado en densidad
- **PCA** — reducción de dimensionalidad para visualización

## Cómo ejecutar

```bash
python lecture_09.py
# o la variante realista:
python lecture_09_realista.py
```
