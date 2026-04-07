# Informe Teórico-Práctico 02 — Machine Learning

**Autor:** Santiago Manco Maya

## Descripción

Informe integrador que combina aprendizaje no supervisado y supervisado sobre el dataset de ingeniería de sistemas con IA, cuya variable objetivo es `preparacion_laboral` (0 = No preparado, 1 = Preparado).

## Estructura

```
informe_teorico_practico_02_ML_SantiagoManco/
├── analisis_completo.py                              # Script principal
├── dataset_ingenieria_sistemas_ia_300_realista.csv   # Dataset (300 registros)
└── output/                                           # Gráficas y métricas generadas
```

## Contenido del análisis

| Parte | Descripción |
|-------|-------------|
| 1 | **Clustering no supervisado** — KMeans, Fuzzy C-Means, Subtractive Clustering, DBSCAN, Agglomerative Clustering |
| 2 | **Re-evaluación de etiquetas** — detección y corrección de ~30% de etiquetas potencialmente incorrectas |
| 3 | **Modelos supervisados** con etiquetas re-evaluadas — Árbol de Decisión, Regresión Logística, Regresión Lineal |
| 4 | **Comparación** entre dataset original y re-etiquetado |

## Cómo ejecutar

```bash
python analisis_completo.py
```

Las figuras y archivos de resultados se guardan en `output/`.
