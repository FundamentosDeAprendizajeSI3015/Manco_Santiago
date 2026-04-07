# Lecture 06 — Ejercicio de Modelado Supervisado

## Descripción

Ejercicio de clasificación binaria comparando **Random Forest** vs **Gradient Boosting** sobre el dataset de ingeniería de sistemas con IA.

- **Variable objetivo:** `preparacion_laboral` (0 = No preparado, 1 = Preparado)
- **División:** 60% Train / 20% Val / 20% Test (estratificado)

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `modelo_aprendizaje.py` | Script principal — entrenamiento y evaluación |
| `dataset_ingenieria_sistemas_ia_300_realista.csv` | Dataset de entrada (300 registros) |

## Cómo ejecutar

```bash
python modelo_aprendizaje.py
```
