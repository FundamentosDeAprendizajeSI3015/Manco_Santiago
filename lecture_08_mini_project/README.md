# Lecture 08 — Mini Proyecto: Financial UdeA Dashboard

## Descripción

Mini proyecto integrador que combina un pipeline de Machine Learning en Python con un dashboard interactivo web para visualizar predicciones y métricas financieras sobre datos sintéticos del programa FIRE-UdeA.

## Estructura

```
lecture_08_mini_project/
├── modelo_aprendizaje.py                    # Pipeline ML en Python
├── dataset_sintetico_FIRE_UdeA_realista.csv # Dataset de entrada
└── financial_udea_dashboard/                # Aplicación web (Next.js + Django)
```

## Componentes

### Backend — Django (`financial_udea_dashboard/backend/`)
API REST que sirve predicciones del modelo ML y expone métricas.

### Frontend — Next.js (`financial_udea_dashboard/`)
Dashboard interactivo con visualizaciones de:
- Métricas del modelo
- Predicciones por estudiante
- Comparaciones de datasets

## Cómo ejecutar

**ML Pipeline:**
```bash
python modelo_aprendizaje.py
```

**Dashboard (frontend):**
```bash
cd financial_udea_dashboard
pnpm install
pnpm dev
```

**Backend (Django):**
```bash
cd financial_udea_dashboard/backend
pip install -r requirements.txt
python manage.py runserver
```
