# Financial UdeA Dashboard

Dashboard web interactivo para visualizar predicciones y métricas del modelo ML aplicado al dataset sintético FIRE-UdeA.

## Stack tecnológico

- **Frontend:** Next.js 14, TypeScript, Tailwind CSS
- **Backend:** Django (REST API), SQLite
- **ML:** scikit-learn (entrenado en `../modelo_aprendizaje.py`)

## Estructura

```
financial_udea_dashboard/
├── app/              # Rutas y páginas (Next.js App Router)
│   ├── api/          # API routes del frontend
│   ├── comparison/   # Página de comparación de datasets
│   ├── metrics/      # Página de métricas del modelo
│   ├── predictions/  # Página de predicciones
│   └── upload/       # Carga de nuevos datasets
├── backend/          # API Django
│   ├── fire_udea/    # App principal Django
│   ├── ml_api/       # Endpoints del modelo ML
│   └── ml_models/    # Modelos entrenados serializados
├── components/       # Componentes React reutilizables
├── hooks/            # Custom React hooks
├── lib/              # Utilidades y configuración
└── public/           # Activos estáticos
```

## Cómo ejecutar

```bash
pnpm install
pnpm dev
```

La aplicación estará disponible en `http://localhost:3000`.
