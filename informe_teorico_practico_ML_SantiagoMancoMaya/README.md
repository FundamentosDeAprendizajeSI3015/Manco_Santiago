# 📊 Proyecto: Impacto del Uso de IA en la Preparación Laboral

## 🎯 Objetivo

Este proyecto implementa un **pipeline completo de ciencia de datos** para analizar cómo el uso de herramientas de Inteligencia Artificial influye en la preparación laboral de estudiantes de Ingeniería de Sistemas.

El modelo busca predecir:

> **¿Un estudiante está laboralmente preparado? (0 = No, 1 = Sí)**

Se trata de un problema de:

* ✅ Clasificación binaria
* ✅ Aprendizaje supervisado
* ✅ Dataset estructurado

---

# 1️⃣ Definición del Problema

Se define formalmente el problema en un archivo:

```
data_output_educacion_ia/definicion_problema.json
```

Contiene:

* Objetivo
* Impacto
* Tipo de problema
* Variables numéricas
* Variables categóricas
* Variables binarias

### Variables utilizadas

#### Variables Numéricas

* promedio_acumulado
* nota_algoritmos
* nota_bases_datos
* horas_estudio_semana

#### Variables Categóricas

* frecuencia_uso_ia
* dependencia_ia
* aprendizaje_activo

#### Variables Binarias

* uso_para_codigo
* uso_para_teoria
* proyectos_personales

---

# 2️⃣ Carga y Recolección de Datos

El dataset:

```python
dataset_ingenieria_sistemas_ia_300_realista.csv
```

Se analiza:

* Dimensión del dataset
* Tipos de datos
* Valores nulos
* Distribución del target

Esto permite verificar:

* Calidad de los datos
* Balance de clases
* Posibles problemas estructurales

---

# 3️⃣ Análisis Exploratorio de Datos (EDA)

Se realiza un análisis estadístico completo:

---

## Tendencia Central

Para variables numéricas:

* Media
* Mediana
* Moda

Archivo generado:

```
tendencia_central_numericas.csv
```

Para variables binarias:

```
tendencia_central_binarias.csv
```

Para categóricas:

```
moda_categoricas.json
```

---

## Cuartiles e IQR

Se calcula:

* Q1
* Q3
* IQR (Rango Intercuartílico)

Archivo:

```
iqr_results.json
```

---

## Percentiles

Se calculan:

* P10
* P50
* P90

Archivo:

```
percentiles.json
```

---

## Correlaciones

Se genera:

* Matriz de correlación
* Heatmap visual
* Correlación Pearson
* Correlación Spearman

Archivo generado:

```
heatmap_correlacion.png
correlation_stats.json
```

Esto permite entender:

* Qué variables impactan más el target
* Relaciones lineales vs monotónicas

---

## Tabla Dinámica (Pivot Table)

Se analiza el promedio acumulado según frecuencia de uso de IA y preparación laboral.

Archivo:

```
pivot_promedio_por_frecuencia.csv
```

---

## Visualizaciones Interactivas

Se generan gráficos en HTML interactivos:

### Scatter Matrix

```
interactive_scatter_matrix.html
```

### Gráfico 3D

```
interactive_scatter_3d.html
```

### UMAP 2D

```
interactive_umap_2d.html
```

### UMAP 3D

```
interactive_umap_3d.html
```

UMAP permite visualizar separabilidad entre clases en espacios reducidos.

---

# 4️⃣ Procesamiento de Datos

Se realiza:

### Limpieza

* Conversión segura a numérico
* Imputación con mediana (numéricas)
* Relleno "**MISSING**" (categóricas)

### One-Hot Encoding

Se aplica:

```python
pd.get_dummies(..., drop_first=True)
```

Para evitar multicolinealidad (Dummy Trap).

---

# 5️⃣ División del Dataset

Se aplica:

```
70% Train
15% Validation
15% Test
```

Con:

```python
stratify=y
```

Esto garantiza que la proporción de clases se mantenga en todos los splits.

---

# ⚖️ Balanceo de Clases (Solo Train)

Se utiliza:

```python
resample()
```

* Se iguala el tamaño de la clase minoritaria
* Se evita que el modelo se sesgue

Importante:
El balanceo **solo se aplica en entrenamiento**, nunca en validación o test.

---

# 📏 Escalado

Se usa:

```python
StandardScaler()
```

* Fit en train
* Transform en val y test

Esto evita data leakage.

---

# 6️⃣ Exportación Final

Se exportan:

```
X_train.parquet
X_val.parquet
X_test.parquet

y_train.parquet
y_val.parquet
y_test.parquet
```

Además:

```
processed_schema.json
```

Contiene:

* Proporción de split
* Balance final de clases

---

# 📂 Estructura de Carpetas

```
data_output_educacion_ia/
│
├── definicion_problema.json
├── tendencia_central_numericas.csv
├── tendencia_central_binarias.csv
├── moda_categoricas.json
├── iqr_results.json
├── percentiles.json
├── correlation_stats.json
├── heatmap_correlacion.png
├── pivot_promedio_por_frecuencia.csv
├── interactive_scatter_matrix.html
├── interactive_scatter_3d.html
├── interactive_umap_2d.html
├── interactive_umap_3d.html
├── X_train.parquet
├── X_val.parquet
├── X_test.parquet
├── y_train.parquet
├── y_val.parquet
├── y_test.parquet
└── processed_schema.json
```

---

# Buenas Prácticas Implementadas

✔ Separación clara de fases
✔ No hay data leakage
✔ Balanceo solo en entrenamiento
✔ Estratificación en split
✔ Escalado correcto
✔ Exportación reproducible
✔ EDA documentado
✔ Visualizaciones interactivas
