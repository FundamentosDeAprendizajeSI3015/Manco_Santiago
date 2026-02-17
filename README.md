# 📘 Fundamentos de Aprendizaje Automático – SI3015

Repositorio de trabajos prácticos – Santiago Manco

Este repositorio contiene el desarrollo progresivo del ciclo de vida de Machine Learning a lo largo de varias semanas, incluyendo:

* Definición del problema
* Análisis exploratorio de datos (EDA)
* Limpieza y preprocesamiento
* Ingeniería de características
* Partición de datos
* Exportación para modelado
* Informe 1 Teórico Práctico

---

# 📅 Semana 2 – Ciclo de Vida ML con Iris

📂 Archivo: `iris_lifecycle.py`

## 🎯 Objetivo

Implementar el ciclo completo de Machine Learning utilizando el dataset clásico **Iris**.

## 🔎 Problema

Clasificación supervisada multiclase para predecir la especie de flor:

* Setosa
* Versicolor
* Virginica

## 🧠 Etapas implementadas

### 1️⃣ Definición del problema

Clasificación multiclase con variable objetivo `species`.

### 2️⃣ Recolección de datos

Se usa el dataset Iris desde `sklearn.datasets`.

### 3️⃣ Procesamiento

* Validación de valores nulos
* Normalización con `StandardScaler`
* División Train/Test (75% / 25%) con estratificación

### 4️⃣ Entrenamiento

Modelo:

* **SVM (Support Vector Machine)** con kernel RBF
* Implementado mediante `Pipeline`

### 5️⃣ Evaluación

Métricas:

* Accuracy
* Precision
* Recall
* F1-score
* Matriz de confusión
* Classification report

📌 Resultado: Se implementa correctamente un pipeline profesional de ML desde cero.

---

# 📅 Semana 3 – Laboratorio FinTech Sintético (EDA + Preprocesamiento)

📂 Archivo: `lab_fintech_sintetico_2025.py`

## 🎯 Objetivo

Realizar un análisis exploratorio completo y preparar datos financieros sintéticos para modelado futuro.

Dataset 100% sintético con fines académicos.

## 🧠 Etapas implementadas

### 0️⃣ Carga y validación del diccionario

* Validación del JSON de metadatos

### 1️⃣ Carga del CSV

* Parsing de fechas
* Ordenamiento temporal

### 2️⃣ EDA básico

* Info del dataset
* Análisis de nulos

### 2.5️⃣ EDA visual interactivo

Se generan archivos HTML con:

* Scatter Matrix
* Coordenadas paralelas
* Scatter 3D
* UMAP 2D
* UMAP 3D

Todos exportados en:

```
data_output_finanzas_sintetico/
```

### 3️⃣ Limpieza

* Imputación:

  * Numéricas → mediana
  * Categóricas → `"__MISSING__"`

### 4️⃣ Ingeniería de características

* Retornos porcentuales
* Log-retornos de precio
* Agrupación por empresa y fecha

### 5️⃣ Preparación para ML

* Eliminación de IDs y fecha
* One-hot encoding
* Escalado
* Split temporal (evita fuga de datos)

### 6️⃣ Exportación

Se generan:

* `fintech_train.parquet`
* `fintech_test.parquet`
* `processed_schema.json`
* `features_columns.txt`

📌 Resultado: Pipeline robusto de preprocesamiento financiero listo para modelado.

---

# 📅 Semana 4 – Impacto del Uso de IA en la Preparación Laboral

📂 Script principal: procesamiento del dataset educativo IA

## 🎯 Objetivo

Analizar cómo el uso de Inteligencia Artificial influye en la preparación laboral de estudiantes.

Problema:
Clasificación supervisada binaria (`preparacion_laboral`).

## 🧠 Etapas implementadas

### 1️⃣ Carga del dataset

### 2️⃣ EDA básico

* Distribución de clases
* Revisión de nulos
* Info estructural

### 2.2️⃣ Medidas de tendencia central

Para:

* Variables numéricas (media, mediana, moda)
* Variables categóricas (moda)
* Variables binarias (media, moda)

Resultados exportados en:

```
data_output_educacion_ia/
```

### 2.5️⃣ Visualización interactiva

Se generan:

* Scatter Matrix
* Coordenadas paralelas
* Scatter 3D
* UMAP 2D
* UMAP 3D

### 3️⃣ Limpieza

* Conversión numérica
* Imputación con mediana
* Manejo de categóricas faltantes

### 4️⃣ Preparación X / y

* One-hot encoding

### 5️⃣ Split profesional

Train / Validation / Test:

* 60% Train
* 20% Validation
* 20% Test
  Con estratificación.

### 6️⃣ Escalado

Entrenado solo con TRAIN (evita data leakage).

### 7️⃣ Exportación

Se generan:

* X_train, X_val, X_test
* y_train, y_val, y_test
* processed_schema.json
* features_columns.txt

📌 Resultado: Pipeline académico completo con validación adecuada.

---

# 📅 Semana 5 – Informe 1 del Proyecto de Aprendizaje

📂 Carpeta: informe_teorico_practico_ML_SantiagoMancoMaya

El informe consolida todo el trabajo realizado en las semanas anteriores y formaliza el desarrollo del proyecto bajo estándares académicos.

---

# 🛠 Tecnologías Utilizadas

* Python
* NumPy
* Pandas
* Scikit-learn
* Plotly
* UMAP
* JSON / Parquet
