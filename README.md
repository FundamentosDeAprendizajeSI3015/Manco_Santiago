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

📂 Archivo: `manco_santiago_iris_analysis.py`

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

📂 Archivo: `lect_03_manco_santiago_lab_fintech_sintetico_2025.py`

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

📂 Archivo: `lecture4_EDA.py`

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

# 📅 Semana 5 – Regresión Lineal y Regresión Logística con Movies Dataset

📂 Archivo: `lecture05_movies_exercise_santiago_manco.py`

## 🎯 Objetivo

Aplicar modelos de **regresión supervisada** para analizar el rendimiento económico de películas utilizando el dataset **Movies**.

Se implementan dos enfoques:

* **Regresión lineal regularizada** para predecir ingresos (`Gross`)
* **Regresión logística** para clasificar películas según su nivel de ingresos

---

## 🧠 Etapas implementadas

### 1️⃣ Carga y limpieza de datos

Se realiza una limpieza exhaustiva del dataset:

* Extracción del año numérico desde la columna `YEAR`
* Conversión de `RunTime` a minutos
* Eliminación de comas en `VOTES`
* Limpieza del símbolo `$` y caracteres en `Gross`
* Conversión de `RATING` a tipo numérico
* Eliminación de registros con valores nulos

---

### 2️⃣ Regresión Lineal

Problema de **predicción de ingresos de película** (`Gross`).

Variables utilizadas:

* `RunTime`
* `RATING`
* `VOTES`
* `YEAR`

Proceso implementado:

* División **Train/Test (80/20)**
* Visualización de distribución **Train vs Test**
* Creación de pipelines con:

  * `PolynomialFeatures`
  * `StandardScaler`
  * Modelos **Ridge** y **Lasso**

Optimización de hiperparámetros mediante:

* **RandomizedSearchCV**
* Validación cruzada (`cv=4`)

Parámetros optimizados:

* grado polinomial
* coeficiente de regularización (`alpha`)

---

### 3️⃣ Evaluación de modelos

Se evalúan los modelos mediante:

* **R²**
* **MAE (Mean Absolute Error)**

Se generan visualizaciones de:

* Predicción vs valor real para **Ridge**
* Predicción vs valor real para **Lasso**

Las gráficas se exportan en:

```
output/
```

---

### 4️⃣ Regresión Logística

Se construye un problema de **clasificación binaria**.

Se crea una nueva variable:

```
High_Gross
```

Definida como:

* `1` → película con ingresos mayores a la **mediana**
* `0` → película con ingresos menores o iguales a la mediana

Pipeline implementado:

* `PolynomialFeatures`
* `StandardScaler`
* `LogisticRegression`

Optimización mediante:

* **RandomizedSearchCV**
* Validación cruzada

---

### 5️⃣ Evaluación del modelo de clasificación

Métricas utilizadas:

* Accuracy
* F1-score
* Matriz de confusión

Se genera una visualización de la **Confusion Matrix**.

---

📌 Resultado:
Se implementa un flujo completo de **regresión y clasificación supervisada**, incluyendo **limpieza avanzada de datos, pipelines, regularización y optimización de hiperparámetros**.

---

# 📅 Semana 6 – Clasificación con Random Forest y Gradient Boosting

📂 Archivo: `modelo_aprendizaje.py`

## 🎯 Objetivo

Comparar el desempeño de modelos de **ensamble basados en árboles** para predecir si un estudiante está **preparado laboralmente** usando un dataset académico realista.

Variable objetivo:

```
preparacion_laboral
```

* `0` → No preparado
* `1` → Preparado

---

## 🧠 Etapas implementadas

### 1️⃣ Carga y exploración de datos

Se analiza el dataset:

* Visualización de primeras filas
* Revisión de estructura (`info`)
* Distribución del target

Se generan visualizaciones:

* Histogramas de variables
* Matriz de correlación con **Seaborn**

---

### 2️⃣ Limpieza de datos

Procesos aplicados:

* Eliminación de registros duplicados
* Revisión de valores nulos
* Imputación de valores numéricos con **mediana**

---

### 3️⃣ Definición de variables

Se separan:

* **Features (X)**
* **Target (y)**

Se identifican automáticamente:

* variables **numéricas**
* variables **categóricas**

---

### 4️⃣ División del dataset

Se aplica una división **estratificada**:

* **60% Train**
* **20% Validation**
* **20% Test**

Esto permite evaluar el modelo sin fuga de información.

---

### 5️⃣ Pipeline de preprocesamiento

Se utiliza `ColumnTransformer` para aplicar transformaciones específicas:

Variables numéricas:

* `StandardScaler`

Variables categóricas:

* `OneHotEncoder`

Esto permite integrar todo el flujo dentro de un **pipeline reproducible**.

---

### 6️⃣ Modelos implementados

Se entrenan dos algoritmos de ensamble:

#### 🌳 Random Forest

* `n_estimators = 200`
* Reduce varianza mediante múltiples árboles.

#### 🚀 Gradient Boosting

* `n_estimators = 200`
* Construye árboles secuenciales corrigiendo errores previos.

---

### 7️⃣ Interpretabilidad de modelos

Para comprender el funcionamiento de los modelos se grafican:

* Un árbol individual del **Random Forest**
* Un árbol del **Gradient Boosting**

Esto permite visualizar:

* divisiones
* variables utilizadas
* decisiones del modelo

---

### 8️⃣ Evaluación de modelos

Se evalúan los modelos en:

* **Train**
* **Validation**
* **Test**

Métricas utilizadas:

* Accuracy
* Precision
* Recall
* F1-score
* AUC

También se generan:

* Matrices de confusión
* Curvas ROC

---

📌 Resultado:
Se implementa una comparación completa entre **Random Forest y Gradient Boosting**, incluyendo **preprocesamiento automático, pipelines, evaluación robusta y visualización de árboles de decisión**.

---

# 🛠 Tecnologías Utilizadas

* Python
* NumPy
* Pandas
* Scikit-learn
* Plotly
* UMAP
* JSON / Parquet
