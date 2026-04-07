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

---

# 📅 Semana 8 – Mini Proyecto: Clasificación Financiera FIRE_UdeA con GridSearchCV

📂 Carpeta: `lecture_08_mini_project/`

📂 Archivos: `modelo_aprendizaje.py`, `financial_udea_dashboard/`

## 🎯 Objetivo

Construir un pipeline profesional completo de clasificación para predecir la **situación financiera** de unidades académicas de la Universidad de Antioquia.

Variable objetivo:

```
label
```

* `0` → Situación financiera estable
* `1` → Situación financiera crítica

---

## 🧠 Etapas implementadas

### 1️⃣ Configuración y carga de datos

Se carga el dataset `dataset_sintetico_FIRE_UdeA_realista.csv` con variables financieras por unidad académica y año.

Variables numéricas:

* `ingresos_totales`, `gastos_personal`, `liquidez`, `dias_efectivo`, `cfo`
* `participacion_ley30`, `participacion_regalias`, `participacion_servicios`, `participacion_matriculas`
* `hhi_fuentes`, `endeudamiento`, `tendencia_ingresos`, `gp_ratio`

Variables categóricas:

* `unidad`, `anio`

### 2️⃣ EDA completo

* Medidas de tendencia central por tipo de variable
* Visualizaciones interactivas (Scatter Matrix, Coordenadas Paralelas, Scatter 3D, UMAP 2D y 3D)
* Histogramas y **Matriz de Correlación**

### 3️⃣ Limpieza de datos

* Eliminación de duplicados
* Imputación de valores nulos (numéricas → mediana, categóricas → `"Unknown"`)

### 4️⃣ División estratificada

* **60% Train / 20% Validation / 20% Test**

### 5️⃣ Pipeline de preprocesamiento

Mediante `ColumnTransformer`:

* Numéricas → `StandardScaler`
* Categóricas → `OneHotEncoder`

### 6️⃣ Modelos implementados

#### 🌳 Random Forest

Optimizado con **GridSearchCV** (`cv=5`):

* `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`

#### 🚀 Gradient Boosting

Optimizado con **GridSearchCV** (`cv=5`):

* `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `subsample`

### 7️⃣ Visualización de árboles

Se grafican árboles individuales de ambos modelos (profundidad = 3) para interpretabilidad.

### 8️⃣ Evaluación completa

En Train, Validation y Test:

* Accuracy, Precision, Recall, F1-score, AUC
* Matrices de confusión y Curvas ROC

### 🌐 Dashboard Interactivo

Se desarrolló un **dashboard financiero web** en:

```
financial_udea_dashboard/
```

📌 Resultado: Pipeline completo con búsqueda exhaustiva de hiperparámetros y dashboard interactivo para análisis financiero universitario.

---

# 📅 Semana 9 – Clustering: KMeans y DBSCAN

📂 Carpeta: `lecture_09_clustering/`

📂 Archivos: `lecture_09.py`, `lecture_09_realista.py`

## 🎯 Objetivo

Aplicar algoritmos de **aprendizaje no supervisado** al dataset FIRE_UdeA para descubrir agrupaciones naturales entre unidades académicas según sus indicadores financieros.

---

## 🧠 Etapas implementadas

### 1️⃣ Carga y separación de variables

Se carga `dataset_sintetico_FIRE_UdeA_realista.csv` y se separa la etiqueta real (`label`) para evaluación posterior.

### 2️⃣ Preprocesamiento

* Numéricas → `SimpleImputer(mean)` + `StandardScaler`
* Categóricas → `SimpleImputer(most_frequent)` + `OneHotEncoder`

### 3️⃣ Reducción de dimensión con PCA

* Proyección 2D para visualización de los datos antes del clustering

### 4️⃣ Selección de k: método del codo + Silhouette

Se prueban valores de `k` entre 2 y 10:

* Gráfica de **inercia** (método del codo)
* Gráfica de **Silhouette Score**
* Se selecciona automáticamente el `k` con mayor silhouette

### 5️⃣ KMeans

* Modelo con `k` óptimo (`n_init=10`)
* Visualización 2D de clusters en espacio PCA

### 6️⃣ DBSCAN

* Clustering basado en densidad (`eps=1.2`, `min_samples=5`)
* Identificación de puntos de ruido (etiqueta `-1`)
* Silhouette calculado excluyendo ruido

### 7️⃣ Comparación

Se comparan KMeans y DBSCAN visualmente sobre las mismas proyecciones PCA.

### 8️⃣ Evaluación contra etiquetas reales

* **Adjusted Rand Index (ARI)** entre clusters KMeans y etiquetas reales

📌 Resultado: Primer acercamiento a clustering no supervisado sobre datos financieros, con evaluación cuantitativa mediante ARI.

---

# 📅 Semana 10 – Clustering Avanzado: Subtractive + Fuzzy C-Means + Análisis de Errores

📂 Carpeta: `lecture_10_clustering/`

📂 Archivos: `lecture_10.py`, `lecture_10_realista.py`, `lecture_10_substractive.py`

## 🎯 Objetivo

Extender el análisis de clustering con algoritmos avanzados (**Subtractive Clustering** y **Fuzzy C-Means**), realizar comparaciones entre cuatro métodos y analizar los errores de clustering por unidad académica.

---

## 🧠 Etapas implementadas

### 1️⃣ Implementación de algoritmos personalizados

Se implementan desde cero dos clases:

#### 🔵 Subtractive Clustering

* Detección automática de número de centros mediante potenciales de densidad
* Parámetros: `ra`, `rb`, `eps_upper`, `eps_lower`
* Normalización interna del espacio de características

#### 🟡 Fuzzy C-Means (FCM)

* Asignación borrosa con grado de pertenencia `μ` para cada punto
* Inicialización opcional con centros del Subtractive Clustering

### 2️⃣ Proyección 2D y 3D con PCA

* Visualización 2D y 3D de los datos crudos antes del clustering

### 3️⃣ KMeans

* Método del codo + Silhouette Score para selección de `k`
* Visualización en 2D y 3D
* Centroides proyectados al espacio PCA 3D

### 4️⃣ DBSCAN

* Clustering por densidad con visualización 2D y 3D

### 5️⃣ Subtractive Clustering

* Número de clústeres determinado automáticamente
* Visualización en 2D y 3D con centros marcados

### 6️⃣ Fuzzy C-Means

* Asignación de etiquetas por máximo grado de pertenencia
* Visualización de clusters y pertenencias

### 7️⃣ Comparación de los cuatro métodos

Visualización conjunta de KMeans, DBSCAN, Subtractive y FCM en el mismo espacio PCA.

### 8️⃣ Evaluación contra etiquetas reales

* **ARI** para KMeans, Subtractive y Fuzzy C-Means
* Visualización de etiquetas reales en 3D

### 9️⃣ Análisis de errores por unidad académica

* Alineación de etiquetas KMeans con etiquetas reales (mejor de las dos asignaciones)
* Cálculo de tasa de error por `unidad`
* Identificación de la unidad con **más errores** y la de **menos errores**
* Exportación de resultados:

```
errores_por_unidad.csv
resultado_clustering_completo.csv
```

📌 Resultado: Comparación completa de cuatro algoritmos de clustering con análisis interpretable de errores por unidad académica, incluyendo implementaciones propias de Subtractive Clustering y Fuzzy C-Means.

---

# 📅 Semana 11 – Informe 2 Teórico Práctico

📂 Carpeta: `informe_teorico_practico_02_ML_SantiagoManco/`

El segundo informe consolida el desarrollo completo del proyecto a lo largo del semestre, profundizando en los resultados de los modelos supervisados y no supervisados aplicados al dataset FIRE_UdeA.

Incluye análisis teórico-práctico de:

* Modelos de clasificación (Random Forest, Gradient Boosting) con optimización de hiperparámetros
* Técnicas de clustering (KMeans, DBSCAN, Subtractive, Fuzzy C-Means)
* Comparación cuantitativa de algoritmos mediante métricas estándar (Accuracy, AUC, Silhouette, ARI)
* Interpretación de resultados en el contexto financiero universitario

---

# 🛠 Tecnologías Utilizadas

* Python
* NumPy
* Pandas
* Scikit-learn
* Plotly
* UMAP
* Matplotlib / Seaborn
* Next.js (dashboard financiero)
* JSON / Parquet / CSV
