# Regresión Lineal y Logística – Movies Dataset

## 📌 Descripción General

Este proyecto aplica **Regresión Lineal (Ridge y Lasso)** y **Regresión Logística** sobre un dataset de películas con el objetivo de:

1. Predecir los ingresos de una película (`Gross`) mediante modelos de regresión.
2. Clasificar si una película tiene ingresos altos usando un modelo de clasificación.

El proyecto incluye:

* Limpieza y transformación de datos
* Ingeniería de variables
* Construcción de pipelines
* Búsqueda de hiperparámetros con `RandomizedSearchCV`
* Evaluación de modelos
* Exportación de visualizaciones

---

# 📂 Dataset

El archivo utilizado es:

```
movies.csv
```

Contiene información como:

* `YEAR` (año)
* `RunTime` (duración)
* `RATING` (calificación)
* `VOTES` (número de votos)
* `Gross` (ingresos)

---

# 🧹 Limpieza de Datos

Se realizaron las siguientes transformaciones:

* Extracción del año numérico desde `YEAR`
* Extracción del valor numérico desde `RunTime`
* Eliminación de comas en `VOTES`
* Eliminación de símbolos (`$`, comas) en `Gross`
* Conversión de variables a tipo numérico
* Eliminación de filas con valores nulos

---

# 📊 Parte 1 – Regresión Lineal

## 🎯 Objetivo

Predecir los ingresos de una película (`Gross`).

## 🔧 Variables utilizadas

* `RunTime`
* `RATING`
* `VOTES`
* `YEAR`

## 🧠 Modelos implementados

Se entrenaron dos modelos con regularización:

* 🔵 Ridge Regression
* 🟢 Lasso Regression

Cada modelo se construyó usando un `Pipeline` con:

1. Expansión polinómica (`PolynomialFeatures`)
2. Escalamiento (`StandardScaler`)
3. Modelo de regresión regularizado

## 🔍 Optimización

Se utilizó:

```
RandomizedSearchCV
```

Para ajustar:

* Grado del polinomio (1 a 3)
* Parámetro de regularización (`alpha`)

## 📈 Métricas evaluadas

* R² (coeficiente de determinación)
* MAE (Error Absoluto Medio)

## 📊 Gráficas generadas

Guardadas en la carpeta `output/`:

* `reg_lineal_train_test.png`
* `ridge_prediction.png`
* `lasso_prediction.png`

---

# 📊 Parte 2 – Regresión Logística

## 🎯 Objetivo

Clasificar si una película tiene ingresos altos.

Se creó la variable:

```
High_Gross = 1 si Gross > mediana
High_Gross = 0 en caso contrario
```

Es decir, se clasifica si la película está por encima o por debajo de la mediana de ingresos del dataset.

---

## 🧠 Modelo

Se utilizó un `Pipeline` con:

1. `PolynomialFeatures`
2. `StandardScaler`
3. `LogisticRegression`

También se optimizaron hiperparámetros usando `RandomizedSearchCV`.

Parámetros ajustados:

* Grado polinómico
* Parámetro C (regularización)

---

## 📈 Métricas evaluadas

* Accuracy
* F1-score
* Matriz de confusión

## 📊 Gráfica generada

* `confusion_matrix.png`

---

# 🏗 Estructura del Proyecto

```
.
│
├── movies.csv
├── lecture05_movies_exercise_santiago_manco.py
├── output/
│   ├── reg_lineal_train_test.png
│   ├── ridge_prediction.png
│   ├── lasso_prediction.png
│   └── confusion_matrix.png
```

---

# ⚙️ Requisitos

Instalar dependencias:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

---

# ▶️ Cómo Ejecutar

```bash
python lecture05_movies_exercise_santiago_manco.py
```

Las imágenes se exportarán automáticamente en la carpeta `output/`.

---

# 📌 Conclusiones

* Ridge y Lasso obtienen resultados similares, lo que indica estabilidad en las variables utilizadas.
* La predicción de ingresos presenta mayor dificultad en valores extremos.
* La regresión logística logra una buena capacidad de clasificación.
* La regularización ayuda a controlar el sobreajuste.
