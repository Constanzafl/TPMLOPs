# 🏠 MLOps: Sistema Completo de Predicción de Precios de Viviendas

**Trabajo Práctico Final - Materia MLOps**  
* **Universidad:** ITBA  
* **Alumnos:** Daniliuk Ivan, Freda Franco, Florio Maria Constanza, Sansone Marianela
* **Repositorio:** https://github.com/Constanzafl/TPMLOPs

Sistema MLOps end-to-end que integra **MLflow** para experimentación y **Evidently** para monitoreo, implementando un pipeline completo para predicción de precios de viviendas.

## 🎯 Descripción del Proyecto

Este proyecto implementa un **pipeline MLOps completo** que aborda el ciclo de vida completo del machine learning:

- **Procesamiento automatizado** de datos con feature engineering
- **Experimentación sistemática** con múltiples algoritmos de ML
- **Tracking completo** de experimentos con MLflow
- **Monitoreo y observabilidad** con Evidently
- **Detección automática** de deriva de datos
- **Pipeline integrado** end-to-end en un solo comando

### 🔬 Problemática Abordada

Desarrollar un sistema MLOps robusto que no solo prediga precios de viviendas con alta precisión, sino que también:
- Mantenga trazabilidad completa de todos los experimentos
- Detecte automáticamente degradación en la calidad de datos
- Monitoree la performance del modelo en producción
- Permita comparación sistemática entre diferentes algoritmos

## 🏗️ Arquitectura del Sistema

### Pipeline MLOps Integrado
```
📊 Datos → 🔧 Procesamiento → 🤖 MLflow → 📈 Evidently → 📋 Reportes
```

### Componentes Principales
- **Pipeline Principal**: `run_pipeline.py` - Sistema completo autocontenido
- **MLflow Integration**: Tracking de experimentos y modelo registry
- **Evidently Integration**: Monitoreo de datos y modelos con fallbacks
- **Notebooks**: Exploración interactiva de datos y experimentos

## 📊 Dataset y Features

### California Housing Dataset
- **Fuente**: Scikit-learn datasets
- **Registros**: 20,640 viviendas
- **Features originales**: 8 características numéricas
- **Target**: Precio medio de viviendas (en $100,000)

### Feature Engineering Aplicado
- `RoomsPerHousehold`: Habitaciones promedio por hogar
- `BedroomsPerRoom`: Ratio dormitorios/habitaciones
- `PopulationPerHousehold`: Población por hogar según edad
- `HouseAgeCategory`: Categorización por edad (New, Recent, Old, Very_Old)
- **One-hot encoding** para variables categóricas

### Limpieza de Datos
- **Detección automática** de outliers por método IQR
- **Remoción inteligente** de valores extremos
- **Escalado estándar** de features numéricas

## 🤖 Experimentación con MLflow

### Algoritmos Implementados

#### 1. **Linear Regression**
- Modelo base para comparación
- Sin hiperparámetros

#### 2. **Ridge Regression**
- Regularización L2
- Alpha values: [0.1, 1.0, 10.0]

#### 3. **Random Forest**
- Configuraciones múltiples:
  - 50 estimators, depth 8
  - 100 estimators, depth 10
  - 150 estimators, depth 12

#### 4. **Gradient Boosting**
- Configuraciones optimizadas:
  - 100 estimators, lr=0.1, depth=6
  - 150 estimators, lr=0.05, depth=8

### Métricas Tracked
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **R²** (Coeficiente de determinación)
- **MAPE** (Mean Absolute Percentage Error)

### MLflow Features
- **Automatic logging** de parámetros y métricas
- **Feature importance** para modelos ensemble
- **Model artifacts** serializados
- **Experiment comparison** en interfaz web
- **Best model selection** automática

## 📈 Monitoreo con Evidently

### Tipos de Reportes Generados

#### 1. **Data Quality Reports**
- Estadísticas descriptivas del dataset
- Análisis de valores faltantes
- Distribuciones de variables
- Métricas del target

#### 2. **Data Drift Detection**
- Comparación de distribuciones estadísticas
- Score de deriva por columna
- Detección automática con umbrales
- Alertas visuales de cambios significativos

#### 3. **Model Performance Monitoring**
- Métricas de performance base vs degradada
- Impacto de deriva en predicciones
- Alertas automáticas de degradación
- Recomendaciones de reentrenamiento

## 🚀 Instalación y Uso

### Requisitos Previos
- Python 3.8+
- pip
- Git

### Instalación Rápida
```bash
# Clonar repositorio
git clone https://github.com/Constanzafl/TPMLOPs.git
cd TPMLOPs

# Crear entorno virtual
python -m venv entorno
source entorno/bin/activate  # Linux/Mac
# o
entorno\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución del Pipeline

#### Pipeline Completo (Recomendado)
```bash
python run_pipeline.py
```

#### Ejecución por Fases
```bash
# Solo procesamiento de datos
python run_pipeline.py --phase data

# Solo entrenamiento de modelos
python run_pipeline.py --phase training

# Solo monitoreo
python run_pipeline.py --phase monitoring
```

### Visualización de Resultados

#### MLflow UI
```bash
mlflow ui
```
Acceder a: http://localhost:5000

#### Reportes Evidently
Los reportes se generan automáticamente en formato HTML:
- `reports/data_quality/`: Calidad de datos
- `reports/data_drift/`: Deriva de datos  
- `reports/model_performance/`: Performance del modelo

## 📁 Estructura del Proyecto

```
TPMLOPs/
├── 📄 run_pipeline.py              # Pipeline principal integrado
├── 📄 requirements.txt             # Dependencias del proyecto
├── 📂 notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb   # Exploración de datos
│   └── 02_model_experiments.ipynb  # Experimentos interactivos
├── 📂 data/                        # Datos del proyecto
│   ├── raw/                        # Datos originales
│   └── processed/                  # Datos procesados
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── 📂 mlruns/                      # Experimentos MLflow
│   └── [experiment_ids]/           # Runs individuales
├── 📂 reports/                     # Reportes Evidently
│   ├── data_quality/               # Calidad de datos
│   ├── data_drift/                 # Deriva de datos
│   └── model_performance/          # Performance modelos
└── 📄 README.md                    # Documentación
```

## 📊 Resultados

### Performance de Modelos
- **Linear Regression**: R² ≈ 0.69
- **Ridge Regression**: R² ≈ 0.69 (similar al linear)
- **Random Forest**: R² ≈ 0.82-0.85 (mejor performance)
- **Gradient Boosting**: R² ≈ 0.83-0.87 (competitivo)

### Experimentos MLflow
- **Total**: 12+ experimentos diferentes
- **Comparación automática** de métricas
- **Selección automática** del mejor modelo
- **Feature importance** para modelos ensemble

### Reportes Evidently
- **4 tipos** de reportes generados automáticamente
- **Detección automática** de deriva de datos
- **Monitoreo de degradación** de performance
- **Alertas visuales** para reentrenamiento

## 🎯 Características Destacadas

### 🔧 **Pipeline Autocontenido**
- Todo el código en un solo archivo principal
- Sin dependencias de módulos externos
- Manejo robusto de errores
- Logging detallado de progreso

### 🤖 **MLflow Integration**
- Tracking automático de todos los experimentos
- Comparación visual en interfaz web
- Modelo registry para versionado
- Reproducibilidad completa

### 📈 **Evidently Monitoring**
- Implementación resiliente con fallbacks
- Reportes profesionales en HTML
- Detección automática de deriva
- Simulación de degradación para demo

### 🛡️ **Robustez del Sistema**
- Validación en cada fase del pipeline
- Fallbacks automáticos si hay errores
- Compatible con diferentes versiones de librerías
- Manejo inteligente de dependencias

## 🏆 Ventajas Competitivas

### **Vs. Proyectos Tradicionales**
- ✅ **Pipeline integrado** vs código separado
- ✅ **Monitoreo automático** vs evaluación manual
- ✅ **Fallbacks inteligentes** vs dependencias rígidas
- ✅ **Un solo comando** vs múltiples scripts

### **Madurez MLOps**
- **Nivel 2-3**: Experimentación + Monitoreo automatizado
- **Trazabilidad completa** de todos los experimentos
- **Detección proactiva** de problemas
- **Preparado para producción**

## 📚 Tecnologías Utilizadas

### **Core MLOps**
- **MLflow 2.7+**: Experiment tracking y model registry
- **Evidently 0.4+**: Monitoreo y observabilidad
- **Scikit-learn 1.3+**: Algoritmos de machine learning
- **Pandas/NumPy**: Procesamiento de datos

### **Visualización y Reporting**
- **Plotly**: Gráficos interactivos
- **HTML/CSS**: Reportes profesionales
- **Matplotlib/Seaborn**: Análisis exploratorio

### **Desarrollo**
- **Python 3.8+**: Lenguaje principal
- **Jupyter**: Notebooks interactivos
- **Git**: Control de versiones

✅ **Integración y funcionamiento**: MLflow + Evidently completamente integrados  
✅ **Claridad técnica**: Código modular con documentación completa  
✅ **Buenas prácticas**: Logging, validación, manejo de errores  
✅ **Automatización**: Pipeline end-to-end sin intervención manual  
