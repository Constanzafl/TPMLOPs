# 🏠 MLOps: Predicción de Precios de Viviendas

**Trabajo Práctico Final - Materia MLOps**

Sistema completo de MLOps para predicción de precios de viviendas utilizando MLflow para experimentación y Evidently para monitoreo de modelos.

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline completo de MLOps que incluye:

- **Procesamiento de datos** con feature engineering
- **Experimentación de modelos** con MLflow tracking
- **Monitoreo y observabilidad** con Evidently
- **Pipeline automatizado** integrado

### 🎯 Problemática Abordada

Desarrollar un sistema MLOps robusto para predecir precios de viviendas que permita:
- Comparar múltiples algoritmos de ML de forma sistemática
- Monitorear la calidad de datos y performance del modelo
- Detectar deriva de datos (data drift)
- Mantener trazabilidad completa de experimentos
- Automatizar el pipeline de ML end-to-end

## 🛠️ Tecnologías Utilizadas

### Core MLOps
- **MLflow**: Tracking de experimentos, registro de modelos
- **Evidently**: Monitoreo de datos y modelos, detección de deriva
- **Scikit-learn**: Algoritmos de machine learning
- **Pandas/NumPy**: Procesamiento de datos

### Herramientas de Desarrollo
- **Python 3.8+**: Lenguaje principal
- **Jupyter**: Notebooks para exploración
- **Plotly**: Visualizaciones interactivas

## 📊 Dataset

**California Housing Dataset**
- **Fuente**: Scikit-learn datasets
- **Registros**: 20,640 viviendas
- **Features**: 8 características numéricas
- **Target**: Precio medio de viviendas (en $100,000)

### Variables del Dataset
- `MedInc`: Ingreso medio del área
- `HouseAge`: Edad media de las casas
- `AveRooms`: Promedio de habitaciones por vivienda
- `AveBedrms`: Promedio de dormitorios por vivienda
- `Population`: Población del área
- `AveOccup`: Promedio de ocupantes por vivienda
- `Latitude`: Latitud geográfica
- `Longitude`: Longitud geográfica

## 🚀 Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/mlops-house-prices.git
cd mlops-house-prices
```

### 2. Crear Entorno Virtual
```bash
python -m venv mlops_env
source mlops_env/bin/activate  # Linux/Mac
# o
mlops_env\Scripts\activate     # Windows
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Verificar Instalación
```bash
python -c "import mlflow, evidently; print('✅ Instalación exitosa')"
```

## 🎯 Uso del Sistema

### Ejecución Completa del Pipeline
```bash
python run_pipeline.py
```

### Ejecución por Fases
```bash
# Solo procesamiento de datos
python run_pipeline.py --phase data

# Solo entrenamiento de modelos
python run_pipeline.py --phase training

# Solo monitoreo
python run_pipeline.py --phase monitoring
```

### Ver Experimentos MLflow
```bash
mlflow ui
```
Abrir navegador en: http://localhost:5000

### Ver Reportes Evidently
Los reportes se generan en formato HTML en la carpeta `reports/`:
- `reports/data_quality/`: Calidad de datos
- `reports/data_drift/`: Deriva de datos
- `reports/model_performance/`: Performance del modelo

## 📁 Estructura del Proyecto

```
mlops-house-prices/
├── 📂 src/                          # Código fuente
│   ├── data_processing.py           # Procesamiento de datos
│   ├── model_training.py            # Entrenamiento con MLflow
│   └── monitoring.py                # Monitoreo con Evidently
├── 📂 data/                         # Datos
│   ├── raw/                         # Datos originales
│   └── processed/                   # Datos procesados
├── 📂 notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # Exploración de datos
│   └── 02_model_experiments.ipynb   # Experimentos de modelos
├── 📂 mlruns/                       # Experimentos MLflow
├── 📂 reports/                      # Reportes Evidently
├── 📂 models/                       # Modelos entrenados
├── 📄 run_pipeline.py               # Pipeline principal
├── 📄 requirements.txt              # Dependencias
└── 📄 README.md                     # Documentación

```

## 🤖 Modelos Implementados

### Algoritmos de ML
1. **Linear Regression**: Modelo base simple
2. **Ridge Regression**: Regularización L2 (α = 0.1, 1.0, 10.0)
3. **Random Forest**: Ensamble de árboles (50, 100, 200 estimadores)
4. **Gradient Boosting**: Boosting gradient (diferentes lr y profundidad)

### Métricas de Evaluación
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **R²** (Coeficiente de determinación)
- **MAPE** (Mean Absolute Percentage Error)

## 📈 Experimentación con MLflow

### Tracking de Experimentos
- **Parámetros**: Hiperparámetros de cada modelo
- **Métricas**: Performance en train y validation
- **Artifacts**: Modelos serializados, gráficos, feature importance
- **Metadata**: Timestamps, código, environment

### Comparación de Modelos
- Interfaz web para comparar experimentos
- Ranking automático por métricas
- Visualización de curvas de aprendizaje
- Registro de mejores modelos

## 🔍 Monitoreo con Evidently

### Tipos de Reportes
1. **Data Quality**: Análisis de calidad de datos
   - Valores faltantes
   - Distribuciones de variables
   - Correlaciones
   - Estadísticas descriptivas

2. **Data Drift**: Detección de deriva de datos
   - Comparación de distribuciones
   - Tests estadísticos
   - Alertas automáticas
   - Visualizaciones de cambios

3. **Model Performance**: Evaluación del modelo
   - Métricas de regresión
   - Residuos y errores
   - Predicted vs Actual plots
   - Distribución de errores

### Tests Automatizados
- Tests de calidad de datos
- Tests de performance del modelo
- Validaciones automáticas
- Alertas por umbrales

## 🎬 Demostración para Presentación

### 1. Exploración de Datos (5 min)
```python
# Ejecutar notebook de exploración
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Experimentos MLflow (8 min)
```python
# Mostrar tracking de experimentos
python run_pipeline.py --phase training
mlflow ui  # Mostrar interfaz web
```

### 3. Monitoreo Evidently (5 min)
```python
# Generar reportes de monitoreo
python run_pipeline.py --phase monitoring
# Abrir reportes HTML generados
```

### 4. Pipeline Completo (2 min)
```python
# Demostración end-to-end
python run_pipeline.py --verbose
```

## 👥 Distribución de Tareas (4 Personas)

### 👤 Persona 1: Data Engineering
- Implementación de `data_processing.py`
- Feature engineering y limpieza
- Notebook de exploración de datos
- Documentación del procesamiento

### 👤 Persona 2: MLflow & Experimentación
- Implementación de `model_training.py`
- Configuración de MLflow tracking
- Comparación de modelos
- Optimización de hiperparámetros

### 👤 Persona 3: Evidently & Monitoreo
- Implementación de `monitoring.py`
- Configuración de reportes Evidently
- Tests automatizados
- Detección de deriva de datos

### 👤 Persona 4: Integración & DevOps
- Pipeline principal `run_pipeline.py`
- Documentación del proyecto
- Testing y validación
- Preparación de la presentación

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
# Configurar MLflow tracking URI
export MLFLOW_TRACKING_URI=file:./mlruns

# Configurar logging level
export MLOPS_LOG_LEVEL=INFO
```

### Personalización
- Modificar hiperparámetros en `model_training.py`
- Ajustar umbrales de deriva en `monitoring.py`
- Agregar nuevos modelos al pipeline
- Personalizar reportes de Evidently

## 🧪 Testing

### Ejecutar Tests
```bash
python -m pytest tests/ -v
```

### Tests Incluidos
- Validación de datos procesados
- Tests de modelos entrenados
- Verificación de reportes generados
- Tests de integración del pipeline

## 📚 Recursos Adicionales

### Documentación Oficial
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

### Tutoriales
- [MLflow Tracking Tutorial](https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)
- [Evidently Getting Started](https://docs.evidentlyai.com/get-started/tutorial)

## 🐛 Troubleshooting

### Problemas Comunes
1. **Error de permisos en MLflow**
   ```bash
   chmod -R 755 mlruns/
   ```

2. **Puerto ocupado para MLflow UI**
   ```bash
   mlflow ui --port 5001
   ```

3. **Memoria insuficiente**
   - Reducir número de experimentos
   - Usar datasets más pequeños para pruebas

### Logs
Los logs se guardan en `mlops_pipeline.log` para debugging.

## 📞 Soporte

Para problemas técnicos:
1. Revisar logs en `mlops_pipeline.log`
2. Verificar instalación de dependencias
3. Consultar documentación oficial
4. Crear issue en el repositorio

## 🎯 Criterios de Evaluación Cubiertos

✅ **Integración de herramientas**: MLflow + Evidently  
✅ **Claridad técnica**: Código documentado y modular  
✅ **Buenas prácticas**: Logging, testing, estructura clara  
✅ **Automatización**: Pipeline completo automatizado  
✅ **Originalidad**: Feature engineering custom y monitoreo avanzado  

## 📄 Licencia

Este proyecto es parte de un trabajo académico para la materia MLOps.

---

**Desarrollado con ❤️ para el curso de MLOps**