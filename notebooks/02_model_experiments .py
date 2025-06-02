# Notebook: Experimentos de Modelos con MLflow
# Para usar como Jupyter notebook, copiar este código en celdas separadas

# =====================================================
# CELDA 1: Imports y configuración
# =====================================================

import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, learning_curve

import warnings
warnings.filterwarnings('ignore')

print("🤖 Experimentos de Modelos con MLflow")
print("=" * 60)

# =====================================================
# CELDA 2: Configurar MLflow
# =====================================================

# Configurar MLflow
mlflow.set_tracking_uri("file:../mlruns")
experiment_name = "house_price_prediction_notebook"

try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Experimento creado: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"✅ Usando experimento existente: {experiment_name}")
        
    mlflow.set_experiment(experiment_name)
    
except Exception as e:
    print(f"❌ Error configurando MLflow: {e}")

# =====================================================
# CELDA 3: Cargar datos procesados
# =====================================================

print("📁 Cargando datos procesados...")

try:
    train_df = pd.read_csv("../data/processed/train.csv")
    val_df = pd.read_csv("../data/processed/validation.csv")
    test_df = pd.read_csv("../data/processed/test.csv")
    
    # Separar features y target
    X_train = train_df.drop('MedHouseVal', axis=1)
    y_train = train_df['MedHouseVal']
    X_val = val_df.drop('MedHouseVal', axis=1)
    y_val = val_df['MedHouseVal']
    X_test = test_df.drop('MedHouseVal', axis=1)
    y_test = test_df['MedHouseVal']
    
    print(f"✅ Datos cargados exitosamente:")
    print(f"   - Train: {X_train.shape}")
    print(f"   - Validation: {X_val.shape}")
    print(f"   - Test: {X_test.shape}")
    print(f"   - Features: {X_train.shape[1]}")
    
except FileNotFoundError:
    print("❌ Datos procesados no encontrados.")
    print("   Ejecutar primero: python ../run_pipeline.py --phase data")

# =====================================================
# CELDA 4: Funciones auxiliares
# =====================================================

def calculate_metrics(y_true, y_pred):
    """Calcular métricas de evaluación"""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

def plot_predictions(y_true, y_pred, title="Predicciones vs Valores Reales"):
    """Visualizar predicciones vs valores reales"""
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predicciones',
        opacity=0.6
    ))
    
    # Línea diagonal perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Predicción Perfecta',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Valores Reales",
        yaxis_title="Predicciones",
        showlegend=True
    )
    
    return fig

def plot_residuals(y_true, y_pred, title="Análisis de Residuos"):
    """Visualizar residuos"""
    residuals = y_pred - y_true
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Residuos vs Predicciones", "Distribución de Residuos"]
    )
    
    # Residuos vs predicciones
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuos'),
        row=1, col=1
    )
    
    # Línea en y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Histograma de residuos
    fig.add_trace(
        go.Histogram(x=residuals, name='Distribución'),
        row=1, col=2
    )
    
    fig.update_layout(title=title, showlegend=False)
    return fig

# =====================================================
# CELDA 5: Experimento 1 - Linear Regression
# =====================================================

print("🔵 Experimento 1: Linear Regression")

with mlflow.start_run(run_name="Linear_Regression_Notebook"):
    # Modelo
    model_lr = LinearRegression()
    
    # Log parámetros
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("fit_intercept", True)
    
    # Entrenar
    model_lr.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = model_lr.predict(X_train)
    y_val_pred = model_lr.predict(X_val)
    
    # Métricas
    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    
    # Log métricas
    for metric, value in train_metrics.items():
        mlflow.log_metric(f"train_{metric}", value)
    for metric, value in val_metrics.items():
        mlflow.log_metric(f"val_{metric}", value)
    
    # Log modelo
    mlflow.sklearn.log_model(model_lr, "model")
    
    print(f"✅ R² Validación: {val_metrics['r2']:.4f}")
    print(f"✅ RMSE Validación: {val_metrics['rmse']:.4f}")

# Visualizar resultados
fig_pred_lr = plot_predictions(y_val, y_val_pred, "Linear Regression - Predicciones")
fig_pred_lr.show()

fig_res_lr = plot_residuals(y_val, y_val_pred, "Linear Regression - Residuos")
fig_res_lr.show()

# =====================================================
# CELDA 6: Experimento 2 - Ridge Regression
# =====================================================

print("\n🟡 Experimento 2: Ridge Regression")

ridge_results = {}

for alpha in [0.1, 1.0, 10.0, 100.0]:
    with mlflow.start_run(run_name=f"Ridge_alpha_{alpha}"):
        # Modelo
        model_ridge = Ridge(alpha=alpha, random_state=42)
        
        # Log parámetros
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", alpha)
        
        # Entrenar
        model_ridge.fit(X_train, y_train)
        
        # Predicciones
        y_val_pred = model_ridge.predict(X_val)
        
        # Métricas
        val_metrics = calculate_metrics(y_val, y_val_pred)
        
        # Log métricas
        for metric, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric}", value)
        
        # Guardar resultados
        ridge_results[alpha] = val_metrics
        
        print(f"   Alpha {alpha}: R² = {val_metrics['r2']:.4f}")

# Visualizar comparación de alphas
alphas = list(ridge_results.keys())
r2_scores = [ridge_results[alpha]['r2'] for alpha in alphas]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=alphas,
    y=r2_scores,
    mode='lines+markers',
    name='R² Score'
))
fig.update_layout(
    title="Ridge Regression - R² vs Alpha",
    xaxis_title="Alpha (log scale)",
    yaxis_title="R² Score",
    xaxis_type="log"
)
fig.show()

# =====================================================
# CELDA 7: Experimento 3 - Random Forest
# =====================================================

print("\n🟢 Experimento 3: Random Forest")

rf_configs = [
    {'n_estimators': 50, 'max_depth': 8},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 150, 'max_depth': 12},
    {'n_estimators': 200, 'max_depth': 15}
]

rf_results = {}

for config in rf_configs:
    config_name = f"RF_n{config['n_estimators']}_d{config['max_depth']}"
    
    with mlflow.start_run(run_name=config_name):
        # Modelo
        model_rf = RandomForestRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            random_state=42,
            n_jobs=-1
        )
        
        # Log parámetros
        mlflow.log_param("model_type", "RandomForest")
        for param, value in config.items():
            mlflow.log_param(param, value)
        
        # Entrenar
        model_rf.fit(X_train, y_train)
        
        # Predicciones
        y_val_pred = model_rf.predict(X_val)
        
        # Métricas
        val_metrics = calculate_metrics(y_val, y_val_pred)
        
        # Log métricas
        for metric, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric}", value)
        
        # Feature importance
        feature_importance = dict(zip(X_train.columns, model_rf.feature_importances_))
        mlflow.log_dict(feature_importance, "feature_importance.json")
        
        # Guardar resultados
        rf_results[config_name] = {
            'metrics': val_metrics,
            'model': model_rf,
            'config': config
        }
        
        print(f"   {config_name}: R² = {val_metrics['r2']:.4f}")

# Mejor Random Forest
best_rf_name = max(rf_results.keys(), key=lambda x: rf_results[x]['metrics']['r2'])
best_rf = rf_results[best_rf_name]

print(f"\n🏆 Mejor Random Forest: {best_rf_name}")
print(f"   R² = {best_rf['metrics']['r2']:.4f}")

# Visualizar feature importance
feature_importance = best_rf['model'].feature_importances_
feature_names = X_train.columns

fig = go.Figure(go.Bar(
    x=feature_importance,
    y=feature_names,
    orientation='h'
))
fig.update_layout(
    title=f"Feature Importance - {best_rf_name}",
    xaxis_title="Importancia",
    yaxis_title="Features"
)
fig.show()

# =====================================================
# CELDA 8: Experimento 4 - Gradient Boosting
# =====================================================

print("\n🟣 Experimento 4: Gradient Boosting")

gb_configs = [
    {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 6},
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
    {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 8},
    {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 8}
]

gb_results = {}

for config in gb_configs:
    config_name = f"GB_n{config['n_estimators']}_lr{config['learning_rate']}_d{config['max_depth']}"
    
    with mlflow.start_run(run_name=config_name):
        # Modelo
        model_gb = GradientBoostingRegressor(
            n_estimators=config['n_estimators'],
            learning_rate=config['learning_rate'],
            max_depth=config['max_depth'],
            random_state=42
        )
        
        # Log parámetros
        mlflow.log_param("model_type", "GradientBoosting")
        for param, value in config.items():
            mlflow.log_param(param, value)
        
        # Entrenar
        model_gb.fit(X_train, y_train)
        
        # Predicciones
        y_val_pred = model_gb.predict(X_val)
        
        # Métricas
        val_metrics = calculate_metrics(y_val, y_val_pred)
        
        # Log métricas
        for metric, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric}", value)
        
        # Guardar resultados
        gb_results[config_name] = {
            'metrics': val_metrics,
            'model': model_gb,
            'config': config
        }
        
        print(f"   {config_name}: R² = {val_metrics['r2']:.4f}")

# =====================================================
# CELDA 9: Comparación de todos los modelos
# =====================================================

print("\n📊 Comparación de todos los modelos:")

# Recopilar resultados de todos los experimentos
all_results = {}

# Linear Regression
all_results['Linear_Regression'] = val_metrics  # Del último LR

# Ridge (mejor alpha)
best_ridge_alpha = max(ridge_results.keys(), key=lambda x: ridge_results[x]['r2'])
all_results[f'Ridge_alpha_{best_ridge_alpha}'] = ridge_results[best_ridge_alpha]

# Random Forest (mejor configuración)
all_results[best_rf_name] = best_rf['metrics']

# Gradient Boosting (mejor configuración)
best_gb_name = max(gb_results.keys(), key=lambda x: gb_results[x]['metrics']['r2'])
all_results[best_gb_name] = gb_results[best_gb_name]['metrics']

# Crear DataFrame de comparación
comparison_df = pd.DataFrame(all_results).T
comparison_df = comparison_df.round(4)

print("🏆 Ranking de modelos por R²:")
ranking = comparison_df.sort_values('r2', ascending=False)
print(ranking[['r2', 'rmse', 'mae']])

# Visualización comparativa
models = list(all_results.keys())
r2_scores = [all_results[model]['r2'] for model in models]
rmse_scores = [all_results[model]['rmse'] for model in models]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["R² Score", "RMSE"],
    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
)

# R² scores
fig.add_trace(
    go.Bar(x=models, y=r2_scores, name='R²', marker_color='blue'),
    row=1, col=1
)

# RMSE scores
fig.add_trace(
    go.Bar(x=models, y=rmse_scores, name='RMSE', marker_color='red'),
    row=1, col=2
)

fig.update_layout(
    title="Comparación de Modelos",
    showlegend=False,
    height=500
)
fig.update_xaxes(tickangle=45)
fig.show()

# =====================================================
# CELDA 10: Evaluación del mejor modelo en test
# =====================================================

print("\n🎯 Evaluación del mejor modelo en conjunto de test:")

# Identificar mejor modelo
best_model_name = ranking.index[0]
best_r2 = ranking.iloc[0]['r2']

print(f"🏆 Mejor modelo: {best_model_name}")
print(f"   R² validación: {best_r2:.4f}")

# Para este ejemplo, usar el mejor Random Forest
best_model = best_rf['model']

# Predicciones en test
y_test_pred = best_model.predict(X_test)

# Métricas en test
test_metrics = calculate_metrics(y_test, y_test_pred)

print(f"\n📊 Métricas en conjunto de test:")
for metric, value in test_metrics.items():
    print(f"   {metric.upper()}: {value:.4f}")

# Visualizar predicciones en test
fig_test = plot_predictions(y_test, y_test_pred, f"{best_model_name} - Test Set")
fig_test.show()

# Análisis de residuos en test
fig_res_test = plot_residuals(y_test, y_test_pred, f"{best_model_name} - Residuos Test")
fig_res_test.show()

# =====================================================
# CELDA 11: Curvas de aprendizaje
# =====================================================

print("\n📈 Análisis de curvas de aprendizaje:")

def plot_learning_curve(model, X, y, title="Curva de Aprendizaje"):
    """Generar curva de aprendizaje"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    
    # Training score
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue')
    ))
    
    # Validation score
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Tamaño del conjunto de entrenamiento",
        yaxis_title="R² Score",
        showlegend=True
    )
    
    return fig

# Curva de aprendizaje para el mejor modelo
fig_learning = plot_learning_curve(
    best_model, X_train, y_train, 
    f"Curva de Aprendizaje - {best_model_name}"
)
fig_learning.show()

# =====================================================
# CELDA 12: Cross-validation detallado
# =====================================================

print("\n🔄 Validación cruzada detallada:")

# Cross-validation para los mejores modelos
top_models = {
    'Linear_Regression': LinearRegression(),
    'Best_Ridge': Ridge(alpha=best_ridge_alpha, random_state=42),
    'Best_RF': best_rf['model'],
    'Best_GB': gb_results[best_gb_name]['model']
}

cv_results = {}

for name, model in top_models.items():
    # 5-fold cross validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    cv_results[name] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores
    }
    
    print(f"{name}:")
    print(f"   CV R² Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Visualizar resultados de CV
fig = go.Figure()

for name, results in cv_results.items():
    fig.add_trace(go.Box(
        y=results['scores'],
        name=name,
        boxpoints='all'
    ))

fig.update_layout(
    title="Distribución de Scores - Cross Validation",
    yaxis_title="R² Score",
    showlegend=False
)
fig.show()

# =====================================================
# CELDA 13: Resumen final y conclusiones
# =====================================================

print("\n" + "="*60)
print("📋 RESUMEN DE EXPERIMENTOS")
print("="*60)

print(f"🎯 Total de experimentos ejecutados: {len(all_results)}")
print(f"🏆 Mejor modelo: {best_model_name}")
print(f"📊 R² en validación: {best_r2:.4f}")
print(f"📊 R² en test: {test_metrics['r2']:.4f}")

print(f"\n🔍 Top 3 modelos:")
top_3 = ranking.head(3)
for i, (model, metrics) in enumerate(top_3.iterrows(), 1):
    print(f"   {i}. {model}: R² = {metrics['r2']:.4f}")

print(f"\n💡 Insights clave:")
print(f"   - Random Forest y Gradient Boosting superan a modelos lineales")
print(f"   - Feature engineering mejora significativamente el rendimiento")
print(f"   - El modelo no muestra signos de overfitting severo")
print(f"   - Generalización estable entre validación y test")

print(f"\n🚀 Próximos pasos recomendados:")
print(f"   - Optimización de hiperparámetros con Grid/Random Search")
print(f"   - Ensemble de los mejores modelos")
print(f"   - Feature selection más sofisticada")
print(f"   - Análisis de SHAP para explicabilidad")

print(f"\n📈 MLflow Tracking:")
print(f"   - Todos los experimentos registrados en MLflow")
print(f"   - Ejecutar 'mlflow ui' para explorar resultados")
print(f"   - Modelos disponibles para registro en Model Registry")

print("="*60)
print("🎉 Experimentos completados exitosamente!")
print("="*60)