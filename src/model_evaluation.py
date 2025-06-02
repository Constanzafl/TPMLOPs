"""
Módulo para evaluación completa de modelos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
import joblib
import json
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Clase para evaluación completa de modelos de regresión"""
    
    def __init__(self, model, model_name="Model"):
        self.model = model
        self.model_name = model_name
        self.evaluation_results = {}
        
    def calculate_regression_metrics(self, y_true, y_pred):
        """Calcular métricas de regresión"""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mse': mean_squared_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'explained_variance': 1 - np.var(y_true - y_pred) / np.var(y_true)
        }
        
        # Métricas adicionales
        residuals = y_pred - y_true
        metrics.update({
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'max_error': np.max(np.abs(residuals)),
            'q95_error': np.percentile(np.abs(residuals), 95)
        })
        
        return metrics
    
    def evaluate_on_sets(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
        """Evaluar modelo en conjuntos de datos"""
        logger.info(f"Evaluando modelo {self.model_name}...")
        
        # Predicciones
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # Métricas train y validation
        train_metrics = self.calculate_regression_metrics(y_train, y_train_pred)
        val_metrics = self.calculate_regression_metrics(y_val, y_val_pred)
        
        self.evaluation_results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'predictions': {
                'train': y_train_pred,
                'val': y_val_pred
            },
            'actual': {
                'train': y_train,
                'val': y_val
            }
        }
        
        # Test set si está disponible
        if X_test is not None and y_test is not None:
            y_test_pred = self.model.predict(X_test)
            test_metrics = self.calculate_regression_metrics(y_test, y_test_pred)
            
            self.evaluation_results['test_metrics'] = test_metrics
            self.evaluation_results['predictions']['test'] = y_test_pred
            self.evaluation_results['actual']['test'] = y_test
        
        logger.info(f"Evaluación completada. R² validación: {val_metrics['r2']:.4f}")
        return self.evaluation_results
    
    def plot_predictions_vs_actual(self, dataset='val', save_path=None):
        """Visualizar predicciones vs valores reales"""
        if dataset not in self.evaluation_results['predictions']:
            raise ValueError(f"Dataset '{dataset}' no encontrado en resultados")
        
        y_true = self.evaluation_results['actual'][dataset]
        y_pred = self.evaluation_results['predictions'][dataset]
        metrics = self.evaluation_results[f'{dataset}_metrics']
        
        # Crear figura
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predicciones',
            opacity=0.6,
            marker=dict(
                color=y_true,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Valor Real")
            )
        ))
        
        # Línea diagonal perfecta
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Predicción Perfecta',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        # Configurar layout
        fig.update_layout(
            title=f'{self.model_name} - Predicciones vs Valores Reales ({dataset.title()})<br>'
                  f'R² = {metrics["r2"]:.4f}, RMSE = {metrics["rmse"]:.4f}',
            xaxis_title='Valores Reales',
            yaxis_title='Predicciones',
            showlegend=True,
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Gráfico guardado en: {save_path}")
        
        return fig
    
    def plot_residuals_analysis(self, dataset='val', save_path=None):
        """Análisis completo de residuos"""
        if dataset not in self.evaluation_results['predictions']:
            raise ValueError(f"Dataset '{dataset}' no encontrado en resultados")
        
        y_true = self.evaluation_results['actual'][dataset]
        y_pred = self.evaluation_results['predictions'][dataset]
        residuals = y_pred - y_true
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Residuos vs Predicciones',
                'Distribución de Residuos',
                'Q-Q Plot de Residuos',
                'Residuos vs Orden'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Residuos vs Predicciones
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuos',
                      opacity=0.6),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. Histograma de residuos
        fig.add_trace(
            go.Histogram(x=residuals, name='Distribución', nbinsx=30),
            row=1, col=2
        )
        
        # 3. Q-Q Plot aproximado
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q Plot'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', 
                      name='Línea Teórica', line=dict(color='red')),
            row=2, col=1
        )
        
        # 4. Residuos vs Orden (para detectar patrones)
        fig.add_trace(
            go.Scatter(x=list(range(len(residuals))), y=residuals, 
                      mode='markers', name='Orden'),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
        
        # Configurar layout
        fig.update_layout(
            title=f'{self.model_name} - Análisis de Residuos ({dataset.title()})',
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Análisis de residuos guardado en: {save_path}")
        
        return fig
    
    def plot_feature_importance(self, feature_names, save_path=None):
        """Visualizar importancia de features (si el modelo lo soporta)"""
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("El modelo no tiene feature_importances_")
            return None
        
        importance = self.model.feature_importances_
        
        # Crear DataFrame para ordenar
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Crear gráfico
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker=dict(color=importance_df['importance'], colorscale='viridis')
        ))
        
        fig.update_layout(
            title=f'{self.model_name} - Importancia de Features',
            xaxis_title='Importancia',
            yaxis_title='Features',
            height=max(400, len(feature_names) * 20)
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Importancia de features guardada en: {save_path}")
        
        return fig, importance_df
    
    def cross_validation_analysis(self, X, y, cv=5, scoring='r2'):
        """Análisis de validación cruzada"""
        logger.info(f"Ejecutando validación cruzada ({cv}-fold)...")
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        cv_results = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'confidence_interval': (
                cv_scores.mean() - 1.96 * cv_scores.std() / np.sqrt(cv),
                cv_scores.mean() + 1.96 * cv_scores.std() / np.sqrt(cv)
            )
        }
        
        logger.info(f"CV {scoring}: {cv_results['mean']:.4f} (+/- {cv_results['std'] * 2:.4f})")
        
        return cv_results
    
    def learning_curve_analysis(self, X, y, cv=5, n_jobs=-1):
        """Análisis de curvas de aprendizaje"""
        logger.info("Generando curvas de aprendizaje...")
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            self.model, X, y, cv=cv, n_jobs=n_jobs,
            train_sizes=train_sizes, scoring='r2'
        )
        
        # Calcular estadísticas
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        learning_results = {
            'train_sizes': train_sizes_abs,
            'train_scores': {
                'mean': train_mean,
                'std': train_std,
                'raw': train_scores
            },
            'val_scores': {
                'mean': val_mean,
                'std': val_std,
                'raw': val_scores
            }
        }
        
        return learning_results
    
    def plot_learning_curves(self, learning_results, save_path=None):
        """Visualizar curvas de aprendizaje"""
        train_sizes = learning_results['train_sizes']
        train_mean = learning_results['train_scores']['mean']
        train_std = learning_results['train_scores']['std']
        val_mean = learning_results['val_scores']['mean']
        val_std = learning_results['val_scores']['std']
        
        fig = go.Figure()
        
        # Training curves
        fig.add_trace(go.Scatter(
            x=train_sizes, y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue'),
            error_y=dict(type='data', array=train_std, visible=True)
        ))
        
        # Validation curves
        fig.add_trace(go.Scatter(
            x=train_sizes, y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='red'),
            error_y=dict(type='data', array=val_std, visible=True)
        ))
        
        fig.update_layout(
            title=f'{self.model_name} - Curvas de Aprendizaje',
            xaxis_title='Tamaño del Conjunto de Entrenamiento',
            yaxis_title='R² Score',
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Curvas de aprendizaje guardadas en: {save_path}")
        
        return fig
    
    def validation_curve_analysis(self, X, y, param_name, param_range, cv=5):
        """Análisis de curvas de validación para un hiperparámetro"""
        logger.info(f"Generando curva de validación para {param_name}...")
        
        train_scores, val_scores = validation_curve(
            self.model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring='r2', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        return {
            'param_range': param_range,
            'train_scores': {'mean': train_mean, 'std': train_std},
            'val_scores': {'mean': val_mean, 'std': val_std}
        }
    
    def error_distribution_analysis(self, dataset='val'):
        """Análisis detallado de la distribución de errores"""
        if dataset not in self.evaluation_results['predictions']:
            raise ValueError(f"Dataset '{dataset}' no encontrado")
        
        y_true = self.evaluation_results['actual'][dataset]
        y_pred = self.evaluation_results['predictions'][dataset]
        
        # Calcular diferentes tipos de errores
        absolute_errors = np.abs(y_pred - y_true)
        relative_errors = np.abs((y_pred - y_true) / y_true) * 100
        squared_errors = (y_pred - y_true) ** 2
        
        error_analysis = {
            'absolute_errors': {
                'mean': np.mean(absolute_errors),
                'median': np.median(absolute_errors),
                'q95': np.percentile(absolute_errors, 95),
                'max': np.max(absolute_errors)
            },
            'relative_errors': {
                'mean': np.mean(relative_errors),
                'median': np.median(relative_errors),
                'q95': np.percentile(relative_errors, 95),
                'max': np.max(relative_errors)
            },
            'error_percentiles': {
                'p10': np.percentile(absolute_errors, 10),
                'p25': np.percentile(absolute_errors, 25),
                'p50': np.percentile(absolute_errors, 50),
                'p75': np.percentile(absolute_errors, 75),
                'p90': np.percentile(absolute_errors, 90),
                'p95': np.percentile(absolute_errors, 95),
                'p99': np.percentile(absolute_errors, 99)
            }
        }
        
        return error_analysis
    
    def generate_evaluation_report(self, output_dir="reports/model_evaluation"):
        """Generar reporte completo de evaluación"""
        logger.info(f"Generando reporte completo para {self.model_name}...")
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.model_name.replace(' ', '_')}_{timestamp}"
        
        report = {
            'model_name': self.model_name,
            'timestamp': timestamp,
            'evaluation_results': self.evaluation_results
        }
        
        # Guardar métricas en JSON
        metrics_path = os.path.join(output_dir, f"{base_filename}_metrics.json")
        with open(metrics_path, 'w') as f:
            # Convertir arrays numpy a listas para JSON
            metrics_json = {}
            for key, value in self.evaluation_results.items():
                if key in ['predictions', 'actual']:
                    metrics_json[key] = {k: v.tolist() if hasattr(v, 'tolist') else v 
                                       for k, v in value.items()}
                else:
                    metrics_json[key] = value
            
            json.dump(metrics_json, f, indent=2)
        
        logger.info(f"Reporte de evaluación guardado en: {output_dir}")
        return report
    
    def compare_predictions_by_range(self, dataset='val', n_bins=5):
        """Comparar performance por rangos de valores objetivo"""
        if dataset not in self.evaluation_results['predictions']:
            raise ValueError(f"Dataset '{dataset}' no encontrado")
        
        y_true = self.evaluation_results['actual'][dataset]
        y_pred = self.evaluation_results['predictions'][dataset]
        
        # Crear bins
        bins = pd.qcut(y_true, q=n_bins, labels=[f"Q{i+1}" for i in range(n_bins)])
        
        # Calcular métricas por bin
        comparison_results = {}
        for bin_label in bins.cat.categories:
            mask = bins == bin_label
            bin_true = y_true[mask]
            bin_pred = y_pred[mask]
            
            if len(bin_true) > 0:
                comparison_results[bin_label] = self.calculate_regression_metrics(bin_true, bin_pred)
                comparison_results[bin_label]['count'] = len(bin_true)
                comparison_results[bin_label]['range'] = f"[{bin_true.min():.2f}, {bin_true.max():.2f}]"
        
        return comparison_results


def compare_models(models_dict, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Comparar múltiples modelos"""
    logger.info("Comparando múltiples modelos...")
    
    comparison_results = {}
    
    for name, model in models_dict.items():
        evaluator = ModelEvaluator(model, name)
        results = evaluator.evaluate_on_sets(X_train, y_train, X_val, y_val, X_test, y_test)
        comparison_results[name] = results
    
    # Crear DataFrame de comparación
    metrics_comparison = {}
    for model_name, results in comparison_results.items():
        metrics_comparison[model_name] = results['val_metrics']
    
    comparison_df = pd.DataFrame(metrics_comparison).T
    comparison_df = comparison_df.round(4)
    
    logger.info("Comparación de modelos completada")
    return comparison_results, comparison_df


def main():
    """Función principal para demostrar uso"""
    # Este es un ejemplo de uso del evaluador
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    
    # Cargar datos de ejemplo
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    evaluator = ModelEvaluator(model, "Random Forest Demo")
    results = evaluator.evaluate_on_sets(X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("Evaluación completada!")
    print(f"R² Validación: {results['val_metrics']['r2']:.4f}")
    print(f"R² Test: {results['test_metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()