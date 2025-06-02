"""
Módulo para entrenamiento de modelos con MLflow
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowModelTrainer:
    """Clase para entrenamiento de modelos con seguimiento MLflow"""
    
    def __init__(self, experiment_name="house_price_prediction"):
        self.experiment_name = experiment_name
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Configurar MLflow"""
        # Configurar tracking URI (local)
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Crear o usar experimento existente
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Experimento creado: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Usando experimento existente: {self.experiment_name}")
                
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.error(f"Error configurando MLflow: {e}")
            raise
    
    def calculate_metrics(self, y_true, y_pred):
        """Calcular métricas de evaluación"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def train_linear_regression(self, X_train, y_train, X_val, y_val):
        """Entrenar modelo de Regresión Lineal"""
        with mlflow.start_run(run_name="Linear_Regression"):
            # Parámetros
            model = LinearRegression()
            
            # Log parámetros
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("fit_intercept", True)
            
            # Entrenar modelo
            logger.info("Entrenando Regresión Lineal...")
            model.fit(X_train, y_train)
            
            # Predicciones
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Métricas training
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            
            # Métricas validation
            val_metrics = self.calculate_metrics(y_val, y_val_pred)
            for metric, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric}", value)
            
            # Log modelo
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Linear Regression - Val R²: {val_metrics['r2']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")
            
            return model, val_metrics
    
    def train_ridge_regression(self, X_train, y_train, X_val, y_val, alpha=1.0):
        """Entrenar modelo Ridge Regression"""
        with mlflow.start_run(run_name=f"Ridge_Regression_alpha_{alpha}"):
            # Parámetros
            model = Ridge(alpha=alpha, random_state=42)
            
            # Log parámetros
            mlflow.log_param("model_type", "Ridge")
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("random_state", 42)
            
            # Entrenar modelo
            logger.info(f"Entrenando Ridge Regression (alpha={alpha})...")
            model.fit(X_train, y_train)
            
            # Predicciones
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Métricas training
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            
            # Métricas validation
            val_metrics = self.calculate_metrics(y_val, y_val_pred)
            for metric, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric}", value)
            
            # Log modelo
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Ridge Regression - Val R²: {val_metrics['r2']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")
            
            return model, val_metrics
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, 
                           n_estimators=100, max_depth=10):
        """Entrenar modelo Random Forest"""
        with mlflow.start_run(run_name=f"RandomForest_n{n_estimators}_d{max_depth}"):
            # Parámetros
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            
            # Log parámetros
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("random_state", 42)
            
            # Entrenar modelo
            logger.info(f"Entrenando Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
            model.fit(X_train, y_train)
            
            # Predicciones
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Métricas training
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            
            # Métricas validation
            val_metrics = self.calculate_metrics(y_val, y_val_pred)
            for metric, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric}", value)
            
            # Feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            mlflow.log_dict(feature_importance, "feature_importance.json")
            
            # Log modelo
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Random Forest - Val R²: {val_metrics['r2']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")
            
            return model, val_metrics
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val,
                               n_estimators=100, learning_rate=0.1, max_depth=6):
        """Entrenar modelo Gradient Boosting"""
        with mlflow.start_run(run_name=f"GradientBoosting_n{n_estimators}_lr{learning_rate}_d{max_depth}"):
            # Parámetros
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
            
            # Log parámetros
            mlflow.log_param("model_type", "GradientBoosting")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("random_state", 42)
            
            # Entrenar modelo
            logger.info(f"Entrenando Gradient Boosting...")
            model.fit(X_train, y_train)
            
            # Predicciones
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Métricas training
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            
            # Métricas validation
            val_metrics = self.calculate_metrics(y_val, y_val_pred)
            for metric, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric}", value)
            
            # Feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            mlflow.log_dict(feature_importance, "feature_importance.json")
            
            # Log modelo
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Gradient Boosting - Val R²: {val_metrics['r2']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")
            
            return model, val_metrics
    
    def run_experiment_suite(self, X_train, y_train, X_val, y_val):
        """Ejecutar suite completa de experimentos"""
        logger.info("Iniciando suite de experimentos...")
        
        results = {}
        
        # 1. Linear Regression
        model_lr, metrics_lr = self.train_linear_regression(X_train, y_train, X_val, y_val)
        results['LinearRegression'] = {'model': model_lr, 'metrics': metrics_lr}
        
        # 2. Ridge Regression con diferentes alphas
        for alpha in [0.1, 1.0, 10.0]:
            model_ridge, metrics_ridge = self.train_ridge_regression(
                X_train, y_train, X_val, y_val, alpha=alpha
            )
            results[f'Ridge_alpha_{alpha}'] = {'model': model_ridge, 'metrics': metrics_ridge}
        
        # 3. Random Forest con diferentes configuraciones
        rf_configs = [
            {'n_estimators': 50, 'max_depth': 8},
            {'n_estimators': 100, 'max_depth': 10},
            {'n_estimators': 200, 'max_depth': 12}
        ]
        
        for config in rf_configs:
            model_rf, metrics_rf = self.train_random_forest(
                X_train, y_train, X_val, y_val, **config
            )
            results[f"RF_n{config['n_estimators']}_d{config['max_depth']}"] = {
                'model': model_rf, 'metrics': metrics_rf
            }
        
        # 4. Gradient Boosting
        gb_configs = [
            {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 6},
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
            {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 8}
        ]
        
        for config in gb_configs:
            model_gb, metrics_gb = self.train_gradient_boosting(
                X_train, y_train, X_val, y_val, **config
            )
            results[f"GB_n{config['n_estimators']}_lr{config['learning_rate']}_d{config['max_depth']}"] = {
                'model': model_gb, 'metrics': metrics_gb
            }
        
        return results
    
    def get_best_model(self, results, metric='r2'):
        """Obtener el mejor modelo basado en una métrica"""
        best_score = -float('inf') if metric == 'r2' else float('inf')
        best_model_name = None
        best_model = None
        
        for name, result in results.items():
            score = result['metrics'][f'val_{metric}']
            
            if metric == 'r2':
                if score > best_score:
                    best_score = score
                    best_model_name = name
                    best_model = result
            else:  # Para métricas donde menor es mejor (mae, rmse, mape)
                if score < best_score:
                    best_score = score
                    best_model_name = name
                    best_model = result
        
        logger.info(f"Mejor modelo: {best_model_name} ({metric}: {best_score:.4f})")
        return best_model_name, best_model
    
    def register_best_model(self, best_model_name, best_model, model_name="house_price_model"):
        """Registrar mejor modelo en MLflow Model Registry"""
        try:
            # Registrar modelo
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            registered_model = mlflow.register_model(model_uri, model_name)
            
            logger.info(f"Modelo {best_model_name} registrado como {model_name}")
            return registered_model
            
        except Exception as e:
            logger.error(f"Error registrando modelo: {e}")
            return None


def main():
    """Función principal para entrenamiento"""
    # Cargar datos procesados
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/validation.csv")
    
    X_train = train_df.drop('MedHouseVal', axis=1)
    y_train = train_df['MedHouseVal']
    X_val = val_df.drop('MedHouseVal', axis=1)
    y_val = val_df['MedHouseVal']
    
    # Crear trainer
    trainer = MLflowModelTrainer()
    
    # Ejecutar experimentos
    results = trainer.run_experiment_suite(X_train, y_train, X_val, y_val)
    
    # Obtener mejor modelo
    best_name, best_model = trainer.get_best_model(results, metric='r2')
    
    # Registrar mejor modelo
    trainer.register_best_model(best_name, best_model)
    
    logger.info("Entrenamiento completado!")
    return trainer, results, best_name, best_model


if __name__ == "__main__":
    main()