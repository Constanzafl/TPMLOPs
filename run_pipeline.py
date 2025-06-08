"""
Pipeline MLOps completo - Integra MLflow + Evidently
Procesamiento → Entrenamiento → Monitoreo
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompletMLOpsPipeline:
    """Pipeline MLOps completo con MLflow y Evidently"""
    
    def __init__(self):
        self.pipeline_results = {}
        self.setup_environment()
        
    def setup_environment(self):
        """Configurar entorno y directorios"""
        directories = [
            'data/raw', 'data/processed', 'models', 'reports',
            'reports/data_quality', 'reports/data_drift', 'reports/model_performance',
            'mlruns'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("✅ Entorno configurado")

    def process_data(self):
        """Fase 1: Procesamiento de datos"""
        logger.info("=== FASE 1: PROCESAMIENTO DE DATOS ===")
        
        try:
            from sklearn.datasets import fetch_california_housing
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            # 1. Cargar datos
            logger.info("📁 Cargando California Housing Dataset...")
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame
            logger.info(f"✅ Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
            
            # 2. Feature engineering
            logger.info("🔧 Feature engineering...")
            df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
            df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
            df['PopulationPerHousehold'] = df['Population'] / df['HouseAge']
            
            # Categorizar edad
            df['HouseAgeCategory'] = pd.cut(df['HouseAge'], 
                                           bins=[0, 10, 20, 35, np.inf],
                                           labels=['New', 'Recent', 'Old', 'Very_Old'])
            df = pd.get_dummies(df, columns=['HouseAgeCategory'], prefix='Age')
            
            # 3. Limpiar outliers
            logger.info("🧹 Limpiando outliers...")
            numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                             if col != 'MedHouseVal']
            
            for col in numeric_columns:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                before = df.shape[0]
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                if before != df.shape[0]:
                    logger.info(f"   {col}: {before - df.shape[0]} outliers removidos")
            
            # 4. Dividir datos
            logger.info("📊 Dividiendo datos...")
            X, y = df.drop('MedHouseVal', axis=1), df['MedHouseVal']
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
            
            # 5. Escalar features
            logger.info("⚖️ Escalando features...")
            scaler = StandardScaler()
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            
            X_train_scaled, X_val_scaled, X_test_scaled = X_train.copy(), X_val.copy(), X_test.copy()
            X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
            X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
            
            # 6. Guardar datos
            logger.info("💾 Guardando datos procesados...")
            pd.concat([X_train_scaled, y_train], axis=1).to_csv('data/processed/train.csv', index=False)
            pd.concat([X_val_scaled, y_val], axis=1).to_csv('data/processed/validation.csv', index=False)
            pd.concat([X_test_scaled, y_test], axis=1).to_csv('data/processed/test.csv', index=False)
            
            self.pipeline_results['data_processing'] = {
                'status': 'success',
                'train_shape': X_train_scaled.shape,
                'val_shape': X_val_scaled.shape,
                'test_shape': X_test_scaled.shape,
                'features': X_train_scaled.shape[1]
            }
            
            logger.info("✅ Procesamiento completado")
            return True, (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento: {e}")
            return False, None

    def train_models(self):
        """Fase 2: Entrenamiento con MLflow"""
        logger.info("=== FASE 2: ENTRENAMIENTO CON MLFLOW ===")
        
        try:
            import mlflow
            import mlflow.sklearn
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression, Ridge
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            # Configurar MLflow
            mlflow.set_tracking_uri("file:./mlruns")
            experiment_name = "house_price_prediction_complete"
            
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                logger.info(f"✅ MLflow configurado: {experiment_name}")
            except Exception as e:
                logger.error(f"❌ Error MLflow: {e}")
                return False, None
            
            # Cargar datos
            train_df = pd.read_csv('data/processed/train.csv')
            val_df = pd.read_csv('data/processed/validation.csv')
            
            X_train = train_df.drop('MedHouseVal', axis=1)
            y_train = train_df['MedHouseVal']
            X_val = val_df.drop('MedHouseVal', axis=1)
            y_val = val_df['MedHouseVal']
            
            def calc_metrics(y_true, y_pred):
                return {
                    'mae': mean_absolute_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'r2': r2_score(y_true, y_pred),
                    'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                }
            
            results = {}
            best_model = None
            best_score = -float('inf')
            
            # 1. Linear Regression
            logger.info("🔵 Linear Regression...")
            with mlflow.start_run(run_name="Linear_Regression"):
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                metrics = calc_metrics(y_val, y_pred)
                
                mlflow.log_param("model_type", "LinearRegression")
                for k, v in metrics.items():
                    mlflow.log_metric(f"val_{k}", v)
                mlflow.sklearn.log_model(model, "model")
                
                results['Linear_Regression'] = metrics
                if metrics['r2'] > best_score:
                    best_score, best_model = metrics['r2'], model
                logger.info(f"   R² = {metrics['r2']:.4f}")
            
            # 2. Ridge Regression
            logger.info("🟡 Ridge Regression...")
            for alpha in [0.1, 1.0, 10.0]:
                with mlflow.start_run(run_name=f"Ridge_alpha_{alpha}"):
                    model = Ridge(alpha=alpha, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    metrics = calc_metrics(y_val, y_pred)
                    
                    mlflow.log_param("model_type", "Ridge")
                    mlflow.log_param("alpha", alpha)
                    for k, v in metrics.items():
                        mlflow.log_metric(f"val_{k}", v)
                    mlflow.sklearn.log_model(model, "model")
                    
                    results[f'Ridge_alpha_{alpha}'] = metrics
                    if metrics['r2'] > best_score:
                        best_score, best_model = metrics['r2'], model
                    logger.info(f"   Alpha {alpha}: R² = {metrics['r2']:.4f}")
            
            # 3. Random Forest
            logger.info("🟢 Random Forest...")
            rf_configs = [
                {'n_estimators': 50, 'max_depth': 8},
                {'n_estimators': 100, 'max_depth': 10},
                {'n_estimators': 150, 'max_depth': 12}
            ]
            
            for config in rf_configs:
                name = f"RF_n{config['n_estimators']}_d{config['max_depth']}"
                with mlflow.start_run(run_name=name):
                    model = RandomForestRegressor(**config, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    metrics = calc_metrics(y_val, y_pred)
                    
                    mlflow.log_param("model_type", "RandomForest")
                    for k, v in config.items():
                        mlflow.log_param(k, v)
                    for k, v in metrics.items():
                        mlflow.log_metric(f"val_{k}", v)
                    
                    # Feature importance
                    importance = dict(zip(X_train.columns, model.feature_importances_))
                    mlflow.log_dict(importance, "feature_importance.json")
                    mlflow.sklearn.log_model(model, "model")
                    
                    results[name] = metrics
                    if metrics['r2'] > best_score:
                        best_score, best_model = metrics['r2'], model
                    logger.info(f"   {name}: R² = {metrics['r2']:.4f}")
            
            # 4. Gradient Boosting
            logger.info("🟣 Gradient Boosting...")
            gb_configs = [
                {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
                {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 8}
            ]
            
            for config in gb_configs:
                name = f"GB_n{config['n_estimators']}_lr{config['learning_rate']}_d{config['max_depth']}"
                with mlflow.start_run(run_name=name):
                    model = GradientBoostingRegressor(**config, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    metrics = calc_metrics(y_val, y_pred)
                    
                    mlflow.log_param("model_type", "GradientBoosting")
                    for k, v in config.items():
                        mlflow.log_param(k, v)
                    for k, v in metrics.items():
                        mlflow.log_metric(f"val_{k}", v)
                    mlflow.sklearn.log_model(model, "model")
                    
                    results[name] = metrics
                    if metrics['r2'] > best_score:
                        best_score, best_model = metrics['r2'], model
                    logger.info(f"   {name}: R² = {metrics['r2']:.4f}")
            
            # Mejor modelo
            best_name = max(results.keys(), key=lambda x: results[x]['r2'])
            logger.info(f"🏆 Mejor modelo: {best_name} (R² = {best_score:.4f})")
            
            self.pipeline_results['training'] = {
                'status': 'success',
                'experiments': len(results),
                'best_model': best_name,
                'best_score': best_score,
                'all_results': results
            }
            
            logger.info("✅ Entrenamiento completado")
            return True, best_model
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            return False, None

    def monitor_with_evidently(self, model):
        """Fase 3: Monitoreo con Evidently"""
        logger.info("=== FASE 3: MONITOREO CON EVIDENTLY ===")
        
        try:
            # Cargar datos
            train_df = pd.read_csv('data/processed/train.csv')
            test_df = pd.read_csv('data/processed/test.csv')
            
            # 1. Reportes de calidad de datos
            logger.info("📊 Generando reportes de calidad...")
            
            # Reporte de calidad training
            train_quality = self._generate_quality_report(train_df, "training_quality")
            
            # Reporte de calidad test
            test_quality = self._generate_quality_report(test_df, "test_quality")
            
            # 2. Simulación y detección de deriva
            logger.info("📈 Simulando deriva de datos...")
            drifted_df = self._simulate_data_drift(test_df)
            
            # Reporte de deriva
            drift_report = self._generate_drift_report(test_df, drifted_df, "data_drift")
            
            # 3. Monitoreo de performance del modelo
            logger.info("🤖 Monitoreando performance del modelo...")
            performance_report = self._generate_performance_report(model, test_df, drifted_df)
            
            reports = {
                'train_quality': train_quality,
                'test_quality': test_quality,
                'data_drift': drift_report,
                'model_performance': performance_report
            }
            
            self.pipeline_results['monitoring'] = {
                'status': 'success',
                'reports': reports,
                'reports_count': len([r for r in reports.values() if r])
            }
            
            logger.info("✅ Monitoreo completado")
            return True, reports
            
        except Exception as e:
            logger.error(f"❌ Error en monitoreo: {e}")
            return False, None

    def _generate_quality_report(self, df, name):
        """Generar reporte de calidad de datos"""
        try:
            # Intentar usar Evidently avanzado
            from evidently.report import Report
            from evidently.metrics import DatasetSummaryMetric, ColumnSummaryMetric
            
            metrics = [
                DatasetSummaryMetric(),
                ColumnSummaryMetric(column_name='MedHouseVal')
            ]
            
            report = Report(metrics=metrics)
            report.run(reference_data=df)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"reports/data_quality/{name}_{timestamp}.html"
            report.save_html(path)
            
            logger.info(f"   ✅ {name}: {path}")
            return path
            
        except ImportError:
            # Fallback a reporte básico
            return self._generate_basic_quality_report(df, name)

    def _generate_basic_quality_report(self, df, name):
        """Reporte básico de calidad"""
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing': df.isnull().sum().sum(),
            'target_mean': df['MedHouseVal'].mean(),
            'target_std': df['MedHouseVal'].std()
        }
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Reporte de Calidad - {name}</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .good {{ border-left: 5px solid #4CAF50; }}
        </style>
        </head>
        <body>
            <h1>📊 Reporte de Calidad - {name}</h1>
            <div class="metric good">
                <h2>📈 Dataset Info</h2>
                <p><strong>Filas:</strong> {stats['rows']:,}</p>
                <p><strong>Columnas:</strong> {stats['columns']}</p>
                <p><strong>Valores faltantes:</strong> {stats['missing']}</p>
            </div>
            <div class="metric good">
                <h2>🎯 Target Statistics</h2>
                <p><strong>Media:</strong> {stats['target_mean']:.4f}</p>
                <p><strong>Desviación:</strong> {stats['target_std']:.4f}</p>
            </div>
        </body>
        </html>
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"reports/data_quality/{name}_basic_{timestamp}.html"
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"   ✅ {name} (básico): {path}")
        return path

    def _simulate_data_drift(self, df, drift=0.15):
        """Simular deriva de datos"""
        drifted = df.copy()
        numeric_cols = [col for col in drifted.select_dtypes(include=[np.number]).columns 
                       if col != 'MedHouseVal']
        
        for col in numeric_cols[:3]:  # Solo primeras 3 columnas
            noise = np.random.normal(0, drifted[col].std() * drift, len(drifted))
            mean_shift = drifted[col].mean() * drift
            drifted[col] = drifted[col] + noise + mean_shift
        
        return drifted

    def _generate_drift_report(self, reference_df, current_df, name):
        """Generar reporte de deriva"""
        try:
            from evidently.report import Report
            from evidently.metrics import DataDriftPreset
            
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"reports/data_drift/{name}_{timestamp}.html"
            report.save_html(path)
            
            logger.info(f"   ✅ Deriva: {path}")
            return path
            
        except ImportError:
            return self._generate_basic_drift_report(reference_df, current_df, name)

    def _generate_basic_drift_report(self, ref_df, cur_df, name):
        """Reporte básico de deriva"""
        numeric_cols = [col for col in ref_df.select_dtypes(include=[np.number]).columns 
                       if col != 'MedHouseVal']
        
        drift_analysis = {}
        for col in numeric_cols:
            ref_mean, cur_mean = ref_df[col].mean(), cur_df[col].mean()
            ref_std = ref_df[col].std()
            drift_score = abs(cur_mean - ref_mean) / ref_std if ref_std > 0 else 0
            
            drift_analysis[col] = {
                'ref_mean': ref_mean,
                'cur_mean': cur_mean,
                'drift_score': drift_score,
                'drift_detected': drift_score > 0.1
            }
        
        total_drifted = sum(1 for a in drift_analysis.values() if a['drift_detected'])
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Reporte de Deriva - {name}</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .good {{ border-left: 5px solid #4CAF50; }}
            .warning {{ border-left: 5px solid #FF9800; }}
            .error {{ border-left: 5px solid #F44336; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
        </style>
        </head>
        <body>
            <h1>📈 Reporte de Deriva - {name}</h1>
            <div class="metric {'error' if total_drifted > 0 else 'good'}">
                <h2>🚨 Resumen de Deriva</h2>
                <p><strong>Columnas con deriva:</strong> {total_drifted}/{len(drift_analysis)}</p>
            </div>
            <div class="metric good">
                <h2>📊 Análisis por Columna</h2>
                <table>
                    <tr><th>Columna</th><th>Media Ref</th><th>Media Actual</th><th>Score</th><th>Estado</th></tr>
        """
        
        for col, analysis in drift_analysis.items():
            status = "🚨 DERIVA" if analysis['drift_detected'] else "✅ OK"
            html += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{analysis['ref_mean']:.4f}</td>
                        <td>{analysis['cur_mean']:.4f}</td>
                        <td>{analysis['drift_score']:.4f}</td>
                        <td>{status}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"reports/data_drift/{name}_basic_{timestamp}.html"
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"   ✅ Deriva (básico): {path}")
        return path

    def _generate_performance_report(self, model, test_df, drifted_df):
        """Generar reporte de performance"""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        X_test = test_df.drop('MedHouseVal', axis=1)
        y_test = test_df['MedHouseVal']
        X_drifted = drifted_df.drop('MedHouseVal', axis=1)
        y_drifted = drifted_df['MedHouseVal']
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_drifted = model.predict(X_drifted)
        
        # Métricas
        base_r2 = r2_score(y_test, y_pred)
        base_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        drifted_r2 = r2_score(y_drifted, y_pred_drifted)
        drifted_rmse = np.sqrt(mean_squared_error(y_drifted, y_pred_drifted))
        
        degradation = abs(drifted_r2 - base_r2) > 0.05
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Monitoreo de Performance</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .good {{ border-left: 5px solid #4CAF50; }}
            .error {{ border-left: 5px solid #F44336; }}
        </style>
        </head>
        <body>
            <h1>🤖 Monitoreo de Performance</h1>
            <div class="metric good">
                <h2>📊 Performance Base</h2>
                <p><strong>R²:</strong> {base_r2:.4f}</p>
                <p><strong>RMSE:</strong> {base_rmse:.4f}</p>
            </div>
            <div class="metric {'error' if degradation else 'good'}">
                <h2>📉 Performance con Deriva</h2>
                <p><strong>R²:</strong> {drifted_r2:.4f} ({drifted_r2 - base_r2:+.4f})</p>
                <p><strong>RMSE:</strong> {drifted_rmse:.4f} ({drifted_rmse - base_rmse:+.4f})</p>
            </div>
            <div class="metric {'error' if degradation else 'good'}">
                <h2>🚨 Alertas</h2>
                {"<p>⚠️ DEGRADACIÓN DETECTADA</p>" if degradation else "<p>✅ Performance estable</p>"}
            </div>
        </body>
        </html>
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"reports/model_performance/performance_{timestamp}.html"
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"   ✅ Performance: {path}")
        return path

    def run_complete_pipeline(self):
        """Ejecutar pipeline completo"""
        logger.info("🚀 INICIANDO PIPELINE MLOPS COMPLETO")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Fase 1: Procesamiento
        success1, _ = self.process_data()
        if not success1:
            logger.error("❌ Pipeline falló en procesamiento")
            return False
        
        # Fase 2: Entrenamiento
        success2, best_model = self.train_models()
        if not success2:
            logger.error("❌ Pipeline falló en entrenamiento")
            return False
        
        # Fase 3: Monitoreo
        success3, reports = self.monitor_with_evidently(best_model)
        if not success3:
            logger.error("❌ Pipeline falló en monitoreo")
            return False
        
        # Resumen final
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 PIPELINE MLOPS COMPLETO EJECUTADO")
        logger.info(f"⏱️  Tiempo total: {execution_time}")
        logger.info(f"🤖 Experimentos MLflow: {self.pipeline_results['training']['experiments']}")
        logger.info(f"🏆 Mejor modelo: {self.pipeline_results['training']['best_model']}")
        logger.info(f"📊 R² mejor modelo: {self.pipeline_results['training']['best_score']:.4f}")
        logger.info(f"📈 Reportes Evidently: {self.pipeline_results['monitoring']['reports_count']}")
        logger.info("\n🔍 Para ver resultados:")
        logger.info("   📊 MLflow: mlflow ui")
        logger.info("   📈 Evidently: abrir .html en reports/")
        logger.info("=" * 60)
        
        return True


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Pipeline MLOps Completo")
    parser.add_argument("--phase", choices=['all', 'data', 'training', 'monitoring'], 
                       default='all', help="Fase a ejecutar")
    
    args = parser.parse_args()
    
    pipeline = CompletMLOpsPipeline()
    
    if args.phase == 'all':
        success = pipeline.run_complete_pipeline()
    elif args.phase == 'data':
        success, _ = pipeline.process_data()
    elif args.phase == 'training':
        success, _ = pipeline.train_models()
    elif args.phase == 'monitoring':
        # Cargar modelo dummy para monitoreo independiente
        from sklearn.ensemble import RandomForestRegressor
        import pandas as pd
        
        train_df = pd.read_csv('data/processed/train.csv')
        X_train = train_df.drop('MedHouseVal', axis=1)
        y_train = train_df['MedHouseVal']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        success, _ = pipeline.monitor_with_evidently(model)
    
    if success:
        logger.info("🎉 Ejecución exitosa!")
    else:
        logger.error("❌ Pipeline falló")


if __name__ == "__main__":
    main()