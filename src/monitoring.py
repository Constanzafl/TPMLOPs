"""
Módulo para monitoreo y observabilidad con Evidently
"""

import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.tests import *
import os
import logging
from datetime import datetime
import joblib

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Clase para monitoreo de modelos con Evidently"""
    
    def __init__(self, target_column='MedHouseVal'):
        self.target_column = target_column
        self.reports_dir = "reports"
        self.setup_directories()
        
    def setup_directories(self):
        """Crear directorios para reportes"""
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(f"{self.reports_dir}/data_quality", exist_ok=True)
        os.makedirs(f"{self.reports_dir}/data_drift", exist_ok=True)
        os.makedirs(f"{self.reports_dir}/model_performance", exist_ok=True)
        
    def setup_column_mapping(self, df):
        """Configurar mapeo de columnas para Evidently"""
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numerical_features:
            numerical_features.remove(self.target_column)
            
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        column_mapping = ColumnMapping()
        column_mapping.target = self.target_column
        column_mapping.numerical_features = numerical_features
        column_mapping.categorical_features = categorical_features
        
        return column_mapping
    
    def generate_data_quality_report(self, df, report_name="data_quality"):
        """Generar reporte de calidad de datos"""
        logger.info("Generando reporte de calidad de datos...")
        
        column_mapping = self.setup_column_mapping(df)
        
        # Crear reporte de calidad
        report = Report(metrics=[
            DataQualityPreset(),
            ColumnSummaryMetric(column_name=self.target_column),
            DatasetSummaryMetric(),
            ColumnMissingValuesMetric(column_name=self.target_column),
            DatasetMissingValuesMetric(),
            ColumnCorrelationsMetric(column_name=self.target_column),
        ])
        
        # Ejecutar reporte
        report.run(reference_data=df, column_mapping=column_mapping)
        
        # Guardar reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.reports_dir}/data_quality/{report_name}_{timestamp}.html"
        report.save_html(report_path)
        
        logger.info(f"Reporte de calidad guardado en: {report_path}")
        return report_path
    
    def generate_data_drift_report(self, reference_df, current_df, report_name="data_drift"):
        """Generar reporte de deriva de datos"""
        logger.info("Generando reporte de deriva de datos...")
        
        column_mapping = self.setup_column_mapping(reference_df)
        
        # Crear reporte de deriva
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            ColumnDriftMetric(column_name=self.target_column),
            DatasetDriftMetric(),
        ])
        
        # Ejecutar reporte
        report.run(
            reference_data=reference_df, 
            current_data=current_df,
            column_mapping=column_mapping
        )
        
        # Guardar reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.reports_dir}/data_drift/{report_name}_{timestamp}.html"
        report.save_html(report_path)
        
        logger.info(f"Reporte de deriva guardado en: {report_path}")
        return report_path
    
    def generate_model_performance_report(self, reference_df, current_df, 
                                        reference_predictions, current_predictions,
                                        report_name="model_performance"):
        """Generar reporte de performance del modelo"""
        logger.info("Generando reporte de performance del modelo...")
        
        # Agregar predicciones a los dataframes
        reference_data = reference_df.copy()
        current_data = current_df.copy()
        
        reference_data['prediction'] = reference_predictions
        current_data['prediction'] = current_predictions
        
        # Configurar mapeo incluyendo predicciones
        column_mapping = self.setup_column_mapping(reference_data)
        column_mapping.prediction = 'prediction'
        
        # Crear reporte de performance
        report = Report(metrics=[
            RegressionQualityMetric(),
            RegressionPredictedVsActualScatter(),
            RegressionPredictedVsActualPlot(),
            RegressionErrorPlot(),
            RegressionAbsPercentageErrorPlot(),
            RegressionErrorDistribution(),
            RegressionErrorNormality(),
        ])
        
        # Ejecutar reporte
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Guardar reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.reports_dir}/model_performance/{report_name}_{timestamp}.html"
        report.save_html(report_path)
        
        logger.info(f"Reporte de performance guardado en: {report_path}")
        return report_path
    
    def run_data_tests(self, reference_df, current_df, test_name="data_tests"):
        """Ejecutar tests automatizados de datos"""
        logger.info("Ejecutando tests de datos...")
        
        column_mapping = self.setup_column_mapping(reference_df)
        
        # Crear suite de tests
        test_suite = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestNumberOfDriftedColumns(),
            TestShareOfMissingValues(),
        ])
        
        # Ejecutar tests
        test_suite.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping
        )
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_path = f"{self.reports_dir}/{test_name}_{timestamp}.html"
        test_suite.save_html(test_path)
        
        logger.info(f"Tests de datos guardados en: {test_path}")
        return test_path, test_suite
    
    def run_model_tests(self, reference_df, current_df, 
                       reference_predictions, current_predictions,
                       test_name="model_tests"):
        """Ejecutar tests automatizados del modelo"""
        logger.info("Ejecutando tests del modelo...")
        
        # Agregar predicciones
        reference_data = reference_df.copy()
        current_data = current_df.copy()
        
        reference_data['prediction'] = reference_predictions
        current_data['prediction'] = current_predictions
        
        # Configurar mapeo
        column_mapping = self.setup_column_mapping(reference_data)
        column_mapping.prediction = 'prediction'
        
        # Crear suite de tests para modelo
        test_suite = TestSuite(tests=[
            TestValueMAE(),
            TestValueRMSE(),
            TestValueMeanError(),
            TestValueAbsMaxError(),
            TestShareOfOutRangeValues(),
            TestRegressionErrorStd(),
        ])
        
        # Ejecutar tests
        test_suite.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_path = f"{self.reports_dir}/{test_name}_{timestamp}.html"
        test_suite.save_html(test_path)
        
        logger.info(f"Tests del modelo guardados en: {test_path}")
        return test_path, test_suite
    
    def simulate_data_drift(self, df, drift_percentage=0.1):
        """Simular deriva de datos para demostración"""
        logger.info(f"Simulando deriva de datos ({drift_percentage*100}%)...")
        
        drifted_df = df.copy()
        
        # Seleccionar features numéricas para modificar
        numeric_cols = drifted_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        # Aplicar deriva a algunas columnas
        for col in numeric_cols[:3]:  # Solo a las primeras 3 columnas
            # Añadir ruido gaussiano
            noise = np.random.normal(0, drifted_df[col].std() * drift_percentage, len(drifted_df))
            drifted_df[col] = drifted_df[col] + noise
            
            # Cambiar la media ligeramente
            mean_shift = drifted_df[col].mean() * drift_percentage
            drifted_df[col] = drifted_df[col] + mean_shift
        
        logger.info("Deriva simulada aplicada")
        return drifted_df
    
    def generate_comprehensive_monitoring_report(self, train_df, test_df, model, 
                                               drift_simulation=True):
        """Generar reporte completo de monitoreo"""
        logger.info("Generando reporte completo de monitoreo...")
        
        reports = {}
        
        # 1. Reporte de calidad de datos de entrenamiento
        reports['train_quality'] = self.generate_data_quality_report(
            train_df, "train_data_quality"
        )
        
        # 2. Reporte de calidad de datos de test
        reports['test_quality'] = self.generate_data_quality_report(
            test_df, "test_data_quality"
        )
        
        # 3. Preparar datos para predicciones
        X_train = train_df.drop(self.target_column, axis=1)
        y_train = train_df[self.target_column]
        X_test = test_df.drop(self.target_column, axis=1)
        y_test = test_df[self.target_column]
        
        # Predicciones
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # 4. Reporte de performance del modelo
        reports['model_performance'] = self.generate_model_performance_report(
            train_df, test_df, train_predictions, test_predictions,
            "model_performance_evaluation"
        )
        
        # 5. Simular deriva de datos si se solicita
        if drift_simulation:
            drifted_test_df = self.simulate_data_drift(test_df, drift_percentage=0.15)
            drifted_predictions = model.predict(drifted_test_df.drop(self.target_column, axis=1))
            
            # Reporte de deriva
            reports['data_drift'] = self.generate_data_drift_report(
                test_df, drifted_test_df, "simulated_data_drift"
            )
            
            # Reporte de performance con deriva
            reports['drift_performance'] = self.generate_model_performance_report(
                test_df, drifted_test_df, test_predictions, drifted_predictions,
                "performance_with_drift"
            )
        
        # 6. Tests automatizados
        if drift_simulation:
            reports['data_tests'] = self.run_data_tests(
                test_df, drifted_test_df, "automated_data_tests"
            )[0]
            
            reports['model_tests'] = self.run_model_tests(
                test_df, drifted_test_df, test_predictions, drifted_predictions,
                "automated_model_tests"
            )[0]
        
        logger.info("Reporte completo de monitoreo generado")
        return reports
    
    def create_monitoring_summary(self, reports):
        """Crear resumen de monitoreo"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'reports_generated': len(reports),
            'report_paths': reports,
            'monitoring_status': 'completed'
        }
        
        # Guardar resumen
        import json
        summary_path = f"{self.reports_dir}/monitoring_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Resumen de monitoreo guardado en: {summary_path}")
        return summary


def main():
    """Función principal para monitoreo"""
    # Cargar datos
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # Cargar modelo (asumiendo que ya está entrenado)
    # En un escenario real, cargarías el modelo desde MLflow
    from sklearn.ensemble import RandomForestRegressor
    
    # Entrenar un modelo simple para demostración
    X_train = train_df.drop('MedHouseVal', axis=1)
    y_train = train_df['MedHouseVal']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Crear monitor
    monitor = ModelMonitor()
    
    # Generar reportes completos
    reports = monitor.generate_comprehensive_monitoring_report(
        train_df, test_df, model, drift_simulation=True
    )
    
    # Crear resumen
    summary = monitor.create_monitoring_summary(reports)
    
    logger.info("Monitoreo completado!")
    return monitor, reports, summary


if __name__ == "__main__":
    main()