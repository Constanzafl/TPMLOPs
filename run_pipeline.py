"""
Pipeline principal MLOps - Predicción de Precios de Viviendas
Integra procesamiento de datos, entrenamiento con MLflow y monitoreo con Evidently
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import pandas as pd

# Agregar src al path
sys.path.append('src')

from data_processing import DataProcessor
from model_training import MLflowModelTrainer
from monitoring import ModelMonitor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlops_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLOpsPipeline:
    """Pipeline completo MLOps para predicción de precios de viviendas"""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.trainer = MLflowModelTrainer()
        self.monitor = ModelMonitor()
        self.pipeline_results = {}
        
    def run_data_processing(self):
        """Ejecutar procesamiento de datos"""
        logger.info("=== FASE 1: PROCESAMIENTO DE DATOS ===")
        
        try:
            # 1. Cargar datos
            df = self.processor.load_data()
            
            # 2. Feature engineering
            df = self.processor.add_features(df)
            
            # 3. Limpiar datos
            df = self.processor.clean_data(df)
            
            # 4. Dividir datos
            X_train, X_val, X_test, y_train, y_val, y_test = self.processor.split_data(df)
            
            # 5. Escalar features
            X_train_scaled, X_val_scaled, X_test_scaled = self.processor.scale_features(
                X_train, X_val, X_test
            )
            
            # 6. Guardar datos procesados
            self.processor.save_processed_data(
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test
            )
            
            # 7. Obtener resumen
            summary = self.processor.get_data_summary(df)
            
            self.pipeline_results['data_processing'] = {
                'status': 'success',
                'summary': summary,
                'train_shape': X_train_scaled.shape,
                'val_shape': X_val_scaled.shape,
                'test_shape': X_test_scaled.shape
            }
            
            logger.info("✅ Procesamiento de datos completado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento de datos: {e}")
            self.pipeline_results['data_processing'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_model_training(self):
        """Ejecutar entrenamiento de modelos con MLflow"""
        logger.info("=== FASE 2: ENTRENAMIENTO DE MODELOS ===")
        
        try:
            # Cargar datos procesados
            train_df = pd.read_csv("data/processed/train.csv")
            val_df = pd.read_csv("data/processed/validation.csv")
            
            X_train = train_df.drop('MedHouseVal', axis=1)
            y_train = train_df['MedHouseVal']
            X_val = val_df.drop('MedHouseVal', axis=1)
            y_val = val_df['MedHouseVal']
            
            # Ejecutar experimentos
            results = self.trainer.run_experiment_suite(X_train, y_train, X_val, y_val)
            
            # Obtener mejor modelo
            best_name, best_model = self.trainer.get_best_model(results, metric='r2')
            
            # Registrar mejor modelo
            registered_model = self.trainer.register_best_model(best_name, best_model)
            
            self.pipeline_results['model_training'] = {
                'status': 'success',
                'total_experiments': len(results),
                'best_model': best_name,
                'best_metrics': best_model['metrics'],
                'registered_model': registered_model is not None
            }
            
            logger.info("✅ Entrenamiento de modelos completado exitosamente")
            logger.info(f"✅ Mejor modelo: {best_name}")
            logger.info(f"✅ R² de validación: {best_model['metrics']['val_r2']:.4f}")
            
            return True, best_model['model']
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento de modelos: {e}")
            self.pipeline_results['model_training'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False, None
    
    def run_monitoring(self, model):
        """Ejecutar monitoreo con Evidently"""
        logger.info("=== FASE 3: MONITOREO Y OBSERVABILIDAD ===")
        
        try:
            # Cargar datos
            train_df = pd.read_csv("data/processed/train.csv")
            test_df = pd.read_csv("data/processed/test.csv")
            
            # Generar reportes completos
            reports = self.monitor.generate_comprehensive_monitoring_report(
                train_df, test_df, model, drift_simulation=True
            )
            
            # Crear resumen
            summary = self.monitor.create_monitoring_summary(reports)
            
            self.pipeline_results['monitoring'] = {
                'status': 'success',
                'reports_generated': len(reports),
                'report_paths': reports,
                'summary': summary
            }
            
            logger.info("✅ Monitoreo completado exitosamente")
            logger.info(f"✅ Reportes generados: {len(reports)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en monitoreo: {e}")
            self.pipeline_results['monitoring'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_complete_pipeline(self):
        """Ejecutar pipeline completo"""
        logger.info("🚀 INICIANDO PIPELINE MLOPS COMPLETO")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Fase 1: Procesamiento de datos
        if not self.run_data_processing():
            logger.error("❌ Pipeline falló en procesamiento de datos")
            return False
        
        # Fase 2: Entrenamiento de modelos
        training_success, best_model = self.run_model_training()
        if not training_success:
            logger.error("❌ Pipeline falló en entrenamiento de modelos")
            return False
        
        # Fase 3: Monitoreo
        if not self.run_monitoring(best_model):
            logger.error("❌ Pipeline falló en monitoreo")
            return False
        
        # Resumen final
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        self.pipeline_results['pipeline_summary'] = {
            'status': 'success',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'execution_time_seconds': execution_time.total_seconds(),
            'phases_completed': 3
        }
        
        logger.info("=" * 60)
        logger.info("🎉 PIPELINE MLOPS COMPLETADO EXITOSAMENTE")
        logger.info(f"⏱️  Tiempo total de ejecución: {execution_time}")
        logger.info("=" * 60)
        
        # Mostrar resumen
        self.print_pipeline_summary()
        
        return True
    
    def print_pipeline_summary(self):
        """Imprimir resumen del pipeline"""
        print("\n" + "="*60)
        print("📊 RESUMEN DEL PIPELINE MLOPS")
        print("="*60)
        
        # Procesamiento de datos
        if 'data_processing' in self.pipeline_results:
            data_result = self.pipeline_results['data_processing']
            print(f"📁 Procesamiento de datos: {'✅ ÉXITO' if data_result['status'] == 'success' else '❌ FALLO'}")
            if data_result['status'] == 'success':
                print(f"   - Total de muestras: {data_result['summary']['total_samples']}")
                print(f"   - Features: {data_result['summary']['features']}")
                print(f"   - Train shape: {data_result['train_shape']}")
                print(f"   - Validation shape: {data_result['val_shape']}")
                print(f"   - Test shape: {data_result['test_shape']}")
        
        # Entrenamiento de modelos
        if 'model_training' in self.pipeline_results:
            train_result = self.pipeline_results['model_training']
            print(f"\n🤖 Entrenamiento de modelos: {'✅ ÉXITO' if train_result['status'] == 'success' else '❌ FALLO'}")
            if train_result['status'] == 'success':
                print(f"   - Experimentos ejecutados: {train_result['total_experiments']}")
                print(f"   - Mejor modelo: {train_result['best_model']}")
                print(f"   - R² de validación: {train_result['best_metrics']['val_r2']:.4f}")
                print(f"   - RMSE de validación: {train_result['best_metrics']['val_rmse']:.4f}")
        
        # Monitoreo
        if 'monitoring' in self.pipeline_results:
            monitor_result = self.pipeline_results['monitoring']
            print(f"\n📈 Monitoreo: {'✅ ÉXITO' if monitor_result['status'] == 'success' else '❌ FALLO'}")
            if monitor_result['status'] == 'success':
                print(f"   - Reportes generados: {monitor_result['reports_generated']}")
                print("   - Tipos de reportes:")
                for report_type in monitor_result['report_paths'].keys():
                    print(f"     • {report_type}")
        
        # Tiempo de ejecución
        if 'pipeline_summary' in self.pipeline_results:
            summary = self.pipeline_results['pipeline_summary']
            execution_time = summary['execution_time_seconds']
            print(f"\n⏱️  Tiempo total: {execution_time:.2f} segundos")
        
        print("\n📂 Archivos generados:")
        print("   - data/processed/ (datos procesados)")
        print("   - mlruns/ (experimentos MLflow)")
        print("   - reports/ (reportes Evidently)")
        print("   - models/ (modelos entrenados)")
        
        print("\n🔍 Para ver los resultados:")
        print("   - MLflow UI: mlflow ui")
        print("   - Reportes Evidently: abrir archivos .html en reports/")
        
        print("="*60)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Pipeline MLOps - Predicción de Precios de Viviendas")
    parser.add_argument("--phase", choices=['all', 'data', 'training', 'monitoring'], 
                       default='all', help="Fase a ejecutar")
    parser.add_argument("--verbose", action='store_true', help="Logging verbose")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Crear pipeline
    pipeline = MLOpsPipeline()
    
    if args.phase == 'all':
        success = pipeline.run_complete_pipeline()
    elif args.phase == 'data':
        success = pipeline.run_data_processing()
    elif args.phase == 'training':
        # Necesita datos procesados
        if not os.path.exists("data/processed/train.csv"):
            logger.error("Datos procesados no encontrados. Ejecute primero: python run_pipeline.py --phase data")
            return
        success, _ = pipeline.run_model_training()
    elif args.phase == 'monitoring':
        # Necesita datos y modelo
        if not os.path.exists("data/processed/train.csv"):
            logger.error("Datos procesados no encontrados")
            return
        # Para monitoreo independiente, usar modelo simple
        from sklearn.ensemble import RandomForestRegressor
        import pandas as pd
        
        train_df = pd.read_csv("data/processed/train.csv")
        X_train = train_df.drop('MedHouseVal', axis=1)
        y_train = train_df['MedHouseVal']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        success = pipeline.run_monitoring(model)
    
    if success:
        logger.info("🎉 Pipeline ejecutado exitosamente!")
    else:
        logger.error("❌ Pipeline falló")


if __name__ == "__main__":
    main()
