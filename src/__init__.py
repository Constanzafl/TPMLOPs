"""
MLOps House Prices Prediction Package

Un sistema completo de MLOps para predicción de precios de viviendas
que integra MLflow para tracking de experimentos y Evidently para
monitoreo de modelos y deriva de datos.

Módulos:
    data_processing: Procesamiento y preparación de datos
    model_training: Entrenamiento de modelos con MLflow
    model_evaluation: Evaluación completa de modelos
    monitoring: Monitoreo con Evidently

Ejemplo de uso:
    from src.data_processing import DataProcessor
    from src.model_training import MLflowModelTrainer
    from src.monitoring import ModelMonitor
    
    # Procesamiento de datos
    processor = DataProcessor()
    data = processor.load_data()
    
    # Entrenamiento de modelos
    trainer = MLflowModelTrainer()
    results = trainer.run_experiment_suite(X_train, y_train, X_val, y_val)
    
    # Monitoreo
    monitor = ModelMonitor()
    reports = monitor.generate_comprehensive_monitoring_report(train_df, test_df, model)
"""

__version__ = "1.0.0"
__author__ = "Equipo MLOps"
__email__ = "equipo@mlops.com"
__description__ = "Sistema MLOps para predicción de precios de viviendas"

# Imports principales para facilitar el uso
from .data_processing import DataProcessor
from .model_training import MLflowModelTrainer
from .model_evaluation import ModelEvaluator, compare_models
from .monitoring import ModelMonitor

# Configuración de logging para el paquete
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Metadata del paquete
__all__ = [
    'DataProcessor',
    'MLflowModelTrainer', 
    'ModelEvaluator',
    'ModelMonitor',
    'compare_models'
]

# Configuración por defecto
DEFAULT_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'val_size': 0.2,
    'mlflow_tracking_uri': 'file:./mlruns',
    'evidently_reports_dir': './reports',
    'log_level': 'INFO'
}

def get_version():
    """Obtener versión del paquete"""
    return __version__

def get_config():
    """Obtener configuración por defecto"""
    return DEFAULT_CONFIG.copy()

def setup_logging(level='INFO'):
    """Configurar logging para el paquete"""
    import logging
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"MLOps House Prices v{__version__} - Logging configurado")
    
    return logger

# Verificar dependencias críticas al importar
def _check_dependencies():
    """Verificar que las dependencias críticas están instaladas"""
    try:
        import mlflow
        import evidently
        import sklearn
        import pandas
        import numpy
    except ImportError as e:
        raise ImportError(
            f"Dependencia faltante: {e}. "
            "Instala todas las dependencias con: pip install -r requirements.txt"
        )

# Ejecutar verificación al importar
_check_dependencies()

# Banner informativo
def print_banner():
    """Mostrar banner del proyecto"""
    banner = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║              🏠 MLOps House Prices Prediction               ║
    ║                        Versión {__version__}                         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Sistema completo de MLOps para predicción de precios       ║
    ║  • MLflow: Tracking de experimentos                         ║
    ║  • Evidently: Monitoreo y observabilidad                   ║
    ║  • Pipeline automatizado end-to-end                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

# Función de ayuda rápida
def quick_start():
    """Guía rápida de inicio"""
    help_text = """
    🚀 GUÍA RÁPIDA DE INICIO
    
    1. Pipeline completo:
       python run_pipeline.py
    
    2. Por fases:
       python run_pipeline.py --phase data      # Solo datos
       python run_pipeline.py --phase training  # Solo entrenamiento
       python run_pipeline.py --phase monitoring # Solo monitoreo
    
    3. Ver experimentos MLflow:
       mlflow ui
    
    4. Explorar notebooks:
       jupyter notebook notebooks/
    
    5. Usar módulos individualmente:
       from src import DataProcessor, MLflowModelTrainer, ModelMonitor
    
    📚 Para más información, consulta el README.md
    """
    print(help_text)

# Configuración de warnings
import warnings

def suppress_warnings():
    """Suprimir warnings comunes no críticos"""
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=DeprecationWarning)

# Auto-configuración al importar
suppress_warnings()

# Información del módulo para debugging
def module_info():
    """Información del módulo"""
    info = {
        'name': __name__,
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': __all__,
        'config': DEFAULT_CONFIG
    }
    return info