"""
Módulo para procesamiento y preparación de datos
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Clase para procesamiento de datos del dataset California Housing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """Cargar el dataset California Housing"""
        logger.info("Cargando dataset California Housing...")
        
        # Cargar datos
        housing = fetch_california_housing(as_frame=True)
        
        # Crear DataFrame
        df = housing.frame
        
        # Información del dataset
        logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        logger.info(f"Columnas: {list(df.columns)}")
        
        return df
    
    def add_features(self, df):
        """Agregar features derivadas"""
        logger.info("Creando features derivadas...")
        
        df = df.copy()
        
        # Feature engineering
        df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
        df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
        df['PopulationPerHousehold'] = df['Population'] / df['HouseAge']
        
        # Categorizar edad de las casas
        df['HouseAgeCategory'] = pd.cut(df['HouseAge'], 
                                       bins=[0, 10, 20, 35, np.inf],
                                       labels=['New', 'Recent', 'Old', 'Very_Old'])
        
        # One-hot encoding para categorías
        df = pd.get_dummies(df, columns=['HouseAgeCategory'], prefix='Age')
        
        logger.info(f"Features después de ingeniería: {df.shape[1]} columnas")
        
        return df
    
    def clean_data(self, df):
        """Limpiar y validar datos"""
        logger.info("Limpiando datos...")
        
        # Verificar valores nulos
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Valores nulos encontrados: {null_counts[null_counts > 0]}")
        
        # Remover outliers extremos (más de 3 desviaciones estándar)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col != 'MedHouseVal':  # No remover outliers del target
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = df.shape[0]
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                outliers_after = df.shape[0]
                
                if outliers_before != outliers_after:
                    logger.info(f"Removidos {outliers_before - outliers_after} outliers de {col}")
        
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.2, random_state=42):
        """Dividir datos en train, validation y test"""
        logger.info("Dividiendo datos en train/val/test...")
        
        # Separar features y target
        X = df.drop('MedHouseVal', axis=1)
        y = df['MedHouseVal']
        
        # Primera división: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Segunda división: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        logger.info(f"Train: {X_train.shape[0]} muestras")
        logger.info(f"Validation: {X_val.shape[0]} muestras")
        logger.info(f"Test: {X_test.shape[0]} muestras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Escalar features numéricas"""
        logger.info("Escalando features...")
        
        # Identificar columnas numéricas
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns
        
        # Fit scaler solo en train
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        # Escalar solo columnas numéricas
        X_train_scaled[numeric_columns] = self.scaler.fit_transform(X_train[numeric_columns])
        X_val_scaled[numeric_columns] = self.scaler.transform(X_val[numeric_columns])
        X_test_scaled[numeric_columns] = self.scaler.transform(X_test[numeric_columns])
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                           output_dir="data/processed"):
        """Guardar datos procesados"""
        logger.info("Guardando datos procesados...")
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar datasets
        pd.concat([X_train, y_train], axis=1).to_csv(
            os.path.join(output_dir, "train.csv"), index=False
        )
        pd.concat([X_val, y_val], axis=1).to_csv(
            os.path.join(output_dir, "validation.csv"), index=False
        )
        pd.concat([X_test, y_test], axis=1).to_csv(
            os.path.join(output_dir, "test.csv"), index=False
        )
        
        logger.info(f"Datos guardados en {output_dir}")
    
    def get_data_summary(self, df):
        """Obtener resumen estadístico de los datos"""
        summary = {
            'total_samples': len(df),
            'features': len(df.columns) - 1,  # -1 por el target
            'target_mean': df['MedHouseVal'].mean(),
            'target_std': df['MedHouseVal'].std(),
            'missing_values': df.isnull().sum().sum(),
            'feature_names': list(df.drop('MedHouseVal', axis=1).columns)
        }
        
        return summary


def main():
    """Función principal para procesamiento de datos"""
    processor = DataProcessor()
    
    # 1. Cargar datos
    df = processor.load_data()
    
    # 2. Feature engineering
    df = processor.add_features(df)
    
    # 3. Limpiar datos
    df = processor.clean_data(df)
    
    # 4. Dividir datos
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(df)
    
    # 5. Escalar features
    X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_features(
        X_train, X_val, X_test
    )
    
    # 6. Guardar datos procesados
    processor.save_processed_data(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test
    )
    
    # 7. Mostrar resumen
    summary = processor.get_data_summary(df)
    logger.info(f"Resumen del procesamiento: {summary}")
    
    return processor, X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


if __name__ == "__main__":
    main()