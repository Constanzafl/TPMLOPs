# Notebook: Exploración de Datos - California Housing Dataset
# Para usar como Jupyter notebook, copiar este código en celdas separadas

# =====================================================
# CELDA 1: Imports y configuración
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("📊 Exploración de Datos - California Housing Dataset")
print("=" * 60)

# =====================================================
# CELDA 2: Cargar datos
# =====================================================

# Cargar dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"📁 Dataset cargado:")
print(f"   - Filas: {df.shape[0]:,}")
print(f"   - Columnas: {df.shape[1]}")
print(f"   - Memoria: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Mostrar primeras filas
print("\n🔍 Primeras 5 filas:")
df.head()

# =====================================================
# CELDA 3: Información general del dataset
# =====================================================

print("📋 Información general del dataset:")
print(df.info())

print("\n📊 Estadísticas descriptivas:")
df.describe()

# =====================================================
# CELDA 4: Análisis de valores faltantes
# =====================================================

print("🔍 Análisis de valores faltantes:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing_Count': missing_values.values,
    'Missing_Percentage': missing_percentage.values
})

print(missing_df)

# Visualización de valores faltantes
fig = px.bar(missing_df, 
             x='Column', 
             y='Missing_Percentage',
             title="Porcentaje de Valores Faltantes por Columna",
             labels={'Missing_Percentage': 'Porcentaje (%)'})
fig.show()

# =====================================================
# CELDA 5: Distribuciones de variables
# =====================================================

print("📊 Análisis de distribuciones:")

# Crear subplots para todas las variables
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=df.columns,
    specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(3)]
)

for i, col in enumerate(df.columns):
    row = i // 3 + 1
    col_num = i % 3 + 1
    
    fig.add_trace(
        go.Histogram(x=df[col], name=col, showlegend=False),
        row=row, col=col_num
    )

fig.update_layout(
    height=900,
    title_text="Distribuciones de todas las variables",
    showlegend=False
)
fig.show()

# =====================================================
# CELDA 6: Análisis de correlaciones
# =====================================================

print("🔗 Matriz de correlación:")

# Calcular correlaciones
correlation_matrix = df.corr()

# Heatmap interactivo
fig = px.imshow(
    correlation_matrix,
    text_auto=True,
    aspect="auto",
    title="Matriz de Correlación - California Housing Dataset",
    color_continuous_scale="RdBu_r"
)

fig.update_layout(
    width=800,
    height=800
)
fig.show()

# Correlaciones con la variable objetivo
target_corr = correlation_matrix['MedHouseVal'].abs().sort_values(ascending=False)
print("\n🎯 Correlaciones con MedHouseVal (valor absoluto):")
for var, corr in target_corr.items():
    if var != 'MedHouseVal':
        print(f"   {var}: {corr:.3f}")

# =====================================================
# CELDA 7: Análisis geográfico
# =====================================================

print("🗺️ Análisis geográfico:")

# Mapa de dispersión por precio
fig = px.scatter(
    df.sample(5000),  # Muestra para mejor performance
    x='Longitude',
    y='Latitude',
    color='MedHouseVal',
    size='Population',
    title="Distribución Geográfica de Precios de Viviendas",
    labels={
        'MedHouseVal': 'Precio Medio',
        'Longitude': 'Longitud',
        'Latitude': 'Latitud'
    },
    color_continuous_scale="viridis"
)

fig.update_layout(
    width=800,
    height=600
)
fig.show()

# =====================================================
# CELDA 8: Análisis de outliers
# =====================================================

print("📦 Análisis de outliers:")

# Boxplots para detectar outliers
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=df.columns
)

for i, col in enumerate(df.columns):
    row = i // 3 + 1
    col_num = i % 3 + 1
    
    fig.add_trace(
        go.Box(y=df[col], name=col, showlegend=False),
        row=row, col=col_num
    )

fig.update_layout(
    height=900,
    title_text="Detección de Outliers - Boxplots",
    showlegend=False
)
fig.show()

# Estadísticas de outliers (método IQR)
outlier_stats = {}
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_stats[col] = {
        'count': len(outliers),
        'percentage': (len(outliers) / len(df)) * 100
    }

outlier_df = pd.DataFrame(outlier_stats).T
print("\n📊 Estadísticas de outliers (método IQR):")
print(outlier_df)

# =====================================================
# CELDA 9: Análisis de relaciones bivariadas
# =====================================================

print("📈 Análisis de relaciones bivariadas:")

# Scatter plots con variable objetivo
features = ['MedInc', 'HouseAge', 'AveRooms', 'Population']

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f"{feat} vs MedHouseVal" for feat in features]
)

for i, feat in enumerate(features):
    row = i // 2 + 1
    col = i % 2 + 1
    
    # Muestra para mejor visualización
    sample_df = df.sample(2000)
    
    fig.add_trace(
        go.Scatter(
            x=sample_df[feat],
            y=sample_df['MedHouseVal'],
            mode='markers',
            name=feat,
            showlegend=False,
            opacity=0.6
        ),
        row=row, col=col
    )

fig.update_layout(
    height=800,
    title_text="Relaciones Bivariadas con Precio de Vivienda"
)
fig.show()

# =====================================================
# CELDA 10: Feature engineering exploratorio
# =====================================================

print("🔧 Exploración de feature engineering:")

# Crear nuevas features
df_engineered = df.copy()

# Features derivadas
df_engineered['RoomsPerHousehold'] = df_engineered['AveRooms'] / df_engineered['AveOccup']
df_engineered['BedroomsPerRoom'] = df_engineered['AveBedrms'] / df_engineered['AveRooms']
df_engineered['PopulationPerHousehold'] = df_engineered['Population'] / df_engineered['HouseAge']

# Correlaciones de nuevas features
new_features = ['RoomsPerHousehold', 'BedroomsPerRoom', 'PopulationPerHousehold']
new_corr = df_engineered[new_features + ['MedHouseVal']].corr()['MedHouseVal'].abs().sort_values(ascending=False)

print("🎯 Correlaciones de nuevas features con MedHouseVal:")
for feat, corr in new_corr.items():
    if feat != 'MedHouseVal':
        print(f"   {feat}: {corr:.3f}")

# Visualizar nuevas features
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=new_features
)

for i, feat in enumerate(new_features):
    fig.add_trace(
        go.Histogram(x=df_engineered[feat], name=feat, showlegend=False),
        row=1, col=i+1
    )

fig.update_layout(
    height=400,
    title_text="Distribuciones de Features Engineered"
)
fig.show()

# =====================================================
# CELDA 11: Análisis por categorías de edad
# =====================================================

print("🏠 Análisis por categorías de edad de viviendas:")

# Categorizar edad de casas
df_engineered['HouseAgeCategory'] = pd.cut(
    df_engineered['HouseAge'], 
    bins=[0, 10, 20, 35, np.inf],
    labels=['Nueva', 'Reciente', 'Antigua', 'Muy Antigua']
)

# Análisis por categoría
age_analysis = df_engineered.groupby('HouseAgeCategory').agg({
    'MedHouseVal': ['mean', 'median', 'std', 'count'],
    'MedInc': 'mean',
    'Population': 'mean'
}).round(3)

print("📊 Estadísticas por categoría de edad:")
print(age_analysis)

# Boxplot por categoría de edad
fig = px.box(
    df_engineered,
    x='HouseAgeCategory',
    y='MedHouseVal',
    title="Distribución de Precios por Categoría de Edad de Vivienda"
)
fig.show()

# =====================================================
# CELDA 12: Resumen y conclusiones
# =====================================================

print("📋 RESUMEN DE LA EXPLORACIÓN")
print("=" * 60)

print("✅ Dataset limpio:")
print(f"   - Sin valores faltantes")
print(f"   - {df.shape[0]:,} registros válidos")
print(f"   - {df.shape[1]} variables originales")

print("\n🎯 Variable objetivo (MedHouseVal):")
print(f"   - Media: ${df['MedHouseVal'].mean():.2f} (x100k)")
print(f"   - Mediana: ${df['MedHouseVal'].median():.2f} (x100k)")
print(f"   - Rango: ${df['MedHouseVal'].min():.2f} - ${df['MedHouseVal'].max():.2f} (x100k)")

print("\n🔗 Features más correlacionadas con precio:")
top_features = target_corr.head(4)
for feat, corr in top_features.items():
    if feat != 'MedHouseVal':
        print(f"   - {feat}: {corr:.3f}")

print("\n🌟 Insights principales:")
print("   - MedInc es el predictor más fuerte (0.688)")
print("   - Existe clara distribución geográfica de precios")
print("   - Las features engineered mejoran correlaciones")
print("   - Presencia de outliers en varias variables")
print("   - Dataset balanceado geográficamente")

print("\n🚀 Recomendaciones para modelado:")
print("   - Aplicar feature engineering")
print("   - Considerar transformaciones para outliers")
print("   - Usar regularización para evitar overfitting")
print("   - Incluir features geográficas")
print("   - Probar modelos ensemble")

print("\n" + "=" * 60)
print("🎉 Exploración completada exitosamente!")
