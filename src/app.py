# app_unified.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, kruskal
from sklearn.impute import KNNImputer
import io

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Dashboard Unificado - Análisis de Datos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #34495e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .dataset-selector {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNCIONES DE CARGA DE DATOS
# ============================================
@st.cache_data
def load_regression_data():
    """Carga datos del análisis de regresión (viajes)"""
    try:
        df = pd.read_csv('../data/raw/dataset_regresion.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo de regresión. Verifica la ruta.")
        return pd.DataFrame()

@st.cache_data
def load_classification_data():
    """Carga datos del análisis de clasificación (fraude)"""
    try:
        df = pd.read_csv("data/raw/dataset_clasificacion.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo de clasificación. Verifica la ruta.")
        return pd.DataFrame()

# Funciones auxiliares para análisis de regresión
def detectar_outliers_iqr(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = (df[columna] < limite_inferior) | (df[columna] > limite_superior)
    return outliers.sum(), outliers.sum()/len(df)*100

# ============================================
# SELECTOR DE DATASET
# ============================================
st.markdown('<h1 class="main-header">📊 Dashboard Unificado de Análisis de Datos</h1>', unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        dataset_option = st.radio(
            "📁 Selecciona el dataset a analizar:",
            ["🚗 Análisis de Viajes (Regresión)", "💰 Detección de Fraude (Clasificación)"],
            horizontal=True,
            help="Selecciona cuál de los dos análisis deseas visualizar"
        )

st.markdown("---")

# ============================================
# SECCIÓN 1: ANÁLISIS DE VIAJES (REGRESIÓN)
# ============================================
if dataset_option == "🚗 Análisis de Viajes (Regresión)":
    
    df = load_regression_data()
    
    if df.empty:
        st.stop()
    
    # Sidebar con información
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/taxi.png", width=80)
        st.markdown("## 📋 Información del Dataset")
        st.markdown(f"""
        - **Filas:** {df.shape[0]:,}
        - **Columnas:** {df.shape[1]}
        - **Tipos de datos:** {df.dtypes.nunique()}
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Variables Principales")
        st.markdown("""
        - **Avg CTAT**: Tiempo de espera del cliente
        - **Avg VTAT**: Tiempo de llegada del vehículo
        - **Booking Value**: Valor del viaje
        - **Ride Distance**: Distancia recorrida
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Selección de Análisis")
        analisis_option = st.selectbox(
            "Ir a:",
            ["Vista General", "Distribuciones", "Outliers", "Correlaciones", "Gráficos de Dispersión", "Análisis por Categorías"]
        )
    
    # 1. VISTA GENERAL
    if analisis_option == "Vista General":
        st.markdown('<h2 class="sub-header">📌 Vista General del Dataset de Viajes</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Registros", f"{len(df):,}")
        with col2:
            st.metric("Variables Numéricas", df.select_dtypes(include=[np.number]).shape[1])
        with col3:
            st.metric("Variables Categóricas", df.select_dtypes(include=['object']).shape[1])
        with col4:
            st.metric("Porcentaje Completitud", f"{(1 - df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.1f}%")
        
        st.markdown("---")
        
        st.markdown("### 🔍 Primeras 5 filas del dataset")
        st.dataframe(df.head(), use_container_width=True)
        
        st.markdown("### 📋 Información de Columnas")
        col_info = pd.DataFrame({
            'Tipo': df.dtypes,
            'No Nulos': df.count(),
            'Nulos (%)': (df.isnull().sum() / len(df) * 100).round(2),
            'Valores Únicos': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
    
    # 2. DISTRIBUCIONES
    elif analisis_option == "Distribuciones":
        st.markdown('<h2 class="sub-header">📊 Distribuciones de Variables</h2>', unsafe_allow_html=True)
        
        variables_numericas = ['Avg CTAT', 'Avg VTAT', 'Ride Distance', 'Booking Value', 'Driver Ratings', 'Customer Rating']
        variables_disponibles = [v for v in variables_numericas if v in df.columns]
        
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_var = st.selectbox("Selecciona variable:", variables_disponibles)
        
        with col2:
            st.markdown(f"### 📈 Distribución de {selected_var}")
        
        fig = px.histogram(
            df, x=selected_var, 
            nbins=50, 
            title=f"Distribución de {selected_var}",
            labels={selected_var: selected_var, 'count': 'Frecuencia'},
            color_discrete_sequence=['#2ecc71']
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📊 Estadísticas Descriptivas")
        stats_df = df[selected_var].describe().to_frame().T
        stats_df = stats_df.round(2)
        st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("### 🎯 Distribuciones de todas las variables numéricas")
        cols = st.columns(2)
        for i, var in enumerate(variables_disponibles):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df[var].dropna(), bins=50, kde=True, color='#3498db', ax=ax)
                ax.set_title(f'Distribución de {var}', fontsize=12, fontweight='bold')
                ax.set_xlabel(var)
                ax.set_ylabel('Frecuencia')
                st.pyplot(fig)
                plt.close()
    
    # 3. OUTLIERS
    elif analisis_option == "Outliers":
        st.markdown('<h2 class="sub-header">⚠️ Análisis de Outliers</h2>', unsafe_allow_html=True)
        
        variables_numericas = ['Avg CTAT', 'Avg VTAT', 'Ride Distance', 'Booking Value', 'Driver Ratings', 'Customer Rating']
        variables_disponibles = [v for v in variables_numericas if v in df.columns]
        
        outliers_data = []
        for var in variables_disponibles:
            n_outliers, pct_outliers = detectar_outliers_iqr(df, var)
            outliers_data.append({
                'Variable': var,
                'Outliers': n_outliers,
                'Porcentaje (%)': round(pct_outliers, 2)
            })
        
        outliers_df = pd.DataFrame(outliers_data)
        st.dataframe(outliers_df, use_container_width=True)
        
        st.markdown("### 📦 Boxplots para detección de outliers")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, var in enumerate(variables_disponibles):
            sns.boxplot(y=df[var], ax=axes[i], color='lightblue')
            axes[i].set_title(f'Boxplot de {var}', fontsize=12)
            axes[i].set_ylabel('Valor')
            median = df[var].median()
            axes[i].axhline(y=median, color='red', linestyle='--', alpha=0.5, label=f'Mediana: {median:.1f}')
            axes[i].legend()
        
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.info("💡 **Interpretación:** Booking Value, Driver Ratings y Customer Rating presentan outliers significativos que podrían requerir tratamiento.")
    
    # 4. CORRELACIONES
    elif analisis_option == "Correlaciones":
        st.markdown('<h2 class="sub-header">📈 Matrices de Correlación</h2>', unsafe_allow_html=True)
        
        variables_numericas = ['Avg CTAT', 'Avg VTAT', 'Ride Distance', 'Booking Value', 'Driver Ratings', 'Customer Rating']
        df_corr = df[variables_numericas]
        
        tab1, tab2, tab3 = st.tabs(["📊 Pearson", "📈 Spearman", "📉 Kendall"])
        
        with tab1:
            corr_matrix = df_corr.corr(method='pearson')
            fig = px.imshow(
                corr_matrix,
                text_auto='.3f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title="Matriz de Correlación - Pearson",
                labels=dict(color="Correlación")
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            corr_matrix = df_corr.corr(method='spearman')
            fig = px.imshow(
                corr_matrix,
                text_auto='.3f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title="Matriz de Correlación - Spearman",
                labels=dict(color="Correlación")
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            corr_matrix = df_corr.corr(method='kendall')
            fig = px.imshow(
                corr_matrix,
                text_auto='.3f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title="Matriz de Correlación - Kendall",
                labels=dict(color="Correlación")
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🎯 Correlaciones con Avg CTAT")
        corr_con_ctat = df_corr.corr()['Avg CTAT'].sort_values(ascending=False)
        corr_df = pd.DataFrame({
            'Variable': corr_con_ctat.index,
            'Correlación': corr_con_ctat.values
        })
        
        fig = px.bar(
            corr_df[corr_df['Variable'] != 'Avg CTAT'],
            x='Correlación',
            y='Variable',
            orientation='h',
            title="Variables más correlacionadas con Avg CTAT",
            color='Correlación',
            color_continuous_scale='RdYlGn',
            labels={'Correlación': 'Coeficiente de correlación'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 5. GRÁFICOS DE DISPERSIÓN
    elif analisis_option == "Gráficos de Dispersión":
        st.markdown('<h2 class="sub-header">🔍 Gráficos de Dispersión (Scatter Plots)</h2>', unsafe_allow_html=True)
        
        variables_numericas = ['Avg CTAT', 'Avg VTAT', 'Ride Distance', 'Booking Value', 'Driver Ratings', 'Customer Rating']
        variables_disponibles = [v for v in variables_numericas if v in df.columns]
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Selecciona variable X:", variables_disponibles, index=0)
        with col2:
            y_var = st.selectbox("Selecciona variable Y:", variables_disponibles, index=1)
        
        st.markdown(f"### 📈 Relación entre {x_var} y {y_var}")
        
        plot_df = df[[x_var, y_var]].dropna()
        
        fig = px.scatter(
            plot_df.sample(min(5000, len(plot_df))),
            x=x_var,
            y=y_var,
            title=f"{x_var} vs {y_var}",
            labels={x_var: x_var, y_var: y_var},
            trendline="ols",
            trendline_color_override="red",
            opacity=0.6
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        corr = plot_df[x_var].corr(plot_df[y_var])
        if abs(corr) > 0.3:
            st.success(f"✅ Correlación: {corr:.3f} - Relación {'positiva' if corr > 0 else 'negativa'} moderada/fuerte")
        else:
            st.info(f"ℹ️ Correlación: {corr:.3f} - Relación débil")
        
        st.markdown("---")
        st.markdown("### 🎯 Relación de Avg CTAT con otras variables")
        
        vars_to_plot = ['Avg VTAT', 'Ride Distance', 'Booking Value', 'Driver Ratings', 'Customer Rating']
        vars_to_plot = [v for v in vars_to_plot if v in df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, var in enumerate(vars_to_plot):
            if var in df.columns:
                plot_data = df[['Avg CTAT', var]].dropna()
                if len(plot_data) > 5000:
                    plot_data = plot_data.sample(5000)
                
                axes[i].scatter(plot_data['Avg CTAT'], plot_data[var], alpha=0.3, s=10)
                axes[i].set_xlabel('Avg CTAT')
                axes[i].set_ylabel(var)
                axes[i].set_title(f'Avg CTAT vs {var}', fontsize=12)
                axes[i].grid(True, alpha=0.3)
        
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # 6. ANÁLISIS POR CATEGORÍAS
    else:
        st.markdown('<h2 class="sub-header">📊 Análisis por Variables Categóricas</h2>', unsafe_allow_html=True)
        
        categorical_cols = ['Vehicle Type', 'Payment Method', 'Booking Status']
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        
        target_var = st.selectbox("Selecciona variable objetivo:", ['Booking Value', 'Avg CTAT', 'Avg VTAT'])
        
        for col in categorical_cols:
            st.markdown(f"### 🚗 Análisis por {col}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                counts = df[col].value_counts().head(10)
                fig = px.bar(
                    x=counts.values, y=counts.index,
                    orientation='h',
                    title=f"Distribución de {col}",
                    labels={'x': 'Frecuencia', 'y': col},
                    color_discrete_sequence=['#3498db']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if target_var in df.columns:
                    agg_data = df.groupby(col)[target_var].agg(['mean', 'median', 'count']).reset_index()
                    fig = px.bar(
                        agg_data,
                        x=col,
                        y='mean',
                        title=f"{target_var} promedio por {col}",
                        labels={'mean': f'{target_var} promedio'},
                        color_discrete_sequence=['#e74c3c']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")

# ============================================
# SECCIÓN 2: ANÁLISIS DE FRAUDE (CLASIFICACIÓN)
# ============================================
else:
    
    df = load_classification_data()
    
    if df.empty:
        st.stop()
    
    # Sidebar con información
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/fraud.png", width=80)
        st.markdown("## 📋 Información del Dataset")
        st.markdown(f"""
        - **Filas:** {df.shape[0]:,}
        - **Columnas:** {df.shape[1]}
        """)
        st.markdown("**Variables principales:**")
        st.markdown("""
        - Age (int)
        - Transaction_Amount (float)
        - Account_Balance (float)
        - Is_Fraud (int - objetivo)
        """)
    
    # Crear pestañas para el análisis de fraude
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 EDA General", 
        "📊 Visualizaciones", 
        "🔬 Pruebas Estadísticas", 
        "🩹 Imputación de Datos"
    ])
    
    # PESTAÑA 1: EDA GENERAL
    with tab1:
        st.header("Análisis Exploratorio de Datos (EDA) - Fraude")
        
        st.subheader("Primeras 5 filas del dataset")
        st.dataframe(df.head())
        
        st.subheader("Información general")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Tipos de datos y valores nulos**")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        with col2:
            st.write("**Estadísticas descriptivas (numéricas)**")
            st.dataframe(df.describe())
        
        st.subheader("Distribuciones de variables numéricas")
        numeric_cols = ['Age', 'Transaction_Amount', 'Account_Balance']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], bins=30, ax=axes[i], kde=True)
            axes[i].set_title(f"Distribución de {col}")
        st.pyplot(fig)
        
        st.subheader("Matriz de correlación (Pearson)")
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, ax=ax, cmap='coolwarm')
        st.pyplot(fig)
    
    # PESTAÑA 2: VISUALIZACIONES
    with tab2:
        st.header("Visualizaciones por tipo de transacción y fraude")
        
        st.subheader("Tipo de transacción vs Fraude")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x='Transaction_Type', hue='Is_Fraud', data=df, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
        st.caption("El fraude se distribuye de forma casi idéntica entre todos los tipos de transacción.")
        
        st.subheader("Dispositivo vs Fraude")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='Device_Type', hue='Is_Fraud', data=df, ax=ax)
        st.pyplot(fig)
        st.caption("El tipo de dispositivo no es un factor determinante para el fraude.")
        
        st.subheader("Categoría de comerciante vs Fraude")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.countplot(x='Merchant_Category', hue='Is_Fraud', data=df, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
        st.caption("Todas las categorías presentan proporciones similares de fraude.")
        
        st.subheader("Comparación de variables numéricas entre grupos")
        cols = ['Transaction_Amount', 'Account_Balance', 'Age']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(cols):
            sns.boxplot(x='Is_Fraud', y=col, data=df, ax=axes[i])
            axes[i].set_title(f"{col} vs Fraude")
        st.pyplot(fig)
        st.caption("No hay diferencias significativas en las distribuciones entre transacciones legítimas y fraudulentas.")
    
    # PESTAÑA 3: PRUEBAS ESTADÍSTICAS
    with tab3:
        st.header("Pruebas de Asociación y Dependencia")
        
        st.subheader("Prueba de Chi-cuadrado")
        categorical_vars = ['Gender', 'Device_Type', 'Merchant_Category', 'Transaction_Type']
        for var in categorical_vars:
            tabla = pd.crosstab(df[var], df['Is_Fraud'])
            chi2, p, dof, expected = chi2_contingency(tabla)
            st.write(f"**{var}** → p-value: {p:.4f}")
            if p > 0.05:
                st.success(f"No hay asociación significativa con el fraude (p > 0.05)")
            else:
                st.warning(f"Asociación significativa detectada (p ≤ 0.05)")
        
        st.subheader("Prueba de Kruskal-Wallis")
        numeric_vars = ['Age', 'Account_Balance', 'Transaction_Amount']
        for var in numeric_vars:
            grupo_no = df[df['Is_Fraud'] == 0][var]
            grupo_si = df[df['Is_Fraud'] == 1][var]
            stat, p = kruskal(grupo_no, grupo_si)
            st.write(f"**{var}** → p-value: {p:.4f}")
            if p > 0.05:
                st.success(f"No hay diferencias significativas entre grupos (p > 0.05)")
            else:
                st.warning(f"Diferencias significativas detectadas (p ≤ 0.05)")
        
        st.subheader("Correlación de Spearman")
        corr_spearman = df[numeric_vars].corr(method='spearman')
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_spearman, annot=True, ax=ax, cmap='viridis', fmt=".4f")
        st.pyplot(fig)
        st.caption("Valores cercanos a cero confirman independencia estocástica entre variables numéricas.")
    
    # PESTAÑA 4: IMPUTACIÓN DE DATOS
    with tab4:
        st.header("Tratamiento de Datos Faltantes (Imputación)")
        
        st.markdown("""
        **Simulación de valores faltantes** Se eliminaron valores aleatorios para practicar técnicas de imputación:
        - Age: 2201 valores nulos  
        - Transaction_Amount: 1495 valores nulos  
        - Account_Balance: 2149 valores nulos  
        """)
        
        np.random.seed(42)
        df_missing = df.copy()
        indices_age = df_missing.sample(n=2201, random_state=42).index
        indices_amount = df_missing.sample(n=1495, random_state=1).index
        indices_balance = df_missing.sample(n=2149, random_state=7).index
        df_missing.loc[indices_age, 'Age'] = np.nan
        df_missing.loc[indices_amount, 'Transaction_Amount'] = np.nan
        df_missing.loc[indices_balance, 'Account_Balance'] = np.nan
        
        df_mean = df_missing.copy()
        df_mean['Transaction_Amount'] = df_mean['Transaction_Amount'].fillna(df_mean['Transaction_Amount'].mean())
        df_mean['Age'] = df_mean['Age'].fillna(df_mean['Age'].mean())
        df_mean['Account_Balance'] = df_mean['Account_Balance'].fillna(df_mean['Account_Balance'].mean())
        
        imputer = KNNImputer(n_neighbors=3)
        df_knn = df_missing.copy()
        df_knn[['Transaction_Amount', 'Age', 'Account_Balance']] = imputer.fit_transform(
            df_knn[['Transaction_Amount', 'Age', 'Account_Balance']]
        )
        
        st.subheader("Comparación de Métodos de Imputación")
        var = st.selectbox("Selecciona variable", ['Transaction_Amount', 'Age', 'Account_Balance'])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(df[var], label="Original", linewidth=2, ax=ax)
        sns.kdeplot(df_mean[var], label="Media", ax=ax)
        sns.kdeplot(df_knn[var], label="KNN", ax=ax)
        ax.legend()
        ax.set_title(f"Distribución de {var} según método de imputación")
        st.pyplot(fig)
        
        st.markdown("""
        **Conclusión sobre imputación:** 
        - La imputación por **media** genera una acumulación artificial alrededor del promedio.  
        - El método **KNN** preserva mejor la variabilidad y estructura original de los datos.  
        - Para este dataset, KNN es la técnica recomendada si se requiriera imputar valores faltantes.
        """)
        
        st.subheader("Estadísticas descriptivas comparativas")
        st.dataframe(df_missing[['Transaction_Amount','Age','Account_Balance']].describe())
        st.dataframe(df_mean[['Transaction_Amount','Age','Account_Balance']].describe())
        st.dataframe(df_knn[['Transaction_Amount','Age','Account_Balance']].describe())

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #7f8c8d;'>📊 Dashboard Unificado - Análisis de Datos de Viajes y Detección de Fraude</p>",
    unsafe_allow_html=True
)