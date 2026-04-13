# app_unified_dark_pro.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency, kruskal
from sklearn.impute import KNNImputer
import io

# ============================================
# CONFIG
# ============================================
st.set_page_config(
    page_title="Dashboard Dark Pro",
    page_icon="🌙",
    layout="wide"
)

st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
.main-header {text-align: center; font-size: 2.2rem;}
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNCIONES
# ============================================
@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path)
    except:
        st.error(f"No se encontró: {path}")
        return pd.DataFrame()

def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="main-header">🌙 Dashboard Unificado</h1>', unsafe_allow_html=True)

dataset_option = st.radio(
    "Dataset:",
    ["Regresión", "Clasificación"],
    horizontal=True
)

# ============================================
# REGRESIÓN
# ============================================
if dataset_option == "Regresión":

    df = load_data("data/raw/dataset_regresion.csv")
    if df.empty:
        st.stop()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs([
        "General", "Distribuciones", "Correlaciones", "Scatter"
    ])

    # GENERAL
    with tab1:
        st.dataframe(df.head())

    # DISTRIBUCIONES
    with tab2:
        var = st.selectbox("Variable", num_cols)
        fig = px.histogram(df, x=var, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # CORRELACIONES
    with tab3:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # SCATTER
    with tab4:
        x = st.selectbox("X", num_cols)
        y = st.selectbox("Y", num_cols, index=1)

        fig = px.scatter(
            df, x=x, y=y,
            template="plotly_dark",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# CLASIFICACIÓN
# ============================================
else:

    df = load_data("data/raw/dataset_clasificacion.csv")
    if df.empty:
        st.stop()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs([
        "EDA", "Visualización", "Estadística", "Imputación"
    ])

    # EDA
    with tab1:
        st.dataframe(df.head())

        fig = px.histogram(df, x=num_cols[0], template="plotly_dark")
        st.plotly_chart(fig)

    # VISUAL
    with tab2:
        if "Transaction_Type" in df.columns:
            fig = px.histogram(
                df,
                x="Transaction_Type",
                color="Is_Fraud",
                barmode="group",
                template="plotly_dark"
            )
            st.plotly_chart(fig)

    # ESTADÍSTICA
    with tab3:
        if "Is_Fraud" in df.columns:
            for col in df.select_dtypes(include="object").columns:
                tabla = pd.crosstab(df[col], df["Is_Fraud"])
                chi2, p, _, _ = chi2_contingency(tabla)
                st.write(f"{col} p-value: {p:.4f}")

    # IMPUTACIÓN
    with tab4:
        df_missing = df.copy()

        for col in num_cols:
            df_missing.loc[df_missing.sample(frac=0.1).index, col] = np.nan

        imputer = KNNImputer()
        df_knn = df_missing.copy()
        df_knn[num_cols] = imputer.fit_transform(df_knn[num_cols])

        var = st.selectbox("Variable", num_cols)

        fig = px.histogram(df_knn, x=var, template="plotly_dark")
        st.plotly_chart(fig)