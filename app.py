import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from prophet import Prophet

st.set_page_config(
    layout="wide",
    page_title="Análisis de Indicadores de Desarrollo Socioeconómico",
    page_icon="🍆💦"
)

st.title("Análisis de Indicadores de Desarrollo Socioeconómico")
st.write("Esta aplicación permite visualizar y analizar indicadores de desarrollo socioeconómico en diferentes países a lo largo del tiempo.")

@st.cache_data
def obtener_datos(indicador, fecha_inicio, fecha_fin):
    url = f'https://api.worldbank.org/v2/country/all/indicator/{indicador}'
    params = {
        'format': 'json',
        'date': f'{fecha_inicio}:{fecha_fin}',
        'per_page': '1000'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()[1]
        if not data:
            st.error(f"No se encontraron datos para el indicador {indicador}.")
            return pd.DataFrame()
        datos_limpios = [
            {
                'Pais': entrada['country']['value'] if isinstance(entrada['country'], dict) else entrada['country'],
                'Codigo': entrada['countryiso3code'],
                'Año': entrada['date'],
                'Valor': entrada['value']
            }
            for entrada in data if entrada['value'] is not None
        ]
        return pd.DataFrame(datos_limpios)
    else:
        st.error("Error al acceder a los datos")
        return None

df_pib = obtener_datos('NY.GDP.PCAP.CD', 2000, 2020)
df_vida = obtener_datos('SP.DYN.LE00.IN', 2000, 2020)

if df_pib is not None and not df_pib.empty and df_vida is not None and not df_vida.empty:
    df_pib['Año'] = df_pib['Año'].astype(int)
    df_pib['Valor'] = df_pib['Valor'].astype(float)
    df_vida['Año'] = df_vida['Año'].astype(int)
    df_vida['Valor'] = df_vida['Valor'].astype(float)
else:
    st.stop()

st.sidebar.header("Opciones de Análisis")
pais_seleccionado = st.sidebar.selectbox("Seleccione un país", df_pib['Pais'].unique())
indicador_seleccionado = st.sidebar.selectbox("Seleccione un indicador", ["PIB per cápita", "Esperanza de vida"])
prediccion_habilitada = st.sidebar.checkbox("Mostrar predicción de tendencia")

df = df_pib if indicador_seleccionado == "PIB per cápita" else df_vida
datos_filtrados = df[df['Pais'] == pais_seleccionado]

st.subheader(f"{indicador_seleccionado} en {pais_seleccionado} (2000 - 2020)")
fig = px.line(datos_filtrados, x="Año", y="Valor", title=f"{indicador_seleccionado} en {pais_seleccionado}", labels={"Valor": "Valor en USD" if indicador_seleccionado == "PIB per cápita" else "Esperanza de Vida en años"})
fig.update_traces(mode="lines+markers")
st.plotly_chart(fig)

if prediccion_habilitada:
    st.subheader(f"Predicción de tendencia para {indicador_seleccionado} en {pais_seleccionado}")
    datos_filtrados_pred = datos_filtrados[['Año', 'Valor']].rename(columns={'Año': 'ds', 'Valor': 'y'})
    modelo = Prophet()
    modelo.fit(datos_filtrados_pred)
    futuro = modelo.make_future_dataframe(periods=5, freq='Y')
    prediccion = modelo.predict(futuro)
    fig_pred = px.line(prediccion, x="ds", y="yhat", title=f"Predicción de {indicador_seleccionado} en {pais_seleccionado} (hasta 2025)")
    fig_pred.add_scatter(x=datos_filtrados_pred['ds'], y=datos_filtrados_pred['y'], mode="markers", name="Datos reales")
    st.plotly_chart(fig_pred)

