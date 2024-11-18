import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from prophet import Prophet
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Análisis de Indicadores de Desarrollo Socioeconómico", page_icon="🌍")

st.title("Análisis de Indicadores de Desarrollo Socioeconómico")
st.write("Esta aplicación permite visualizar y analizar indicadores de desarrollo socioeconómico en diferentes países a lo largo del tiempo.")

@st.cache_data
def obtener_datos(indicador, fecha_inicio, fecha_fin):
    url = f'https://api.worldbank.org/v2/country/all/indicator/{indicador}'
    params = {'format': 'json', 'date': f'{fecha_inicio}:{fecha_fin}', 'per_page': '1000'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()[1]
        if not data:
            return pd.DataFrame()
        datos_limpios = [{'Pais': entrada['country']['value'], 'Codigo': entrada['countryiso3code'], 'Año': entrada['date'], 'Valor': entrada['value']} for entrada in data if entrada['value'] is not None]
        return pd.DataFrame(datos_limpios)
    return None

@st.cache_data
def cargar_coordenadas_paises():
    coordenadas_url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    return coordenadas_url

indicadores = {
    "PIB per cápita": "NY.GDP.PCAP.CD",
    "Esperanza de vida": "SP.DYN.LE00.IN",
    "Tasa de alfabetización": "SE.ADT.LITR.ZS",
    "Población total": "SP.POP.TOTL"
}

rango_inicio = 1960
rango_fin = 2020

df_indicadores = {nombre: obtener_datos(codigo, rango_inicio, rango_fin) for nombre, codigo in indicadores.items()}

if any(df is None or df.empty for df in df_indicadores.values()):
    st.stop()

st.sidebar.header("Opciones de Análisis")
pais_seleccionado = st.sidebar.selectbox("Seleccione un país", df_indicadores["PIB per cápita"]['Pais'].unique())
indicador_seleccionado = st.sidebar.selectbox("Seleccione un indicador", list(indicadores.keys()))
prediccion_habilitada = st.sidebar.checkbox("Mostrar predicción de tendencia")

df = df_indicadores[indicador_seleccionado]
datos_filtrados = df[df['Pais'] == pais_seleccionado]

st.subheader(f"{indicador_seleccionado} en {pais_seleccionado} ({rango_inicio} - {rango_fin})")
fig = px.line(datos_filtrados, x="Año", y="Valor", title=f"{indicador_seleccionado} en {pais_seleccionado}", labels={"Valor": f"Valor de {indicador_seleccionado}"})
fig.update_traces(mode="lines+markers")
st.plotly_chart(fig)

if prediccion_habilitada:
    datos_filtrados_pred = datos_filtrados[['Año', 'Valor']].rename(columns={'Año': 'ds', 'Valor': 'y'})
    modelo = Prophet()
    modelo.fit(datos_filtrados_pred)
    futuro = modelo.make_future_dataframe(periods=5, freq='Y')
    prediccion = modelo.predict(futuro)
    fig_pred = px.line(prediccion, x="ds", y="yhat", title=f"Predicción de {indicador_seleccionado} en {pais_seleccionado} (hasta 2025)")
    fig_pred.add_scatter(x=datos_filtrados_pred['ds'], y=datos_filtrados_pred['y'], mode="markers", name="Datos reales")
    st.plotly_chart(fig_pred)

# Mapa Regional
st.subheader("Mapa de Calor: Comparación Regional")
año_seleccionado = st.slider("Seleccione un año", min_value=rango_inicio, max_value=rango_fin, value=2020)

# Generamos el mapa sin necesidad de un botón
df_anual = df[df['Año'] == str(año_seleccionado)]
coordenadas_url = cargar_coordenadas_paises()

m = folium.Map(location=[0, 0], zoom_start=2)
for _, row in df_anual.iterrows():
    folium.CircleMarker(
        location=[row.get('lat', 0), row.get('lon', 0)],
        radius=5,
        popup=f"{row['Pais']}: {row['Valor']}",
        color="blue",
        fill=True
    ).add_to(m)

st.subheader("Mapa de Calor")
st_folium(m, width=700)

# Opciones de exportación
st.sidebar.subheader("Exportar")
if st.sidebar.button("Exportar datos a CSV"):
    datos_filtrados.to_csv(f"{pais_seleccionado}_{indicador_seleccionado}.csv", index=False)
    st.sidebar.success("Datos exportados exitosamente.")
