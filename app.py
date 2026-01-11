"""
Cas d'estudi per la PR2 de Visualització de dades.

Impacte de l'orografia local: Aplicació Streamlit per visualitzar els 
resultats del model WRF en comparació amb les dades d'observació

Autor: Arnau Toledano
Curs: PR2, UOC - Visualització de dades
"""

import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from scipy.spatial import distance

# Coordenades de l'estació 
DAS_LAT = 42.38605
DAS_LON = 1.86640

# Configuracions de domini
DOMAINS = {
    "9km": {
        "wrfout": "data/wrfout_d01_9km.nc",
        "geo_em": "data/geo_em_d01_9km.nc",
        "resolution": "9 km"
    },
    "3km": {
        "wrfout": "data/wrfout_d02_3km.nc",
        "geo_em": "data/geo_em_d02_3km.nc",
        "resolution": "3 km"
    },
    "1km": {
        "wrfout": "data/wrfout_d03_1km.nc",
        "geo_em": "data/geo_em_d03_1km.nc",
        "resolution": "1 km"
    },
    "333m": {
        "wrfout": "data/wrfout_d04_333m.nc",
        "geo_em": "data/geo_em_d04_333m.nc",
        "resolution": "333 m"
    }
}


def generate_synthetic_terrain_data(domain_key):
    """Generar dades sintètiques de terreny per a fins de demostració."""
    if domain_key == "9km":
        nx, ny = 40, 40
        terrain_scale = 1000
    elif domain_key == "3km":
        nx, ny = 60, 60
        terrain_scale = 1200
    elif domain_key == "1km":
        nx, ny = 80, 80
        terrain_scale = 1400
    else:  # 333m
        nx, ny = 120, 120
        terrain_scale = 1600
    
    # Crear arrays de coordenades centrats a l'estació DAS
    lat_range = np.linspace(DAS_LAT - 0.5, DAS_LAT + 0.5, ny)
    lon_range = np.linspace(DAS_LON - 0.5, DAS_LON + 0.5, nx)
    
    lons, lats = np.meshgrid(lon_range, lat_range)
    
    # Generar terreny sintètic de vall (més baix al centre, més alt als marges)
    # Per a la vall de la Cerdanya, crear una vall que corre d'est a oest
    dist_from_center_lat = (lats - DAS_LAT) ** 2
    dist_from_valley_axis = dist_from_center_lat * 5  
    
    # Elevació base amb estructura de vall
    base_elevation = 1200
    terrain = base_elevation + terrain_scale * dist_from_valley_axis
    
    # Afegir una mica de variació aleatòria per realisme
    np.random.seed(42)
    terrain += np.random.normal(0, terrain_scale * 0.1, terrain.shape)
    
    # Suavitzar el terreny per a resolucions més gruixudes
    from scipy.ndimage import gaussian_filter
    if domain_key == "9km":
        terrain = gaussian_filter(terrain, sigma=3)
    elif domain_key == "3km":
        terrain = gaussian_filter(terrain, sigma=2)
    elif domain_key == "1km":
        terrain = gaussian_filter(terrain, sigma=1)
    
    # Crear xarray Dataset
    ds = xr.Dataset(
        {
            "HGT_M": (["south_north", "west_east"], terrain),
            "XLAT_M": (["south_north", "west_east"], lats),
            "XLONG_M": (["south_north", "west_east"], lons),
        }
    )
    
    return ds


def generate_synthetic_wrf_data(domain_key, variable):
    """Generar dades sintètiques de sortida WRF per a fins de demostració."""
    # Crear array de temps (24 hores)
    start_time = datetime(2024, 1, 15, 0, 0)
    times = pd.date_range(start_time, periods=24, freq='H')
    
    # Generar dades sintètiques de temperatura o vent
    if variable == "Temperature":
        hours = np.arange(24)
        base_temp = 275  
        diurnal_amp = 8  
        
        # Afegir bias basat en la resolució 
        if domain_key == "9km":
            bias = 5  
        elif domain_key == "3km":
            bias = 2
        elif domain_key == "1km":
            bias = 0.5
        else:  # 333m
            bias = 0.1
        
        temp_values = base_temp + diurnal_amp * np.sin(2 * np.pi * (hours - 6) / 24) + bias
        temp_values += np.random.normal(0, 0.5, len(times))
        
        data = temp_values
        var_name = "T2"
    else:  # vent
        base_wind = 3.0  # m/s
        
        # Afegir soroll basat en la resolució
        if domain_key == "9km":
            noise = 1.5
        elif domain_key == "3km":
            noise = 1.0
        elif domain_key == "1km":
            noise = 0.6
        else:  # 333m
            noise = 0.3
        
        wind_values = base_wind + np.random.normal(0, noise, len(times))
        wind_values = np.maximum(wind_values, 0)  
        
        data = wind_values
        var_name = "WSPD"
    
    # Crear DataFrame
    df = pd.DataFrame({
        'time': times,
        var_name: data
    })
    
    return df


def generate_synthetic_observations(variable):
    """Generar dades sintètiques d'observacions per a fins de demostració."""
    # Crear array de temps (24 hores)
    start_time = datetime(2024, 1, 15, 0, 0)
    times = pd.date_range(start_time, periods=24, freq='H')
    
    if variable == "Temperature":
        # "True" temperatura amb cicle diürn (sense bias)
        hours = np.arange(24)
        base_temp = 275 
        diurnal_amp = 8
        
        temp_values = base_temp + diurnal_amp * np.sin(2 * np.pi * (hours - 6) / 24)
        temp_values += np.random.normal(0, 0.3, len(times)) 
        
        data = temp_values
        var_name = "T2"
    else:  # vent
        base_wind = 3.0
        wind_values = base_wind + np.random.normal(0, 0.2, len(times))
        wind_values = np.maximum(wind_values, 0)
        
        data = wind_values
        var_name = "WSPD"
    
    # Crear DataFrame
    df = pd.DataFrame({
        'time': times,
        var_name: data
    })
    
    return df


@st.cache_data
def load_terrain_data(domain_key):
    """
    Carregar dades de terreny des del fitxer geo_em utilitzant xarray.
    Si el fitxer no existeix, generar dades sintètiques per a la demostració.
    """
    geo_file = DOMAINS[domain_key]["geo_em"]
    
    if Path(geo_file).exists():
        ds = xr.open_dataset(geo_file)
        # Els fitxers geo_em de WRF normalment tenen les variables HGT_M, XLAT_M, XLONG_M
        return ds
    else:
        # Dades sintètquiques
        st.warning(f"Fitxer de terreny no trobat: {geo_file}. S'utilitzen dades sintètiques per a la demostració.")
        return generate_synthetic_terrain_data(domain_key)


@st.cache_data
def load_wrf_data(domain_key, variable):
    """
    Carregar dades de sortida de WRF utilitzant xarray i extreure sèries temporals a l'estació Das.
    Si el fitxer no existeix, generar dades sintètiques.
    """
    wrf_file = DOMAINS[domain_key]["wrfout"]
    
    if Path(wrf_file).exists():
        ds = xr.open_dataset(wrf_file)
        
        # Coordenades
        lats = ds['XLAT'].isel(Time=1).values
        lons = ds['XLONG'].isel(Time=1).values
        
        # Trobar el punt de la graella més proper a l'estació Das
        distances = np.sqrt((lats - DAS_LAT)**2 + (lons - DAS_LON)**2)
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        
        # Extreure sèries temporals en aquest punt
        if variable == "Temperature":
            var_data = ds['T2'][:, min_idx[0], min_idx[1]].values  # 2m temperature
            var_name = "T2"
        else:  # Vent
            # Calcular la velocitat del vent a partir dels components U i V
            u = ds['U10'][:, min_idx[0], min_idx[1]].values
            v = ds['V10'][:, min_idx[0], min_idx[1]].values
            var_data = np.sqrt(u**2 + v**2)
            var_name = "WSPD"
        
        times = pd.to_datetime(ds['Times'].values)
        
        df = pd.DataFrame({
            'time': times,
            var_name: var_data
        })
        
        return df
    else:
        st.warning(f"Fitxer WRF no trobat: {wrf_file}. S'utilitzen dades sintètiques per a la demostració.")
        return generate_synthetic_wrf_data(domain_key, variable)


@st.cache_data
def load_observations(variable):
    """
    Carregar dades observacionals des d'un CSV utilitzant pandas.
    Si el fitxer no existeix, generar dades sintètiques.
    """
    obs_file = "data/meteocat_DP.csv"
    
    if Path(obs_file).exists():
        df = pd.read_csv(obs_file, parse_dates=['time'])
        return df
    else:
        st.warning(f"Fitxer d'observacions no trobat: {obs_file}. S'utilitzen dades sintètiques per a la demostració.")
        return generate_synthetic_observations(variable)


def calculate_rmse(obs, model):
    """Calcular l'error quadràtic mitjà entre observacions i model."""
    return np.sqrt(np.mean((obs - model)**2))


def calculate_bias(obs, model):
    """Calcular el biaix mitjà (model - observacions)."""
    return np.mean(model - obs)


def create_terrain_map(terrain_ds):
    """Crear un mapa matplotlib mostrant l'altitud del terreny amb l'estació Das marcada."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extreure dades de terreny
    hgt = terrain_ds['HGT_M'].values
    lats = terrain_ds['XLAT_M'].values
    lons = terrain_ds['XLONG_M'].values
    
    # Crear gràfic de contorns
    contour = ax.contourf(lons, lats, hgt, levels=20, cmap='terrain')
    ax.contour(lons, lats, hgt, levels=10, colors='black', linewidths=0.5, alpha=0.3)
    
    # Marcar l'estació Das
    ax.plot(DAS_LON, DAS_LAT, 'ro', markersize=10, label=' Das - Aeròdrom [DP]', 
            markeredgewidth=2, markeredgecolor='white')
    
    # Etiquetes i format
    ax.set_xlabel('Longitud [°E]', fontsize=12)
    ax.set_ylabel('Latitud [°N]', fontsize=12)
    ax.set_title("Altitud [msnm]", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, label='Altitud [msnm]')
    
    plt.tight_layout()
    return fig


def create_validation_plot(obs_df, wrf_df, variable):
    """Crear un gràfic de línies Plotly comparant observacions vs dades WRF."""
    fig = go.Figure()
    
    # Determinar el nom de la variable i les unitats
    if variable == "Temperature":
        var_name = "T2"
        y_label = "Temperatura (K)"
        # Convertir a Celsius 
        obs_values = obs_df[var_name].values - 273.15
        wrf_values = wrf_df[var_name].values - 273.15
        y_label = "Temperatura [°C]"
    else:  # Vent
        var_name = "WSPD"
        y_label = "Velocitat del Vent [m/s]"
        obs_values = obs_df[var_name].values
        wrf_values = wrf_df[var_name].values
    
    # Afegir observacions
    fig.add_trace(go.Scatter(
        x=obs_df['time'],
        y=obs_values,
        mode='lines',
        name='Observacions',
        line=dict(color='black', width=2)
    ))
    
    # Add WRF
    fig.add_trace(go.Scatter(
        x=wrf_df['time'],
        y=wrf_values,
        mode='lines',
        name='Model WRF',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=f'Sèries Temporals de {variable} Das - Aeròdrom [DP]',
        xaxis_title='Temps',
        yaxis_title=y_label,
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def main():
    """Aplicació principal de Streamlit."""
    
    # Configuració de la pàgina
    st.set_page_config(
        page_title="Estudi de Cas d'Alta Resolució WRF",
        page_icon="🏔️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Capçalera
    st.title("🏔️ Estudi de Cas d'Alta Resolució: L'Impacte de l'Orografia Local")
    st.markdown("**Visualització de les sortides del model WRF vs Observacions - Estació SMC Das - Aeròdrom [DP]**")
    st.markdown("---")
    
    # Controls de la barra lateral
    st.sidebar.header("Configuració")
    
    # Variable selector
    variable = st.sidebar.selectbox(
        "Selecciona Variable",
        ["Temperature", "Wind"],
        help="Tria la variable meteorològica a visualitzar"
    )
    
    # Domain/Resolution selector
    domain = st.sidebar.selectbox(
        "Selecciona Domini / Resolució",
        ["9km", "3km", "1km", "333m"],
        help="Tria la resolució del model WRF"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Estació SMC Das - Aeròdrom [DP]")
    st.sidebar.markdown(f"**Latitud:** {DAS_LAT}°N")
    st.sidebar.markdown(f"**Longitud:** {DAS_LON}°E")
    st.sidebar.markdown(f"**Ubicació:** Cerdanya, Catalunya")
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Aquesta aplicació demostra com la resolució del model afecta "
        "la representació del terreny complex i les variables meteorològiques."
    )
    
    # Carregar dades
    with st.spinner("Carregant dades..."):
        terrain_ds = load_terrain_data(domain)
        wrf_df = load_wrf_data(domain, variable)
        obs_df = load_observations(variable)
    
    # Disseny principal: Dues columnes
    col1, col2 = st.columns([1, 1])
    
    # Columna esquerra: Context geogràfic
    with col1:
        st.subheader("Context Geogràfic")
        st.markdown(f"**Resolució del Domini:** {DOMAINS[domain]['resolution']}")
        
        # Crear i mostrar mapa del terreny
        terrain_fig = create_terrain_map(terrain_ds)
        st.pyplot(terrain_fig)
        plt.close()  
        
        st.caption("El punt vermell indica la ubicació de l'estació Das a la vall de la Cerdanya")
    
    # Columna dreta: Validació
    with col2:
        st.subheader("📊 Validació del Model")
        
        # Crear i mostrar gràfic de validació
        validation_fig = create_validation_plot(obs_df, wrf_df, variable)
        st.plotly_chart(validation_fig, use_container_width=True)
        
        # Calcular mètriques
        if variable == "Temperature":
            var_name = "T2"
            obs_values = obs_df[var_name].values
            wrf_values = wrf_df[var_name].values
            rmse = calculate_rmse(obs_values, wrf_values)
            bias = calculate_bias(obs_values, wrf_values)
            
            # Mostrar mètriques
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    "RMSE",
                    f"{rmse:.2f} K",
                    help="Root Mean Square Error"
                )
            with metric_col2:
                st.metric(
                    "Bias",
                    f"{bias:.2f} K",
                    delta=f"{bias:.2f} K",
                    delta_color="inverse",
                    help="Mean bias (Model - Observations)"
                )
        else:  # Vent
            var_name = "WSPD"
            obs_values = obs_df[var_name].values
            wrf_values = wrf_df[var_name].values
            rmse = calculate_rmse(obs_values, wrf_values)
            bias = calculate_bias(obs_values, wrf_values)
            
            # Mostrar mètriques
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    "RMSE",
                    f"{rmse:.2f} m/s",
                    help="Root Mean Square Error"
                )
            with metric_col2:
                st.metric(
                    "Bias",
                    f"{bias:.2f} m/s",
                    delta=f"{bias:.2f} m/s",
                    delta_color="inverse",
                    help="Mean bias (Model - Observations)"
                )
    
    # Missatges de storytelling basats en la selecció del domini
    st.markdown("---")
    
    if domain == "333m":
        st.info(
            "**Alta Resolució (333m):** A aquesta resolució, el model pot resoldre "
            "la estructura de la vall de la Cerdanya molt millor. El terreny està representat amb "
            "detalls fins, permetent que el model capturi patrons de circulació locals, "
            "inversions de temperatura i vents de vall amb més precisió. Això condueix a "
            "previsions de temperatura i vent significativament millorades."
        )
    elif domain == "9km":
        st.warning(
            "**Baixa Resolució (9km):** A aquesta resolució, el model veu la vall "
            "com un terreny essencialment pla. Les característiques muntanyoses dels Pirineus i "
            "l'estructura de la vall no es poden representar adequadament en la graella del model. Això condueix a "
            "**errors significatius de temperatura** (normalment un biaix càlid durant esdeveniments de piscina freda) "
            "i una representació pobre dels patrons locals de vent. Es necessita una resolució més alta per a "
            "previsions precises en terrenys complexos."
        )
    elif domain == "3km":
        st.info(
            "**Resolució Mitjana (3km):** Aquesta resolució comença a capturar algunes característiques de la vall "
            "però encara perd detalls importants. La representació del terreny és "
            "millorada en comparació amb 9km però continua sent suavitzada. Les previsions de temperatura i vent "
            "són millors però encara mostren errors notables en topografies complexes."
        )
    else:  # 1km
        st.info(
            "**Resolució Intermèdia (1km):** A 1km de resolució, el model captura "
            "gran part de l'estructura de la vall. Les característiques del terreny estan ben representades, i els processos locals de la meteorologia són millor simulats."
            "meteorological processes are better simulated. This resolution offers a good "
            "balance between computational cost and accuracy for mountain meteorology."
        )
    
    # Peu de pàgina
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Arnau Toledano | UOC - Visualització de dades | 2026"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
