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
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ==========================================
# CONFIGURACIÓ I CONSTANTS
# ==========================================

# Coordenades de l'estació SMC Das - Aeròdrom [DP]
DAS_LAT = 42.38605
DAS_LON = 1.86640

# Configuració dels fitxers del model
DOMAINS = {
    "9km": {
        "wrfout": "data/wrfout_d01_2024-03-01_06:00:00",
        "geo_em": "data/geo_em.d01.nc",
        "resolution": "9 km"
    },
    "3km": {
        "wrfout": "data/wrfout_d02_2024-03-01_06:00:00",
        "geo_em": "data/geo_em.d02.nc",
        "resolution": "3 km"
    },
    "1km": {
        "wrfout": "data/wrfout_d03_2024-03-01_06:00:00",
        "geo_em": "data/geo_em.d03.nc",
        "resolution": "1 km"
    },
    "333m": {
        "wrfout": "data/wrfout_d04_2024-03-01_06:00:00",
        "geo_em": "data/geo_em.d04.nc",
        "resolution": "333 m"
    }
}

# ==========================================
# FUNCIONS DE CÀRREGA DE DADES
# ==========================================

def generate_synthetic_data_fallback(domain_key, type_data):
    """Fallback de dades sintètiques."""
    if type_data == 'terrain':
        nx, ny = 50, 50
        lat_range = np.linspace(DAS_LAT - 0.5, DAS_LAT + 0.5, ny)
        lon_range = np.linspace(DAS_LON - 0.5, DAS_LON + 0.5, nx)
        lons, lats = np.meshgrid(lon_range, lat_range)
        terrain = 1000 + 500 * np.exp(-((lats - DAS_LAT)**2 + (lons - DAS_LON)**2)/0.1)
        ds = xr.Dataset({
            "HGT_M": (["south_north", "west_east"], terrain),
            "XLAT_M": (["south_north", "west_east"], lats),
            "XLONG_M": (["south_north", "west_east"], lons),
        })
        return ds
    
    elif type_data == 'wrf':
        times = pd.date_range("2024-03-01 06:00", periods=24, freq='h')
        df = pd.DataFrame({
            'datetime': times,
            'T2': 280 + np.random.randn(24) * 5,
            'WSPD': 5 + np.random.randn(24) * 2
        })
        return df

    return None

@st.cache_data
def load_terrain_data(domain_key):
    """Carrega el fitxer geo_em."""
    geo_file = DOMAINS[domain_key]["geo_em"]
    
    if Path(geo_file).exists():
        ds = xr.open_dataset(geo_file)
        return ds
    else:
        st.warning(f"⚠️ No s'ha trobat el fitxer {geo_file}. Usant dades sintètiques.")
        return generate_synthetic_data_fallback(domain_key, 'terrain')

@st.cache_data
def load_wrf_data(domain_key):
    """Carrega dades WRF i converteix dates."""
    wrf_file = DOMAINS[domain_key]["wrfout"]
    
    if Path(wrf_file).exists():
        ds = xr.open_dataset(wrf_file)
        
        # 1. Coordenades i punt més proper
        lats = ds['XLAT'].isel(Time=0).values
        lons = ds['XLONG'].isel(Time=0).values
        dist_sq = (lats - DAS_LAT)**2 + (lons - DAS_LON)**2
        min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        iy, ix = min_idx
        
        # 2. Variables
        t2 = ds['T2'][:, iy, ix].values
        u10 = ds['U10'][:, iy, ix].values
        v10 = ds['V10'][:, iy, ix].values
        wspd = np.sqrt(u10**2 + v10**2)
        
        # 3. Dates (Descodificar bytes + format)
        raw_times = ds['Times'].values
        decoded_times = [str(t.decode("utf-8")) for t in raw_times]
        times = pd.to_datetime(decoded_times, format='%Y-%m-%d_%H:%M:%S')
        
        df = pd.DataFrame({
            'datetime': times,
            'T2': t2,
            'WSPD': wspd
        })
        
        return df
    else:
        st.warning(f"⚠️ No s'ha trobat el fitxer {wrf_file}. Usant dades sintètiques.")
        return generate_synthetic_data_fallback(domain_key, 'wrf')

@st.cache_data
def load_observations():
    """Carrega observacions i normalitza la zona horària."""
    obs_file = "data/meteocat_DP.csv"
    
    if Path(obs_file).exists():
        df = pd.read_csv(obs_file, parse_dates=['datetime'])
        
        # --- CORRECCIÓ DE ZONA HORÀRIA ---
        # Si la columna té zona horària (UTC), l'eliminem per fer-la compatible amb WRF
        if df['datetime'].dt.tz is not None:
            df['datetime'] = df['datetime'].dt.tz_localize(None)
        # ---------------------------------
        
        # Reanomenar columnes
        df = df.rename(columns={
            'T2_obs': 'T2',           
            'vent_velocitat': 'WSPD'
        })
        return df
    else:
        st.error("Fitxer d'observacions no trobat.")
        return pd.DataFrame({'datetime': [], 'T2': [], 'WSPD': []})

# ==========================================
# CÀLCUL DE MÈTRIQUES
# ==========================================

def calculate_rmse(obs, model):
    return np.sqrt(np.mean((obs - model)**2))

def calculate_bias(obs, model):
    return np.mean(model - obs)

# ==========================================
# VISUALITZACIÓ
# ==========================================

def create_terrain_map(terrain_ds):
    """Mapa de contorns de l'orografia (amb .squeeze)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    hgt = terrain_ds['HGT_M'].squeeze().values
    lats = terrain_ds['XLAT_M'].squeeze().values
    lons = terrain_ds['XLONG_M'].squeeze().values
    
    contour = ax.contourf(lons, lats, hgt, levels=20, cmap='terrain')
    ax.contour(lons, lats, hgt, levels=10, colors='black', linewidths=0.5, alpha=0.3)
    
    ax.plot(DAS_LON, DAS_LAT, 'ro', markersize=10, markeredgewidth=2, 
            markeredgecolor='white', label='Das - Aeròdrom [DP]')
    
    ax.set_title("Topografia del Model [msnm]", fontsize=14, fontweight='bold')
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend()
    plt.colorbar(contour, ax=ax, label='Altitud (m)')
    
    return fig

def create_validation_plot(obs_df, wrf_df, variable):
    """Gràfic interactiu amb Plotly (Centrat en el període del model)"""
    fig = go.Figure()
    
    if variable == "Temperature":
        obs_y = obs_df['T2']
        wrf_y = wrf_df['T2'] - 273.15
        unit = "°C"
        title_var = "Temperatura"
    else:
        obs_y = obs_df['WSPD']
        wrf_y = wrf_df['WSPD']
        unit = "m/s"
        title_var = "Velocitat del Vent"

    # Afegim les Observacions
    fig.add_trace(go.Scatter(
        x=obs_df['datetime'], y=obs_y,
        mode='lines', name='Observació (SMC)',
        line=dict(color='black', width=2)
    ))
    
    # Afegim el Model
    fig.add_trace(go.Scatter(
        x=wrf_df['datetime'], y=wrf_y,
        mode='lines', name='Model WRF',
        line=dict(color='blue', width=2)
    ))
    
    # Agafem la data d'inici i final del MODEL WRF
    start_time = wrf_df['datetime'].min()
    end_time = wrf_df['datetime'].max()
    
    fig.update_layout(
        title=f"Validació: {title_var} a Das",
        yaxis_title=f"{title_var} [{unit}]",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis_range=[start_time, end_time]  
    )
    return fig

# ==========================================
# MAIN APP
# ==========================================

def main():
    st.set_page_config(page_title="WRF Alta Resolució", page_icon="🏔️", layout="wide")
    
    st.title("🏔️ Impacte de l'Orografia: WRF vs Observacions")
    st.markdown("**Cas d'Estudi: La Cerdanya (Estació Das - Aeròdrom)**")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Configuració")
        variable_sel = st.selectbox("Variable", ["Temperature", "Wind"])
        domain_sel = st.selectbox("Resolució del Model", list(DOMAINS.keys()))
        
        st.markdown("---")
        st.info(f"**Estació:** Das [DP]\n**Lat:** {DAS_LAT}\n**Lon:** {DAS_LON}")

    with st.spinner("Carregant i processant dades..."):
        terrain_ds = load_terrain_data(domain_sel)
        wrf_df = load_wrf_data(domain_sel)
        obs_df = load_observations()

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"Representació del Terreny ({DOMAINS[domain_sel]['resolution']})")
        fig_map = create_terrain_map(terrain_ds)
        st.pyplot(fig_map)
        plt.close()
        
    with col2:
        st.subheader("Validació Temporal")
        
        fig_val = create_validation_plot(obs_df, wrf_df, variable_sel)
        st.plotly_chart(fig_val, use_container_width=True)
        
        # --- CÀLCUL CIENTÍFIC DE MÈTRIQUES (MERGE) ---
        var_col = 'T2' if variable_sel == "Temperature" else 'WSPD'
        
        df_obs_clean = obs_df[['datetime', var_col]].rename(columns={var_col: 'obs'})
        df_wrf_clean = wrf_df[['datetime', var_col]].rename(columns={var_col: 'model'})
        
        # Merge inner join (ara funcionarà perquè les dates són compatibles)
        merged = pd.merge(df_obs_clean, df_wrf_clean, on='datetime', how='inner')
        
        if not merged.empty:
            obs_vals = merged['obs'].values
            mod_vals = merged['model'].values
            
            if variable_sel == "Temperature":
                mod_vals = mod_vals - 273.15
                unit_label = "°C"
            else:
                unit_label = "m/s"
            
            rmse = calculate_rmse(obs_vals, mod_vals)
            bias = calculate_bias(obs_vals, mod_vals)
            
            m1, m2 = st.columns(2)
            m1.metric("RMSE (Error)", f"{rmse:.2f} {unit_label}")
            m2.metric("Bias (Model - Obs)", f"{bias:.2f} {unit_label}", delta_color="inverse")
            
            st.caption(f"Mètriques calculades sobre {len(merged)} punts coincidents.")
        else:
            st.error("No hi ha coincidència temporal entre model i observacions.")

    st.markdown("---")
    if domain_sel == "333m":
        st.success("✅ **Alta Resolució (333m):** El model captura la vall.")
    elif domain_sel == "9km":
        st.warning("⚠️ **Baixa Resolució (9km):** El model veu la Cerdanya plana.")
    else:
        st.info(f"ℹ️ **Resolució Intermèdia ({domain_sel}):** Detall moderat.")

if __name__ == "__main__":
    main()