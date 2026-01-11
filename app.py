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

# Das station coordinates
DAS_LAT = 42.38
DAS_LON = 1.86

# Domain configurations
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
    """Generate synthetic terrain data for demonstration purposes."""
    # Grid size depends on resolution
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
    
    # Create coordinate arrays centered on Das station
    lat_range = np.linspace(DAS_LAT - 0.5, DAS_LAT + 0.5, ny)
    lon_range = np.linspace(DAS_LON - 0.5, DAS_LON + 0.5, nx)
    
    lons, lats = np.meshgrid(lon_range, lat_range)
    
    # Generate synthetic valley terrain (lower in center, higher on edges)
    # For Cerdanya valley, create a valley running E-W
    dist_from_center_lat = (lats - DAS_LAT) ** 2
    dist_from_valley_axis = dist_from_center_lat * 5  # Valley runs E-W
    
    # Base elevation with valley structure
    base_elevation = 1200
    terrain = base_elevation + terrain_scale * dist_from_valley_axis
    
    # Add some random variation for realism
    np.random.seed(42)
    terrain += np.random.normal(0, terrain_scale * 0.1, terrain.shape)
    
    # Smooth terrain for coarser resolutions
    from scipy.ndimage import gaussian_filter
    if domain_key == "9km":
        terrain = gaussian_filter(terrain, sigma=3)
    elif domain_key == "3km":
        terrain = gaussian_filter(terrain, sigma=2)
    elif domain_key == "1km":
        terrain = gaussian_filter(terrain, sigma=1)
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "HGT_M": (["south_north", "west_east"], terrain),
            "XLAT_M": (["south_north", "west_east"], lats),
            "XLONG_M": (["south_north", "west_east"], lons),
        }
    )
    
    return ds


def generate_synthetic_wrf_data(domain_key, variable):
    """Generate synthetic WRF output data for demonstration purposes."""
    # Create time array (24 hours)
    start_time = datetime(2024, 1, 15, 0, 0)
    times = pd.date_range(start_time, periods=24, freq='H')
    
    # Generate synthetic temperature or wind data
    if variable == "Temperature":
        # Temperature with diurnal cycle
        hours = np.arange(24)
        base_temp = 275  # Kelvin (about 2°C)
        diurnal_amp = 8  # K
        
        # Add bias based on resolution (coarser = more error)
        if domain_key == "9km":
            bias = 5  # Significant warm bias
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
    else:  # Wind
        # Wind speed with some variation
        base_wind = 3.0  # m/s
        
        # Add noise based on resolution
        if domain_key == "9km":
            noise = 1.5
        elif domain_key == "3km":
            noise = 1.0
        elif domain_key == "1km":
            noise = 0.6
        else:  # 333m
            noise = 0.3
        
        wind_values = base_wind + np.random.normal(0, noise, len(times))
        wind_values = np.maximum(wind_values, 0)  # No negative wind
        
        data = wind_values
        var_name = "WSPD"
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': times,
        var_name: data
    })
    
    return df


def generate_synthetic_observations(variable):
    """Generate synthetic observational data for demonstration purposes."""
    # Create time array (24 hours)
    start_time = datetime(2024, 1, 15, 0, 0)
    times = pd.date_range(start_time, periods=24, freq='H')
    
    if variable == "Temperature":
        # "True" temperature with diurnal cycle (no bias)
        hours = np.arange(24)
        base_temp = 275  # Kelvin
        diurnal_amp = 8
        
        temp_values = base_temp + diurnal_amp * np.sin(2 * np.pi * (hours - 6) / 24)
        temp_values += np.random.normal(0, 0.3, len(times))  # Small measurement error
        
        data = temp_values
        var_name = "T2"
    else:  # Wind
        # "True" wind speed
        base_wind = 3.0
        wind_values = base_wind + np.random.normal(0, 0.2, len(times))
        wind_values = np.maximum(wind_values, 0)
        
        data = wind_values
        var_name = "WSPD"
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': times,
        var_name: data
    })
    
    return df


@st.cache_data
def load_terrain_data(domain_key):
    """
    Load terrain data from geo_em file using xarray.
    If file doesn't exist, generate synthetic data.
    """
    geo_file = DOMAINS[domain_key]["geo_em"]
    
    if Path(geo_file).exists():
        ds = xr.open_dataset(geo_file)
        # WRF geo_em files typically have HGT_M, XLAT_M, XLONG_M variables
        return ds
    else:
        # Generate synthetic data for demonstration
        st.warning(f"Terrain file not found: {geo_file}. Using synthetic data for demonstration.")
        return generate_synthetic_terrain_data(domain_key)


@st.cache_data
def load_wrf_data(domain_key, variable):
    """
    Load WRF output data using xarray and extract time series at Das station.
    If file doesn't exist, generate synthetic data.
    """
    wrf_file = DOMAINS[domain_key]["wrfout"]
    
    if Path(wrf_file).exists():
        ds = xr.open_dataset(wrf_file)
        
        # Get coordinates
        lats = ds['XLAT'].isel(Time=0).values
        lons = ds['XLONG'].isel(Time=0).values
        
        # Find nearest grid point to Das station
        distances = np.sqrt((lats - DAS_LAT)**2 + (lons - DAS_LON)**2)
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        
        # Extract time series at that point
        if variable == "Temperature":
            var_data = ds['T2'][:, min_idx[0], min_idx[1]].values  # 2m temperature
            var_name = "T2"
        else:  # Wind
            # Calculate wind speed from U and V components
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
        # Generate synthetic data for demonstration
        st.warning(f"WRF file not found: {wrf_file}. Using synthetic data for demonstration.")
        return generate_synthetic_wrf_data(domain_key, variable)


@st.cache_data
def load_observations(variable):
    """
    Load observational data from CSV using pandas.
    If file doesn't exist, generate synthetic data.
    """
    obs_file = "data/das_observations.csv"
    
    if Path(obs_file).exists():
        df = pd.read_csv(obs_file, parse_dates=['time'])
        return df
    else:
        # Generate synthetic data for demonstration
        st.warning(f"Observation file not found: {obs_file}. Using synthetic data for demonstration.")
        return generate_synthetic_observations(variable)


def calculate_rmse(obs, model):
    """Calculate Root Mean Square Error between observations and model."""
    return np.sqrt(np.mean((obs - model)**2))


def calculate_bias(obs, model):
    """Calculate mean bias (model - observations)."""
    return np.mean(model - obs)


def create_terrain_map(terrain_ds):
    """Create a matplotlib map showing terrain height with Das station marked."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract terrain data
    hgt = terrain_ds['HGT_M'].values
    lats = terrain_ds['XLAT_M'].values
    lons = terrain_ds['XLONG_M'].values
    
    # Create contour plot
    contour = ax.contourf(lons, lats, hgt, levels=20, cmap='terrain')
    ax.contour(lons, lats, hgt, levels=10, colors='black', linewidths=0.5, alpha=0.3)
    
    # Mark Das station
    ax.plot(DAS_LON, DAS_LAT, 'ro', markersize=10, label='Das Station', 
            markeredgewidth=2, markeredgecolor='white')
    
    # Labels and formatting
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title('Terrain Height (m)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, label='Height (m)')
    
    plt.tight_layout()
    return fig


def create_validation_plot(obs_df, wrf_df, variable):
    """Create a Plotly line chart comparing observations vs WRF data."""
    fig = go.Figure()
    
    # Determine variable name and units
    if variable == "Temperature":
        var_name = "T2"
        y_label = "Temperature (K)"
        # Convert to Celsius for display
        obs_values = obs_df[var_name].values - 273.15
        wrf_values = wrf_df[var_name].values - 273.15
        y_label = "Temperature (°C)"
    else:  # Wind
        var_name = "WSPD"
        y_label = "Wind Speed (m/s)"
        obs_values = obs_df[var_name].values
        wrf_values = wrf_df[var_name].values
    
    # Add observation trace
    fig.add_trace(go.Scatter(
        x=obs_df['time'],
        y=obs_values,
        mode='lines',
        name='Observations',
        line=dict(color='black', width=2)
    ))
    
    # Add WRF trace
    fig.add_trace(go.Scatter(
        x=wrf_df['time'],
        y=wrf_values,
        mode='lines',
        name='WRF Model',
        line=dict(color='blue', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{variable} Time Series - Das Station',
        xaxis_title='Time',
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
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="WRF High Resolution Case Study",
        page_icon="🏔️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🏔️ High Resolution Case Study: The Impact of Local Orography")
    st.markdown("**Visualizing WRF Model Outputs vs Observations - Das Station, Cerdanya**")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    
    # Variable selector
    variable = st.sidebar.selectbox(
        "Select Variable",
        ["Temperature", "Wind"],
        help="Choose the meteorological variable to visualize"
    )
    
    # Domain/Resolution selector
    domain = st.sidebar.selectbox(
        "Select Domain / Resolution",
        ["9km", "3km", "1km", "333m"],
        help="Choose the WRF model resolution"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Das Station")
    st.sidebar.markdown(f"**Latitude:** {DAS_LAT}°N")
    st.sidebar.markdown(f"**Longitude:** {DAS_LON}°E")
    st.sidebar.markdown(f"**Location:** Cerdanya Valley")
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application demonstrates how model resolution affects "
        "the representation of complex terrain and meteorological variables."
    )
    
    # Load data
    with st.spinner("Loading data..."):
        terrain_ds = load_terrain_data(domain)
        wrf_df = load_wrf_data(domain, variable)
        obs_df = load_observations(variable)
    
    # Main layout: Two columns
    col1, col2 = st.columns([1, 1])
    
    # Left column: Geographic context
    with col1:
        st.subheader("🗺️ Geographic Context")
        st.markdown(f"**Domain Resolution:** {DOMAINS[domain]['resolution']}")
        
        # Create and display terrain map
        terrain_fig = create_terrain_map(terrain_ds)
        st.pyplot(terrain_fig)
        plt.close()  # Clean up
        
        st.caption("Red dot indicates Das station location in the Cerdanya valley")
    
    # Right column: Validation
    with col2:
        st.subheader("📊 Model Validation")
        
        # Create and display validation plot
        validation_fig = create_validation_plot(obs_df, wrf_df, variable)
        st.plotly_chart(validation_fig, use_container_width=True)
        
        # Calculate metrics
        if variable == "Temperature":
            var_name = "T2"
            obs_values = obs_df[var_name].values
            wrf_values = wrf_df[var_name].values
            rmse = calculate_rmse(obs_values, wrf_values)
            bias = calculate_bias(obs_values, wrf_values)
            
            # Display metrics
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
        else:  # Wind
            var_name = "WSPD"
            obs_values = obs_df[var_name].values
            wrf_values = wrf_df[var_name].values
            rmse = calculate_rmse(obs_values, wrf_values)
            bias = calculate_bias(obs_values, wrf_values)
            
            # Display metrics
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
    
    # Storytelling messages based on domain selection
    st.markdown("---")
    
    if domain == "333m":
        st.info(
            "✅ **High Resolution (333m):** At this resolution, the model can resolve "
            "the Cerdanya valley structure much better. The terrain is represented with "
            "fine details, allowing the model to capture local circulation patterns, "
            "temperature inversions, and valley winds more accurately. This leads to "
            "significantly improved temperature and wind forecasts."
        )
    elif domain == "9km":
        st.warning(
            "⚠️ **Coarse Resolution (9km):** At this resolution, the model sees the valley "
            "as essentially flat terrain. The mountainous features of the Pyrenees and the "
            "valley structure cannot be properly represented in the model grid. This leads to "
            "**significant temperature errors** (typically warm bias during cold pool events) "
            "and poor representation of local wind patterns. Higher resolution is needed for "
            "accurate forecasts in complex terrain."
        )
    elif domain == "3km":
        st.info(
            "ℹ️ **Medium Resolution (3km):** This resolution starts to capture some valley "
            "features but still misses important details. The representation of terrain is "
            "improved compared to 9km but remains smoothed. Temperature and wind forecasts "
            "are better but still show noticeable errors in complex topography."
        )
    else:  # 1km
        st.info(
            "ℹ️ **Intermediate Resolution (1km):** At 1km resolution, the model captures "
            "much of the valley structure. Terrain features are well represented, and local "
            "meteorological processes are better simulated. This resolution offers a good "
            "balance between computational cost and accuracy for mountain meteorology."
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Arnau Toledano | UOC - Visualització de dades | 2024"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
