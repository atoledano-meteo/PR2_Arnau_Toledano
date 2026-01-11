# High Resolution Case Study: The Impact of Local Orography

Segona part de la pràctica de l'assignatura de Visualització de dades de la UOC.

## Project Description

This Streamlit application visualizes WRF (Weather Research and Forecasting) model outputs compared to observational data for the Das station in Cerdanya, demonstrating how model resolution affects the representation of complex terrain and meteorological variables.

## Features

- Interactive visualization of WRF model outputs at multiple resolutions (9km, 3km, 1km, 333m)
- Comparison with observational data from Das station
- Terrain height visualization showing orographic features
- Time series validation plots with RMSE metrics
- Educational storytelling elements explaining resolution impacts

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/atoledano-meteo/PR2_Arnau_Toledano.git
   cd PR2_Arnau_Toledano
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Ensure your virtual environment is activated
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

## Data Structure

The application expects data in the following structure:
```
data/
├── wrfout_d01_9km.nc       # WRF output at 9km resolution
├── wrfout_d02_3km.nc       # WRF output at 3km resolution
├── wrfout_d03_1km.nc       # WRF output at 1km resolution
├── wrfout_d04_333m.nc      # WRF output at 333m resolution
├── geo_em_d01_9km.nc       # Terrain data at 9km resolution
├── geo_em_d02_3km.nc       # Terrain data at 3km resolution
├── geo_em_d03_1km.nc       # Terrain data at 1km resolution
├── geo_em_d04_333m.nc      # Terrain data at 333m resolution
└── das_observations.csv    # Observational data from Das station
```

**Note:** If actual data files are not available, the application will generate synthetic/mock data for demonstration purposes.

## Application Structure

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `data/` - Data directory for NetCDF and CSV files
- `README.md` - This file

## Usage

1. Use the **sidebar** to select:
   - Variable to visualize (Temperature or Wind)
   - Domain/Resolution (9km, 3km, 1km, or 333m)

2. View the **left column** for:
   - Geographic context with terrain height map
   - Das station location marked in red

3. View the **right column** for:
   - Time series comparison plot (Observations vs WRF)
   - RMSE metric for model validation

4. Read the **info messages** that explain how resolution affects model accuracy

## Das Station Information

- **Location:** Cerdanya valley, Catalonia
- **Latitude:** 42.38°N
- **Longitude:** 1.86°E
- **Purpose:** Validation point for high-resolution WRF simulations

## Technical Details

- Built with Streamlit for interactive visualization
- Uses xarray for efficient NetCDF data handling
- Implements caching (@st.cache_data) for performance
- Plotly for interactive time series plots
- Matplotlib for geographic visualizations

## License

See LICENSE file for details.

## Author

Arnau Toledano - UOC Visualització de dades
