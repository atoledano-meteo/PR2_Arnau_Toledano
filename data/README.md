# Data Directory

This directory should contain the following data files for the WRF visualization application:

## WRF Output Files (NetCDF)
- `wrfout_d01_9km.nc` - WRF model output at 9km resolution
- `wrfout_d02_3km.nc` - WRF model output at 3km resolution
- `wrfout_d03_1km.nc` - WRF model output at 1km resolution
- `wrfout_d04_333m.nc` - WRF model output at 333m resolution

## Terrain/Geographic Files (NetCDF)
- `geo_em_d01_9km.nc` - Terrain elevation data at 9km resolution
- `geo_em_d02_3km.nc` - Terrain elevation data at 3km resolution
- `geo_em_d03_1km.nc` - Terrain elevation data at 1km resolution
- `geo_em_d04_333m.nc` - Terrain elevation data at 333m resolution

## Observational Data (CSV)
- `das_observations.csv` - Observational data from Das station in Cerdanya

### Expected CSV Format
The observational CSV file should have the following columns:
- `time` - Timestamp (ISO format)
- `T2` - 2-meter temperature (K)
- `WSPD` - Wind speed (m/s)

## Note
If these files are not present, the application will automatically generate synthetic data for demonstration purposes. This allows the application to run without requiring the actual large NetCDF files.

## Das Station Information
- **Latitude:** 42.38°N
- **Longitude:** 1.86°E
- **Location:** Cerdanya Valley, Catalonia
