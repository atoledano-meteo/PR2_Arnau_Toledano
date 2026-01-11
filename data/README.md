# Data Directory

Les dades de sortida del WPS i el WRF (`wrfout*` i `geo_em*`) són excluídes al `.gitignore` perquè pesen molt.

## Dades de sortida del WRF (NetCDF)
- `wrfout_d01_2024-03-01_06:00:00` - Dades a 9km de resolució espacial horitzontal del domini 1
- `wrfout_d02_2024-03-01_06:00:00` - Dades a 3km de resolució espacial horitzontal del domini 2
- `wrfout_d03_2024-03-01_06:00:00` - Dades a 1km de resolució espacial horitzontal del domini 3
- `wrfout_d04_2024-03-01_06:00:00` - Dades a 333m de resolució espacial horitzontal del domini 4

## Dades estàtiques del WPS (NetCDF)
- `geo_em_d01.nc` - Dades a 9km de resolució espacial horitzontal del domini 1
- `geo_em_d02.nc` - Dades a 3km de resolució espacial horitzontal del domini 2
- `geo_em_d03.nc` - Dades a 1km de resolució espacial horitzontal del domini 3
- `geo_em_d04.nc` - Dades a 333m de resolució espacial horitzontal del domini 4

## Dades observacionals (CSV)
- `metecat_DP.csv` - Observacions de l'estació del SMC

Si totes aquestes dades no són presents, l'aplicació generarà dades sintètiques per demostrar el funcionament. Els fitxers de sortida wrfouts pesen molt i no entren al registre de Git.

## Informació de l'estació de Das - Aeròdrom [DP]

- **Latitud:** 42,38605
- **Longitud:** 1,86640
- **Altitud:** 1097 msnm
- **Localització:** Cerdanya, Catalunya
