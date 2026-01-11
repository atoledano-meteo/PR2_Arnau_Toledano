# Cas d'estudi: Impacte de l'orografia local

Segona part de la pràctica de l'assignatura de Visualització de dades de la UOC.

## Descripció del projecte

Aquesta aplicació Streamlit visualitza els resultats del model WRF (Weather Research and Forecasting) en comparació amb les dades d'observació de l'estació Das a la Cerdanya, demostrant com la resolució del model afecta la representació de variables meteorològiques i de terreny complexes.

## Característiques

- Visualització interactiva de les sortides del model WRF a múltiples resolucions (9 km, 3 km, 1 km, 333 m)
- Comparació amb dades d'observació de l'estació Das
- Visualització de l'alçada del terreny que mostra les característiques orogràfiques
- Gràfics de validació de sèries temporals amb mètriques RMSE
- Elements narratius educatius que expliquen els impactes de la resolució

## Configuració

### Requisits

- Python 3.8 o una versió més recent
- Gestió de paquets `pip`

### Instal·lació

1. Clonar el repositori:

   ```bash
   git clone https://github.com/atoledano-meteo/PR2_Arnau_Toledano.git
   cd PR2_Arnau_Toledano
   ```

2. Crear un entorn virtual:

   ```bash
   conda create -n pr2_env python=3.11 -y
   ```

3. Activar-lo:

   - macOS/Linux:
     ```bash
     conda activate pr2_env
     ```

4. Instal·lar les dependències:

   ```bash
   pip install -r requirements.txt
   ```

### L'aplicació amb Streamlit

1. Assegura que l'entorn virtual està activat
2. Fes còrrer la app Streamlit:
   ```bash
   streamlit run app.py
   ```
3. Obre un navegador amb l'enllaç que aparegui a la terminal (per exemple `http://localhost:8501`)

## Estructura de les dades

Llegiu el ["README"](/data/README.md) del directori `/data`. 

## Estructura de l'aplicació

- `app.py` - Aplicació Streamlit principal
- `requirements.txt` - Dependències de Python
- `data/` - Directori de dades 
- `README.md` - Aquest fitxer

## Ús

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

## Llicència

["Llicència"](LICENSE) d'ús públic.

## Autor

Arnau Toledano - UOC Visualització de dades 2026. 
