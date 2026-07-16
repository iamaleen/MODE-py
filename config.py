##-----------------------------------------------------------------------------
## Configuración - PATHS Y PARÁMETROS
##-----------------------------------------------------------------------------

import os

# Definimos las rutas 

##-----------------------------------------------------------------------------
# Dominio de 3 km
##-----------------------------------------------------------------------------

# Rutas de figuras
# Rutas de estadisticas
# Rutas de datos del modelo
path_figures = "FIGURES/MODE/"
path_statistics = "FIGURES/MODE/STATISTICS/"
path_wrf = '/home/b1m/Aleen/B1TMET/GUYANA/SCRIPTS/DATASETS/WRF_OUTPUTS/SisPI_hires_2024-03-23_12/'

# Rutas de datos de observación
path_gpm = '/home/b1m/Aleen/B1TMET/GUYANA/SCRIPTS/DATASETS/GPM/GPM_30MIN/SisPI_hires_2024-03-23_12/'

# Definir el intervalo de verificacion (1H, 3H o cada 6H)
accum_window_to_mode='1H'

# Parámetros MODE: ALGORITMO DE MERGING

# Umbral de distancia para considerar el mismo objeto
# Temporal Objects
min_dist_same_object = 10.0

# Umbral de distancia para considerar el mismo objeto
# Spatial Objects
spatial_threshold=180.0                

# Parámetros MODE: AlGORITMO DE MATCHING
# =============================================================================
# MODE-py PARAMETERS
# =============================================================================

MODE_PARAMS = {
    'threshold': 3.0,                 
    'conv_radius_forecast': 6,        
    'conv_radius_observed': 3,        
    'time_window': 1,
    #'min_object_size': 450, #350  15
    'min_object_size_forecast': 400,       
    'min_object_size_observed': 30,    
    'interest_threshold': 0.6
}

# =============================================================================
# SYNTHETIC BENCHMARK PARAMETERS
# =============================================================================

SYNTHETIC_MODE_PARAMS = {

    'threshold': 2.0,
    'conv_radius_forecast': 5,
    'conv_radius_observed': 3,
    'time_window': 1,
    'min_object_size_forecast': 10,
    'min_object_size_observed': 2,
    'interest_threshold': 0.5

}


# Verificar y crear directorios
def setup_directories():
    """Crea los directorios necesarios"""
    if not os.path.exists(path_figures):
        os.makedirs(path_figures)
    if not os.path.exists(path_statistics):
        os.makedirs(path_statistics)
    print(f"Directorio de figuras: {path_figures}")
    print(f"Directorio de figuras: {path_statistics}")

# Llamar setup al importar
setup_directories()
