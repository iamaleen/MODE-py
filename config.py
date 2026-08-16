#=============================================================================
# config.py
#=============================================================================
# Central Configuration Module for MODE-py
# This script defines all global paths, MODE algorithm parameters, and 
# synthetic benchmark settings. It acts as the single source of truth for 
# experiment configuration, allowing users to adjust thresholds, radii, 
# and interest weights without modifying the core algorithmic code.
#
# Note: All paths are relative to the project root or defined as absolute 
# paths for the specific computing environment.
#=============================================================================


import os

# Figure paths
# Statistics paths
# Model data paths
path_figures = "FIGURES/MODE/"
path_statistics = "FIGURES/MODE/STATISTICS/"
path_wrf = '/home/b1m/Aleen/B1TMET/GUYANA/SCRIPTS/DATASETS/WRF_OUTPUTS/SisPI_hires_2024-03-23_00/'

# Observational data pathways
path_gpm = '/home/b1m/Aleen/B1TMET/GUYANA/SCRIPTS/DATASETS/GPM/GPM_30MIN/SisPI_hires_2024-03-23_00/'

# Verification interval (1H, 3H, or every 6H)
accum_window_to_mode='1H'


##-----------------------------------------------------------------------------
# MODE-py PARAMETERS
##-----------------------------------------------------------------------------
# Distance threshold for considering the same temporal object
min_dist_same_object = 10.0

# Distance threshold for considering the same spatial object
spatial_threshold = 180.0                


MODE_PARAMS = {
    'threshold': 3.0,                 
    'conv_radius_forecast': 12,        
    'conv_radius_observed': 3,        
    'time_window': 1,
    'min_object_size_forecast': 400,       
    'min_object_size_observed': 30,    
    'interest_threshold': 0.6
}


##-----------------------------------------------------------------------------
# INTEREST FUNCTION WEIGHTS
##-----------------------------------------------------------------------------

INTEREST_WEIGHTS = {
    'distance': 0.25,
    'area_ratio': 0.20,
    'overlap': 0.25,
    'orientation': 0.10,
    'temporal': 0.20
}


##-----------------------------------------------------------------------------
# SYNTHETIC BENCHMARK PARAMETERS
##-----------------------------------------------------------------------------

SYNTHETIC_MODE_PARAMS = {

    'threshold': 2.0,
    'conv_radius_forecast': 5,
    'conv_radius_observed': 3,
    'time_window': 1,
    'min_object_size_forecast': 10,
    'min_object_size_observed': 2,
    'interest_threshold': 0.5

}

# Check and create directories
def setup_directories():
    """Create necesary directories"""
    if not os.path.exists(path_figures):
        os.makedirs(path_figures)
    if not os.path.exists(path_statistics):
        os.makedirs(path_statistics)
    print(f"Figure paths: {path_figures}")
    print(f"Statistics paths: {path_statistics}")

setup_directories()
