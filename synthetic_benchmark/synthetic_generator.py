#=============================================================================
# synthetic_generator.py
#=============================================================================
# Synthetic Benchmark Dataset Generator for MODE-py
# Generates Static Geometric and Dynamic Spatiotemporal Benchmark Suites.
#=============================================================================
import os
import numpy as np
import pandas as pd
import xarray as xr

#=============================================================================
# CONFIGURATION
#=============================================================================
# Dominio
LAT_MIN = 18.0
LAT_MAX = 24.0
LON_MIN = -86.0
LON_MAX = -74.0

# Resoluciones
WRF_RES_KM = 3.0
GPM_RES_KM = 10.0
WRF_RES_DEG = WRF_RES_KM / 111.0
GPM_RES_DEG = GPM_RES_KM / 111.0

# Directorios de salida
STATIC_OUTPUT_DIR = "synthetic_output/static"
DYNAMIC_OUTPUT_DIR = "synthetic_output/dynamic"
os.makedirs(STATIC_OUTPUT_DIR, exist_ok=True)
os.makedirs(DYNAMIC_OUTPUT_DIR, exist_ok=True)

# Configuración temporal
STATIC_TIMES = pd.date_range("2020-08-01 00:00", periods=1, freq="1h")
#DYNAMIC_TIMES = pd.date_range("2020-08-01 00:00", periods=24, freq="1H")

DYNAMIC_TIMES_TC = pd.date_range(
    "2020-08-01 00:00",
    periods=24,
    freq="1h"
)

DYNAMIC_TIMES_SPLITMERGE = pd.date_range(
    "2020-08-01 00:00",
    periods=5,
    freq="1h"
)

#=============================================================================
# GRID GENERATION & MESHGRIDS
#=============================================================================
wrf_lats = np.arange(LAT_MIN, LAT_MAX, WRF_RES_DEG)
wrf_lons = np.arange(LON_MIN, LON_MAX, WRF_RES_DEG)
gpm_lats = np.arange(LAT_MIN, LAT_MAX, GPM_RES_DEG)
gpm_lons = np.arange(LON_MIN, LON_MAX, GPM_RES_DEG)

X_fcst, Y_fcst = np.meshgrid(wrf_lons, wrf_lats)
X_obs, Y_obs = np.meshgrid(gpm_lons, gpm_lats)

#=============================================================================
# LOW-LEVEL GEOMETRY FUNCTIONS
#=============================================================================
def create_gaussian_ellipse(X, Y, center_x, center_y, sigma_x, sigma_y, angle_deg=0, amplitude=50):
    """Genera una elipse gaussiana rotada."""
    theta = np.deg2rad(angle_deg)
    Xc = X - center_x
    Yc = Y - center_y
    
    Xr = Xc * np.cos(theta) + Yc * np.sin(theta)
    Yr = -Xc * np.sin(theta) + Yc * np.cos(theta)
    
    field = amplitude * np.exp(-((Xr**2 / (2 * sigma_x**2)) + (Yr**2 / (2 * sigma_y**2))))
    return field

def create_gaussian_blob(X, Y, center, sigma, amplitude=50):
    """Genera un blob circular (caso especial de elipse)."""
    return create_gaussian_ellipse(X, Y, center[0], center[1], sigma, sigma, angle_deg=0, amplitude=amplitude)

def create_convective_line(X, Y, center, length_sigma, width_sigma, angle_deg, amplitude=50):
    """Genera una línea convectiva elongada."""
    return create_gaussian_ellipse(X, Y, center[0], center[1], length_sigma, width_sigma, angle_deg=angle_deg, amplitude=amplitude)

#=============================================================================
# STATIC BENCHMARKS
#=============================================================================
def displacement_case():
    obs = create_gaussian_ellipse(X_obs, Y_obs, center_x=-78.0, center_y=20.0, sigma_x=0.50, sigma_y=0.20, angle_deg=20)
    fcst = create_gaussian_ellipse(X_fcst, Y_fcst, center_x=-76.5, center_y=20.5, sigma_x=0.50, sigma_y=0.20, angle_deg=20)
    return obs, fcst

def orientation_case():
    obs = create_gaussian_ellipse(X_obs, Y_obs, center_x=-80.0, center_y=21.0, sigma_x=0.70, sigma_y=0.15, angle_deg=20)
    fcst = create_gaussian_ellipse(X_fcst, Y_fcst, center_x=-80.0, center_y=21.0, sigma_x=0.70, sigma_y=0.15, angle_deg=70)
    return obs, fcst

def fragmentation_case():
    obs = create_gaussian_ellipse(X_obs, Y_obs, center_x=-80.0, center_y=21.0, sigma_x=0.80, sigma_y=0.25, angle_deg=20)
    fcst = np.zeros_like(X_fcst)
    centers = [(-80.3, 20.9), (-81.5, 21.0), (-79.0, 21.1)]
    for cx, cy in centers:
        fcst += create_gaussian_ellipse(X_fcst, Y_fcst, center_x=cx, center_y=cy, sigma_x=0.15, sigma_y=0.05, angle_deg=20, amplitude=50)
    return obs, fcst

def convective_line_case():
    obs = create_convective_line(X_obs, Y_obs, center=(-84.5, 22.0), length_sigma=1.10, width_sigma=0.12, angle_deg=50)
    fcst = create_convective_line(X_fcst, Y_fcst, center=(-84.0, 21.5), length_sigma=1.20, width_sigma=0.08, angle_deg=45)
    return obs, fcst

def multicore_case():
    fcst = np.zeros_like(X_fcst)
    fcst += create_gaussian_blob(X_fcst, Y_fcst, center=(-80.5, 21.0), sigma=0.25)
    fcst += create_gaussian_ellipse(X_fcst, Y_fcst, center_x= -78.5, center_y= 22.0, sigma_x=0.5, sigma_y=0.20) 
    obs = create_gaussian_ellipse(X_obs, Y_obs, center_x=-80.0, center_y=21.0, sigma_x=0.80, sigma_y=0.25)
    return obs, fcst

#=============================================================================
# DYNAMIC BENCHMARKS
#=============================================================================

def tropical_cyclone_translation_case(t):

    """
    Tropical cyclone with recurving trajectory and spiral rainbands.
    """

    # ------------------------------------------------------------------
    # Observed cyclone trajectory
    # ------------------------------------------------------------------
    cx_obs = -84.0 + 0.15*t
    cy_obs = 19.5 + 0.015*(t**1.5)

    # ------------------------------------------------------------------
    # Forecast trajectory (slightly displaced)
    # ------------------------------------------------------------------
    cx_fcst = cx_obs + 0.15
    cy_fcst = cy_obs + 0.10

    # ------------------------------------------------------------------
    # Central core
    # ------------------------------------------------------------------
    obs = create_gaussian_blob(
        X_obs,
        Y_obs,
        center=(cx_obs, cy_obs),
        sigma=0.30,
        amplitude=70
    )

    fcst = create_gaussian_blob(
        X_fcst,
        Y_fcst,
        center=(cx_fcst, cy_fcst),
        sigma=0.32,
        amplitude=70
    )

    # ------------------------------------------------------------------
    # Spiral rainbands
    # ------------------------------------------------------------------
    rainband_angles = [120, 285]

    for angle in rainband_angles:

        dx = 0.8*np.cos(np.deg2rad(angle))
        dy = 0.8*np.sin(np.deg2rad(angle))

        obs += create_gaussian_ellipse(
            X_obs,
            Y_obs,
            center_x=cx_obs + dx,
            center_y=cy_obs + dy,
            sigma_x=0.40,
            sigma_y=0.12,
            angle_deg=angle + 35,
            amplitude=35
        )

        fcst += create_gaussian_ellipse(
            X_fcst,
            Y_fcst,
            center_x=cx_fcst + dx,
            center_y=cy_fcst + dy,
            sigma_x=0.35,
            sigma_y=0.12,
            angle_deg=angle + 35,
            amplitude=35
        )

    return obs, fcst
#=============================================================================

def splitting_convective_cell_case(t):

    """
    Single convective cell splitting into two cells.
    Duration: 5 timesteps.
    """

    stage = min(t,4)

    if stage == 0:

        obs = create_gaussian_blob(
            X_obs,Y_obs,
            center=(-80.0,21.0),
            sigma=0.45,
            amplitude=50
        )

        fcst = create_gaussian_blob(
            X_fcst,Y_fcst,
            center=(-80.0,21.0),
            sigma=0.45,
            amplitude=40
        )

    elif stage == 1:

        obs = create_gaussian_blob(
            X_obs,Y_obs,
            center=(-80.0,20.4),
            sigma=0.45,
            amplitude=40
        )

        fcst = create_gaussian_blob(
            X_fcst,Y_fcst,
            center=(-80.0,20.0),
            sigma=0.45,
            amplitude=30
        )


    elif stage == 2:

        obs = (
            create_gaussian_blob(
                X_obs,Y_obs,
                (-80.4,21.0),
                sigma=0.25,
                amplitude=60
            )
            +
            create_gaussian_blob(
                X_obs,Y_obs,
                (-79.6,21.0),
                sigma=0.25,
                amplitude=60
            )
        )

        fcst = (
            create_gaussian_blob(
                X_fcst,Y_fcst,
                (-80.4,21.0),
                sigma=0.2,
                amplitude=60
            )
            +
            create_gaussian_blob(
                X_fcst,Y_fcst,
                (-79.6,21.0),
                sigma=0.2,
                amplitude=60
            )
        )

    else:

        obs = (
            create_gaussian_blob(
                X_obs,Y_obs,
                (-80.8,21.0),
                sigma=0.22,
                amplitude=60
            )
            +
            create_gaussian_blob(
                X_obs,Y_obs,
                (-79.2,21.0),
                sigma=0.22,
                amplitude=60
            )
        )

        fcst = (
            create_gaussian_blob(
                X_fcst,Y_fcst,
                (-80.5,21.0),
                sigma=0.15,
                amplitude=60
            )
            +
            create_gaussian_blob(
                X_fcst,Y_fcst,
                (-79.5,21.0),
                sigma=0.15,
                amplitude=60
            )
        )

    return obs, fcst
#=============================================================================


def merging_convective_cells_case(t):

    """
    Two convective cells merging into one.
    Duration: 5 timesteps.
    """

    stage = min(t,4)

    if stage == 0:

        obs = (
            create_gaussian_blob(
                X_obs,Y_obs,
                (-80.8,21.0),
                sigma=0.22,
                amplitude=60
            )
            +
            create_gaussian_blob(
                X_obs,Y_obs,
                (-79.5,21.0),
                sigma=0.22,
                amplitude=60
            )
        )

        fcst = (
            create_gaussian_blob(
                X_fcst,Y_fcst,
                (-80.5,21.0),
                sigma=0.20,
                amplitude=60
            )
            +
            create_gaussian_blob(
                X_fcst,Y_fcst,
                (-79.5,21.0),
                sigma=0.20,
                amplitude=60
            )
        )

    elif stage == 1:

        obs = (
            create_gaussian_blob(
                X_obs,Y_obs,
                (-80.4,21.0),
                sigma=0.25,
                amplitude=60
            )
            +
            create_gaussian_blob(
                X_obs,Y_obs,
                (-79.6,21.5),
                sigma=0.25,
                amplitude=60
            )
        )

        fcst = (
            create_gaussian_blob(
                X_fcst,Y_fcst,
                (-80.4,21.0),
                sigma=0.25,
                amplitude=55
            )
            +
            create_gaussian_blob(
                X_fcst,Y_fcst,
                (-79.6,20.5),
                sigma=0.25,
                amplitude=55
            )
        )

    elif stage == 2:

        obs = create_gaussian_blob(
            X_obs,Y_obs,
            center=(-80.0,21.0),
            sigma=0.48,
            amplitude=60
        )

        fcst = create_gaussian_blob(
            X_fcst,Y_fcst,
            center=(-79.9,21.0),
            sigma=0.35,
            amplitude=55
        )


    else:

        obs = create_gaussian_blob(
            X_obs,Y_obs,
            center=(-80.0,21.0),
            sigma=0.45,
            amplitude=60
        )

        fcst = create_gaussian_blob(
            X_fcst,Y_fcst,
            center=(-79.9,21.0),
            sigma=0.48,
            amplitude=60
        )

    return obs, fcst


#=============================================================================
# CASE REGISTRY
#=============================================================================
STATIC_CASES = {
    "displacement": displacement_case,
    "orientation": orientation_case,
    "fragmentation": fragmentation_case,
    "convective_line": convective_line_case,
    "multicore": multicore_case
}

DYNAMIC_CASES = {
    "tropical_cyclone_translation":
        tropical_cyclone_translation_case,
    "splitting_convective_cell":
        splitting_convective_cell_case,
    "merging_convective_cells":
        merging_convective_cells_case
}

#=============================================================================
# NETCDF WRITER
#=============================================================================
def create_and_save_dataset(wrf_fields, gpm_fields, times, case_name, output_dir):
    """Construye y exporta los datasets de NetCDF."""
    wrf_ds = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), wrf_fields)},
        coords={"time": times, "lat": wrf_lats, "lon": wrf_lons}
    )
    gpm_ds = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), gpm_fields)},
        coords={"time": times, "lat": gpm_lats, "lon": gpm_lons}
    )

    wrf_file = os.path.join(output_dir, f"synthetic_wrf_{case_name}.nc")
    gpm_file = os.path.join(output_dir, f"synthetic_gpm_{case_name}.nc")

    wrf_ds.to_netcdf(wrf_file)
    gpm_ds.to_netcdf(gpm_file)
    
    print(f"  WRF saved: {wrf_file}")
    print(f"  GPM saved: {gpm_file}")

#=============================================================================
# MAIN
#=============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING STATIC BENCHMARK SUITE")
    print("="*70)
    
    for case_name, case_func in STATIC_CASES.items():
        print(f"\nGenerating static case: {case_name}")
        
        # Los casos estáticos solo necesitan 1 paso de tiempo
        obs_field, fcst_field = case_func()
        
        # Añadir ruido realista (corregido: shape > 0)
        obs_noise = np.random.gamma(shape=0.0, scale=0.0, size=obs_field.shape)
        fcst_noise = np.random.gamma(shape=0.0, scale=0.0, size=fcst_field.shape)
        
        obs_field = np.clip(obs_field + obs_noise, 0, None)
        fcst_field = np.clip(fcst_field + fcst_noise, 0, None)
        
        # Expandir dims para que coincida con (time, lat, lon)
        wrf_fields = np.expand_dims(fcst_field, axis=0)
        gpm_fields = np.expand_dims(obs_field, axis=0)
        
        create_and_save_dataset(wrf_fields, gpm_fields, STATIC_TIMES, case_name, STATIC_OUTPUT_DIR)

    print("\n" + "="*70)
    print("GENERATING DYNAMIC BENCHMARK SUITE")
    print("="*70)


    for case_name, case_func in DYNAMIC_CASES.items():

        print(f"\nGenerating dynamic case: {case_name}")

        # ----------------------------------------------------------
        # Time configuration
        # ----------------------------------------------------------

        if case_name == "tropical_cyclone_translation":

            case_times = DYNAMIC_TIMES_TC

        else:

            case_times = DYNAMIC_TIMES_SPLITMERGE

        wrf_fields = []
        gpm_fields = []

        for t in range(len(case_times)):

            obs_field, fcst_field = case_func(t)

            obs_noise = np.random.gamma(
                shape=0.0,
                scale=0.0,
                size=obs_field.shape
            )

            fcst_noise = np.random.gamma(
                shape=0.0,
                scale=0.0,
                size=fcst_field.shape
            )

            obs_field = np.clip(
                obs_field + obs_noise,
                0,
                None
            )

            fcst_field = np.clip(
                fcst_field + fcst_noise,
                0,
                None
            )

            gpm_fields.append(obs_field)
            wrf_fields.append(fcst_field)

        wrf_fields = np.array(wrf_fields)
        gpm_fields = np.array(gpm_fields)

        create_and_save_dataset(

            wrf_fields,

            gpm_fields,

            case_times,

            case_name,

            DYNAMIC_OUTPUT_DIR
        )

    print("\n" + "="*70)
    print("ALL SYNTHETIC BENCHMARK DATASETS GENERATED SUCCESSFULLY.")
    print("="*70 + "\n")
