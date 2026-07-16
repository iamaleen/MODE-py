#=============================================================================
# synthetic_visualization.py
#=============================================================================
#
# Visualization Utilities for Synthetic MODE Benchmark Suite
#
# Generates:
# 1. Static Benchmark Summary
# 2. Dynamic Benchmark Summary
# 3. MODE Processing Pipeline Diagnostics (NEW)
#
#=============================================================================

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  


# CORRECCIÓN DE RUTA: Permite importar mode_verifier desde el directorio padre
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from mode_verifier import MODE3DVerifier
from field_visualization import cmap

#=============================================================================
# DIRECTORIES
#=============================================================================

STATIC_DIR = os.path.join(CURRENT_DIR, "synthetic_output/static")
DYNAMIC_DIR = os.path.join(CURRENT_DIR, "synthetic_output/dynamic")

FIG_DIR = os.path.join(CURRENT_DIR, "synthetic_figures")
os.makedirs(FIG_DIR, exist_ok=True)

#=============================================================================
# CASE DEFINITIONS
#=============================================================================

STATIC_CASES = [
    "displacement",
    "orientation",
    "fragmentation",
    "convective_line",
    "multicore"
]

DYNAMIC_CASES = [
    "tropical_cyclone_translation",
    "splitting_convective_cell",
    "merging_convective_cells"
]

#=============================================================================
# HELPERS
#=============================================================================

def load_case(case_name, case_type):

    if case_type == "static":
        wrf_file = os.path.join(STATIC_DIR, f"synthetic_wrf_{case_name}.nc")
        gpm_file = os.path.join(STATIC_DIR, f"synthetic_gpm_{case_name}.nc")
    else:
        wrf_file = os.path.join(DYNAMIC_DIR, f"synthetic_wrf_{case_name}.nc")
        gpm_file = os.path.join(DYNAMIC_DIR, f"synthetic_gpm_{case_name}.nc")

    wrf = xr.open_dataset(wrf_file)
    gpm = xr.open_dataset(gpm_file)

    return wrf, gpm


#=============================================================================
# STATIC SUMMARY 
#=============================================================================

def plot_static_benchmark_summary():
    print("\nCreating static benchmark summary...")

    fig, axes = plt.subplots(2, 3, figsize=(16,10))
    axes = axes.flatten()

    fig.suptitle(f"Static Benchmark Suite", fontsize=14)

    for idx, case_name in enumerate(STATIC_CASES):
        wrf, gpm = load_case(case_name, "static")
        ax = axes[idx]

        fcst = wrf["precipitation"].isel(time=0)
        obs = gpm["precipitation"].isel(time=0)

        # Forecast como relleno
        cf = ax.contourf(fcst.lon, fcst.lat, fcst, levels=15, cmap="turbo") #cmap="turbo"

        # Observación como contorno
        ax.contour(obs.lon, obs.lat, obs, levels=[10], colors="black", linewidths=2)

        ax.set_title(case_name.replace("_"," ").title(), fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    axes[-1].axis("off")
    plt.tight_layout()

    outfile = os.path.join(FIG_DIR, "static_benchmark_suite.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()



#=============================================================================
# DYNAMIC SUMMARY 
#=============================================================================

def plot_dynamic_benchmark_summary():
    print("\nCreating dynamic benchmark summary...")

    fig, axes = plt.subplots(1, 3, figsize=(18,6))

    # Tropical Cyclone
    wrf, gpm = load_case("tropical_cyclone_translation", "dynamic")
    ax = axes[0]
    selected_times = [0, 6, 12, 18, 23]
    for t in selected_times:
        field = wrf["precipitation"].isel(time=t)
        ax.contour(field.lon, field.lat, field, levels=[10], linewidths=1.5)
    ax.set_title("Tropical Cyclone Translation", fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Splitting Cell
    wrf, gpm = load_case("splitting_convective_cell", "dynamic")
    ax = axes[1]
    selected_times = [0, 2, 4]
    for t in selected_times:
        field = wrf["precipitation"].isel(time=t)
        ax.contour(field.lon, field.lat, field, levels=[10], linewidths=1.5)
    ax.set_title("Splitting Convective Cell", fontsize=11)
    ax.set_xlabel("Longitude")

    # Merging Cells
    wrf, gpm = load_case("merging_convective_cells", "dynamic")
    ax = axes[2]
    selected_times = [0, 2, 4]
    for t in selected_times:
        field = wrf["precipitation"].isel(time=t)
        ax.contour(field.lon, field.lat, field, levels=[10], linewidths=1.5)
    ax.set_title("Merging Convective Cells", fontsize=11)
    ax.set_xlabel("Longitude")

    plt.tight_layout()
    outfile = os.path.join(FIG_DIR, "dynamic_benchmark_suite.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


#=============================================================================
# DYNAMIC EVOLUTION FIGURES 
#=============================================================================

def plot_splitting_evolution():
    print("\nCreating splitting evolution...")
    wrf, gpm = load_case("splitting_convective_cell", "dynamic")

    fig, axes = plt.subplots(1, 5, figsize=(18,4))
    times = [0,1,2,3,4]

    for ax, t in zip(axes, times):
        field = wrf["precipitation"].isel(time=t)
        ax.contourf(field.lon, field.lat, field, levels=15, cmap="turbo") #cmap="turbo"
        ax.set_title(f"T{t}")

    plt.tight_layout()
    outfile = os.path.join(FIG_DIR, "splitting_convective_cell.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")
    

def plot_merging_evolution():
    print("\nCreating merging evolution...")
    wrf, gpm = load_case("merging_convective_cells", "dynamic")

    fig, axes = plt.subplots(1, 5, figsize=(18,4))
    times = [0,1,2,3,4]

    for ax, t in zip(axes, times):
        field = wrf["precipitation"].isel(time=t)
        ax.contourf(field.lon, field.lat, field, levels=15, cmap="turbo") #cmap="turbo"
        ax.set_title(f"T{t}")

    plt.tight_layout()
    outfile = os.path.join(FIG_DIR, "merging_convective_cells.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")

def plot_tropical_cyclone_evolution():
    print("\nCreating merging tropical cyclon evolution...")
    wrf, gpm = load_case("tropical_cyclone_translation", "dynamic")

    fig, axes = plt.subplots(1, 5, figsize=(18,4))
    times = [0, 6, 12, 18, 23]

    for ax, t in zip(axes, times):
        field = wrf["precipitation"].isel(time=t)
        ax.contourf(field.lon, field.lat, field, levels=15, cmap="turbo") #cmap="turbo"
        ax.set_title(f"T{t}")

    plt.tight_layout()
    outfile = os.path.join(FIG_DIR, "tropical_cyclone_translation.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


#=============================================================================
# MODE PIPELINE DIAGNOSTICS 
#=============================================================================

def plot_thresholding_step(verifier, time_idx=0, output_dir=FIG_DIR):
    """Muestra: Original -> Suavizado -> Binario (Umbral)"""
    print(f"\nPlotting Thresholding Step (t={time_idx})...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"MODE Pipeline: Thresholding (t={time_idx})", fontsize=14)
    
    t_slice = time_idx
    
    # Forecast Row
    orig_fcst = verifier.forecast.isel(time=t_slice).values
    smooth_fcst = verifier.smoothed_fcst.isel(time=t_slice).values
    bin_fcst = verifier.binary_fcst.isel(time=t_slice).values.astype(int)
    lons_fcst, lats_fcst = verifier.forecast.lon.values, verifier.forecast.lat.values
    
    axes[0,0].contourf(lons_fcst, lats_fcst, orig_fcst, levels=15, cmap=cmap) #cmap="turbo"
    axes[0,0].set_title("Forecast Original")
    
    axes[0,1].contourf(lons_fcst, lats_fcst, smooth_fcst, levels=15, cmap="Grays") #cmap="turbo"
    axes[0,1].set_title(f"Forecast Smoothed (R={verifier.conv_radius_forecast})")
    
    axes[0,2].contourf(lons_fcst, lats_fcst, bin_fcst, levels=[-0.5, 0.5, 1.5], colors=['white', 'red'])
    axes[0,2].set_title(f"Forecast Binary (T={verifier.threshold})")
    
    # Observed Row
    orig_obs = verifier.observed.isel(time=t_slice).values
    smooth_obs = verifier.smoothed_obs.isel(time=t_slice).values
    bin_obs = verifier.binary_obs.isel(time=t_slice).values.astype(int)
    lons_obs, lats_obs = verifier.observed.lon.values, verifier.observed.lat.values
    
    axes[1,0].contourf(lons_obs, lats_obs, orig_obs, levels=15, cmap=cmap) #cmap="turbo"
    axes[1,0].set_title("Observed Original")
    
    axes[1,1].contourf(lons_obs, lats_obs, smooth_obs, levels=15, cmap="Grays") #cmap="turbo"
    axes[1,1].set_title(f"Observed Smoothed (R={verifier.conv_radius_observed})")
    
    axes[1,2].contourf(lons_obs, lats_obs, bin_obs, levels=[-0.5, 0.5, 1.5], colors=['white', 'blue'])
    axes[1,2].set_title(f"Observed Binary (T={verifier.threshold})")
    
    for ax in axes.flatten():
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        
    plt.tight_layout()
    outfile = os.path.join(output_dir, f"pipeline_thresholding_t{time_idx}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
 


def plot_object_identification_step(verifier, time_idx=0, output_dir=FIG_DIR, color_scheme='custom'):
    """Muestra los objetos identificados como manchas continuas de colores, con centroides e IDs."""
    print(f"\nPlotting Object Identification Step (t={time_idx})...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"MODE Pipeline: Object Identification (t={time_idx})", fontsize=14)
    
    target_time = pd.Timestamp(verifier.forecast.time.isel(time=time_idx).values)
    
    # =========================================================================
    # FUNCIÓN AUXILIAR PARA ELEGIR LA PALETA DE COLORES
    # =========================================================================
    def get_colors(n, scheme):
        if n == 0:
            return []
        if scheme == 'custom':
            # Paleta personalizada de alto contraste (ColorBrewer Set1 + extras)
            hex_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
                          '#ffff33', '#a65628', '#f781bf', '#999999', '#17becf']
            return [hex_colors[i % len(hex_colors)] for i in range(n)]
        elif scheme == 'tab20':
            return [plt.cm.tab20(i / max(1, n-1)) for i in range(n)]
        elif scheme == 'Set1':
            return [plt.cm.Set1(i / max(1, min(n, 9)-1)) for i in range(n)]
        elif scheme == 'Set2':
            return [plt.cm.Set2(i / max(1, min(n, 8)-1)) for i in range(n)]
        elif scheme == 'Paired':
            return [plt.cm.Paired(i / max(1, min(n, 12)-1)) for i in range(n)]
        else:
            # Fallback a cualquier colormap de matplotlib (ej. 'hsv', 'gist_rainbow')
            cmap = plt.cm.get_cmap(scheme)
            return [cmap(i / max(1, n-1)) for i in range(n)]

    # =========================================================================
    # FORECAST
    # =========================================================================
    axes[0].set_title("Forecast Objects")
    
    fcst_objs_at_t = [obj for obj in verifier.forecast_objects if target_time in obj['time_points']]
    n_fcst = len(fcst_objs_at_t)
    
    shape_fcst = verifier.binary_fcst.isel(time=time_idx).shape
    label_grid_fcst = np.zeros(shape_fcst)
    
    colors_fcst = get_colors(n_fcst, color_scheme)
    
    for idx, obj in enumerate(fcst_objs_at_t):
        obj_2d = next((o for o in obj['objects_2d'] if o['time'] == target_time), None)
        if obj_2d is not None and 'coords_pixel' in obj_2d:
            rows = obj_2d['coords_pixel'][:, 0]
            cols = obj_2d['coords_pixel'][:, 1]
            label_grid_fcst[rows, cols] = idx + 1
    
    if n_fcst > 0:
        cmap_fcst = mcolors.ListedColormap(['white'] + colors_fcst)
        bounds_fcst = np.arange(-0.5, n_fcst + 1.5, 1)
        norm_fcst = mcolors.BoundaryNorm(bounds_fcst, cmap_fcst.N)
        
        axes[0].contourf(verifier.forecast.lon.values, verifier.forecast.lat.values, 
                         label_grid_fcst, levels=bounds_fcst, cmap=cmap_fcst, norm=norm_fcst)

    else:
        bin_fcst = verifier.binary_fcst.isel(time=time_idx).values.astype(int)
        axes[0].contourf(verifier.forecast.lon.values, verifier.forecast.lat.values, bin_fcst, 
                         levels=[-0.5, 0.5, 1.5], colors=['white', 'lightgray'])
    
    for idx, obj in enumerate(fcst_objs_at_t):
        cx, cy = obj['centroid_mean_geo']
        axes[0].plot(cy, cx, 'x', color=colors_fcst[idx], markersize=10, markeredgewidth=2)
        axes[0].text(cy, cx, f" ID:{obj['id']}", color='black', fontsize=9, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor=colors_fcst[idx], linewidth=1.0))

    # =========================================================================
    # OBSERVED
    # =========================================================================
    axes[1].set_title("Observed Objects")
    
    obs_objs_at_t = [obj for obj in verifier.observed_objects if target_time in obj['time_points']]
    n_obs = len(obs_objs_at_t)
    
    shape_obs = verifier.binary_obs.isel(time=time_idx).shape
    label_grid_obs = np.zeros(shape_obs)
    
    colors_obs = get_colors(n_obs, color_scheme)
    
    for idx, obj in enumerate(obs_objs_at_t):
        obj_2d = next((o for o in obj['objects_2d'] if o['time'] == target_time), None)
        if obj_2d is not None and 'coords_pixel' in obj_2d:
            rows = obj_2d['coords_pixel'][:, 0]
            cols = obj_2d['coords_pixel'][:, 1]
            label_grid_obs[rows, cols] = idx + 1
    
    if n_obs > 0:
        cmap_obs = mcolors.ListedColormap(['white'] + colors_obs)
        bounds_obs = np.arange(-0.5, n_obs + 1.5, 1)
        norm_obs = mcolors.BoundaryNorm(bounds_obs, cmap_obs.N)
        
        axes[1].contourf(verifier.observed.lon.values, verifier.observed.lat.values, 
                         label_grid_obs, levels=bounds_obs, cmap=cmap_obs, norm=norm_obs)

        
    else:
        bin_obs = verifier.binary_obs.isel(time=time_idx).values.astype(int)
        axes[1].contourf(verifier.observed.lon.values, verifier.observed.lat.values, bin_obs, 
                         levels=[-0.5, 0.5, 1.5], colors=['white', 'lightgray'])
    
    for idx, obj in enumerate(obs_objs_at_t):
        cx, cy = obj['centroid_mean_geo']
        axes[1].plot(cy, cx, 'x', color=colors_obs[idx], markersize=10, markeredgewidth=2)
        axes[1].text(cy, cx, f" ID:{obj['id']}", color='black', fontsize=9, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor=colors_obs[idx], linewidth=1.5))
        
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
    plt.tight_layout()
    outfile = os.path.join(output_dir, f"pipeline_identification_t{time_idx}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()



def plot_cross_identification_step(verifier, time_idx=0, output_dir=FIG_DIR, color_scheme='custom'):
    """
    Muestra cruce de identificación:
    Panel Izq: Contorno OBS + Relleno FCST
    Panel Der: Contorno FCST + Relleno OBS
    """
    print(f"\nPlotting Cross Identification Step (t={time_idx})...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"MODE Pipeline: Object Identification (t={time_idx})", fontsize=14)
    
    target_time = pd.Timestamp(verifier.forecast.time.isel(time=time_idx).values)
    
    # =========================================================================
    # FUNCIÓN AUXILIAR PARA PALETA DE COLORES (IDÉNTICA A TU VERSIÓN)
    # =========================================================================
    def get_colors(n, scheme):
        if n == 0: return []
        if scheme == 'custom':
            hex_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
                          '#ffff33', '#a65628', '#f781bf', '#999999', '#17becf']
            return [hex_colors[i % len(hex_colors)] for i in range(n)]
        elif scheme == 'tab20':
            return [plt.cm.tab20(i / max(1, n-1)) for i in range(n)]
        elif scheme == 'Set1':
            return [plt.cm.Set1(i / max(1, min(n, 9)-1)) for i in range(n)]
        elif scheme == 'Set2':
            return [plt.cm.Set2(i / max(1, min(n, 8)-1)) for i in range(n)]
        elif scheme == 'Paired':
            return [plt.cm.Paired(i / max(1, min(n, 12)-1)) for i in range(n)]
        else:
            cmap = plt.cm.get_cmap(scheme)
            return [cmap(i / max(1, n-1)) for i in range(n)]

    # Helper para dibujar RELLENOS CONTINUOS (tu lógica original)
    def draw_fills(ax, objects, lons, lats, shape, colors, n_objs):
        label_grid = np.zeros(shape)
        for idx, obj in enumerate(objects):
            obj_2d = next((o for o in obj['objects_2d'] if o['time'] == target_time), None)
            if obj_2d and 'coords_pixel' in obj_2d:
                rows, cols = obj_2d['coords_pixel'][:, 0], obj_2d['coords_pixel'][:, 1]
                label_grid[rows, cols] = idx + 1
        if n_objs > 0:
            cmap = mcolors.ListedColormap(['white'] + colors)
            bounds = np.arange(-0.5, n_objs + 1.5, 1)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            ax.contourf(lons, lats, label_grid, levels=bounds, cmap=cmap, norm=norm)

    # Helper para dibujar BORDES CONTINUOS (máscara binaria + contour)
    def draw_contours(ax, objects, lons, lats, shape):
        binary_grid = np.zeros(shape)
        for obj in objects:
            obj_2d = next((o for o in obj['objects_2d'] if o['time'] == target_time), None)
            if obj_2d and 'coords_pixel' in obj_2d:
                rows, cols = obj_2d['coords_pixel'][:, 0], obj_2d['coords_pixel'][:, 1]
                binary_grid[rows, cols] = 1
        if np.any(binary_grid):
            ax.contour(lons, lats, binary_grid, levels=[0.5, 1.5], 
                       colors='black', linewidths=1.5, alpha=0.9)

    # Helper para dibujar centroides e IDs
    def draw_labels(ax, objects, colors):
        for idx, obj in enumerate(objects):
            cx, cy = obj['centroid_mean_geo']
            color = colors[idx % len(colors)] if colors else 'black'
            ax.plot(cy, cx, 'x', color=color, markersize=10, markeredgewidth=2)
            ax.text(cy, cx, f" ID:{obj['id']}", color='black', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, linewidth=1.5))

    # =========================================================================
    # PREPARACIÓN DE DATOS
    # =========================================================================
    fcst_objs = [obj for obj in verifier.forecast_objects if target_time in obj['time_points']]
    obs_objs  = [obj for obj in verifier.observed_objects if target_time in obj['time_points']]
    n_fcst = len(fcst_objs)
    n_obs  = len(obs_objs)
    
    colors_fcst = get_colors(n_fcst, color_scheme)
    colors_obs  = get_colors(n_obs, color_scheme)

    # =========================================================================
    # PANEL 1: Contorno OBS + Relleno FCST
    # =========================================================================
    axes[0].set_title("Forecast Objects with Observation Contour", fontsize=11)
    lons_fcst, lats_fcst = verifier.forecast.lon.values, verifier.forecast.lat.values
    
    # 1. Relleno de PRONOSTICO (colores continuos)
    draw_fills(axes[0], fcst_objs, lons_fcst, lats_fcst, verifier.binary_fcst.isel(time=time_idx).shape, colors_fcst, n_fcst)
    
    # 2. Contorno de OBSERVACION (línea negra continua)
    draw_contours(axes[0], obs_objs, verifier.observed.lon.values, verifier.observed.lat.values, 
                  verifier.binary_obs.isel(time=time_idx).shape)
                  
    # 3. Etiquetas (centroide + ID) para ambos
    #draw_labels(axes[0], fcst_objs, colors_fcst)
    #draw_labels(axes[0], obs_objs, colors_obs)

    # =========================================================================
    # PANEL 2: Contorno FCST + Relleno OBS
    # =========================================================================
    axes[1].set_title("Observation Objects with Forecast Contour", fontsize=11)
    lons_obs, lats_obs = verifier.observed.lon.values, verifier.observed.lat.values
    
    # 1. Relleno de OBSERVACION (colores continuos)
    draw_fills(axes[1], obs_objs, lons_obs, lats_obs, verifier.binary_obs.isel(time=time_idx).shape, colors_obs, n_obs)
    
    # 2. Contorno de PRONOSTICO (línea negra continua)
    draw_contours(axes[1], fcst_objs, lons_fcst, lats_fcst, verifier.binary_fcst.isel(time=time_idx).shape)
    
    # 3. Etiquetas
    #draw_labels(axes[1], obs_objs, colors_obs)
    #draw_labels(axes[1], fcst_objs, colors_fcst)

    # =========================================================================
    # AJUSTES FINALES
    # =========================================================================
    for ax in axes:
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        #ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    outfile = os.path.join(output_dir, f"pipeline_cross_identification_t{time_idx}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()



def plot_filtering_step(verifier, time_idx=0, output_dir=FIG_DIR):
    """Muestra los objetos que sobrevivieron al filtrado por tamaño mínimo."""
    print(f"\nPlotting Filtering Step (t={time_idx})...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"MODE Pipeline: Filtered Objects (t={time_idx})", fontsize=14)
    
    target_time = pd.Timestamp(verifier.forecast.time.isel(time=time_idx).values)
    
    # Forecast
    bin_fcst = verifier.binary_fcst.isel(time=time_idx).values.astype(int)
    axes[0].contourf(verifier.forecast.lon.values, verifier.forecast.lat.values, bin_fcst, levels=[-0.5, 0.5, 1.5], colors=['white', 'lightgray'], alpha=0.5)
    axes[0].set_title(f"Forecast Filtered (min_size={verifier.min_object_size_forecast})")
    
    fcst_objs_at_t = [obj for obj in verifier.forecast_objects if target_time in obj['time_points']]
    for obj in fcst_objs_at_t:
        cx, cy = obj['centroid_mean_geo']
        axes[0].plot(cy, cx, 'x', color='red', markersize=8)
        axes[0].text(cy, cx, f" A={obj['area_mean']:.0f}", color='red', fontsize=8, fontweight='bold')

    # Observed
    bin_obs = verifier.binary_obs.isel(time=time_idx).values.astype(int)
    axes[1].contourf(verifier.observed.lon.values, verifier.observed.lat.values, bin_obs, levels=[-0.5, 0.5, 1.5], colors=['white', 'lightgray'], alpha=0.5)
    axes[1].set_title(f"Observed Filtered (min_size={verifier.min_object_size_observed})")
    
    obs_objs_at_t = [obj for obj in verifier.observed_objects if target_time in obj['time_points']]
    for obj in obs_objs_at_t:
        cx, cy = obj['centroid_mean_geo']
        axes[1].plot(cy, cx, 'x', color='blue', markersize=8)
        axes[1].text(cy, cx, f" A={obj['area_mean']:.0f}", color='blue', fontsize=8, fontweight='bold')
        
    for ax in axes:
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        
    plt.tight_layout()
    outfile = os.path.join(output_dir, f"pipeline_filtering_t{time_idx}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()



def plot_matching_step(verifier, time_idx=0, output_dir=FIG_DIR):
    """Muestra el emparejamiento espacial y la matriz de interés."""
    print(f"\nPlotting Matching Step (t={time_idx})...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"MODE Pipeline: Object Matching (t={time_idx})", fontsize=14)
    
    target_time = pd.Timestamp(verifier.forecast.time.isel(time=time_idx).values)
    lons_fcst, lats_fcst = verifier.forecast.lon.values, verifier.forecast.lat.values
    
    fcst_at_t = [obj for obj in verifier.forecast_objects if target_time in obj['time_points']]
    obs_at_t = [obj for obj in verifier.observed_objects if target_time in obj['time_points']]
    
    matches_at_t = []
    for match in verifier.matches:
        fcst_obj = next((o for o in fcst_at_t if o['id'] == match['forecast_id']), None)
        obs_obj = next((o for o in obs_at_t if o['id'] == match['observed_id']), None)
        if fcst_obj and obs_obj:
            matches_at_t.append({'forecast': fcst_obj, 'observed': obs_obj, 'interest': match['interest']})
            
    # =========================================================================
    # Panel 1: Spatial Matching (Campo FCST + Contorno OBS emparejado)
    # =========================================================================
    ax1 = axes[0]
    orig = verifier.forecast.isel(time=time_idx).values
    ax1.contourf(lons_fcst, lats_fcst, orig, levels=15, cmap=cmap) 
    ax1.set_title("Matched Pairs: Forecast Field with Observed Contours")
    
    match_colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFA500', '#800080']
    
    for i, match in enumerate(matches_at_t):
        color = match_colors[i % len(match_colors)]
        fcst_c = match['forecast']['centroid_mean_geo']  # (lat, lon)
        obs_c = match['observed']['centroid_mean_geo']
        
        # 1. Dibujar el contorno del objeto OBSERVADO emparejado sobre el campo pronosticado
        obs_obj = match['observed']
        obj_2d_obs = next((o for o in obs_obj['objects_2d'] if o['time'] == target_time), None)
        if obj_2d_obs is not None and 'coords_geo' in obj_2d_obs:
            coords_obs = obj_2d_obs['coords_geo']
            lons_obs = coords_obs[:, 1]
            lats_obs = coords_obs[:, 0]
            
            # Contorno del objeto observado
            ax1.plot(lons_obs, lats_obs, color=color, linewidth=2.0, alpha=0.1)
            # Relleno muy sutil para que se note el área observada sin tapar el campo de fondo
            #ax1.fill(lons_obs, lats_obs, color=color, alpha=0.9)
            
        # 2. Línea que une los centroides (vector de desplazamiento)
        ax1.plot([fcst_c[1], obs_c[1]], [fcst_c[0], obs_c[0]], '-', color=color, linewidth=2, alpha=0.7)
        
        # 3. Marcadores de centroides
        ax1.plot(fcst_c[1], fcst_c[0], 'x', color=color, markersize=10, markeredgewidth=2) # Forecast
        ax1.plot(obs_c[1], obs_c[0], 'o', color=color, markersize=8, markeredgewidth=2)    # Observed
        
        # 4. Etiqueta del valor de interés en el punto medio
        mid_lon, mid_lat = (fcst_c[1] + obs_c[1]) / 2, (fcst_c[0] + obs_c[0]) / 2
        ax1.text(mid_lon, mid_lat, f"{match['interest']:.2f}", ha='center', va='center', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=color, linewidth=1.5))
                 
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    
    # =========================================================================
    # Panel 2: Interest Matrix Heatmap
    # =========================================================================
    ax2 = axes[1]
    if verifier.interest_matrix is not None and verifier.interest_matrix.size > 0:
        fcst_indices = [i for i, obj in enumerate(verifier.forecast_objects) if target_time in obj['time_points']]
        obs_indices = [j for j, obj in enumerate(verifier.observed_objects) if target_time in obj['time_points']]
        
        if fcst_indices and obs_indices:
            sub_matrix = verifier.interest_matrix[np.ix_(fcst_indices, obs_indices)]
            im = ax2.imshow(sub_matrix, cmap='turbo', aspect='auto', vmin=0, vmax=1)  #YlOrRd
            ax2.set_title("Interest Matrix (Fuzzy Logic)")
            ax2.set_xlabel("Observation Objects")
            ax2.set_ylabel("Forecast Objects")
            ax2.set_xticks(np.arange(sub_matrix.shape[1]))
            ax2.set_yticks(np.arange(sub_matrix.shape[0]))
            ax2.set_xticklabels([f"O{j}" for j in obs_indices], rotation=45)
            ax2.set_yticklabels([f"F{i}" for i in fcst_indices])
            
            for i in range(sub_matrix.shape[0]):
                for j in range(sub_matrix.shape[1]):
                    val = sub_matrix[i, j]
                    text_color = "white" if val > 0.6 else "black"
                    ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        else:
            ax2.text(0.5, 0.5, "No objects at this time", ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Interest Matrix")
    else:
        ax2.text(0.5, 0.5, "No interest matrix available", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Interest Matrix")
        
    plt.tight_layout()
    outfile = os.path.join(output_dir, f"pipeline_matching_t{time_idx}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()



def generate_mode_pipeline_report(verifier, time_idx=0, suite="static", case_name="displacement"):
    """Ejecuta TODAS las funciones de diagnóstico del pipeline en una carpeta específica."""
    target_dir = os.path.join(FIG_DIR, suite, case_name)
    os.makedirs(target_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"GENERATING COMPLETE MODE PIPELINE REPORT: {suite.upper()} / {case_name.upper()}")
    print(f"Output directory: {target_dir}")
    print("="*70)
    
    # Pasos internos del algoritmo
    plot_thresholding_step(verifier, time_idx, output_dir=target_dir)
    plot_object_identification_step(verifier, time_idx, output_dir=target_dir)
    plot_filtering_step(verifier, time_idx, output_dir=target_dir)
    plot_matching_step(verifier, time_idx, output_dir=target_dir)  

   
    plot_cross_identification_step(verifier, time_idx, output_dir=target_dir)
    
#=============================================================================
# MAIN
#=============================================================================

if __name__ == "__main__":

    # 1. Generar resúmenes de benchmarks (NO requiere el verificador)
    plot_static_benchmark_summary()
    plot_dynamic_benchmark_summary()

    plot_tropical_cyclone_evolution()
    plot_splitting_evolution()
    plot_merging_evolution()

    # 2. Generar diagnóstico del Pipeline de MODE (REQUIERE un objeto verifier)
    print("\nLoading verifier for pipeline diagnostics...")
    
    # Configuración del caso a diagnosticar 
    CASE_TO_TEST = "tropical_cyclone_translation"
    SUITE_TO_TEST = "dynamic"
    
    # Ruta al archivo .pkl guardado por run_synthetic_benchmark.py
    pkl_path = os.path.join(
        CURRENT_DIR, 
        "synthetic_benchmark_results", 
        SUITE_TO_TEST, 
        CASE_TO_TEST, 
        f"benchmark_{CASE_TO_TEST}.pkl" 
    )
    
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            verifier = pickle.load(f)
        print(f"Successfully loaded verifier from: {pkl_path}")
        
        # Generar las 4 figuras del pipeline en la carpeta específica: synthetic_figures/static/multicore/
        generate_mode_pipeline_report(
            verifier, 
            time_idx=0, 
            suite=SUITE_TO_TEST, 
            case_name=CASE_TO_TEST
        )
    else:
        print(f"Warning: Verifier pickle not found at {pkl_path}.")
        print("Please run the benchmark first to generate the .pkl file.")

    print("\nAll benchmark figures generated successfully.\n")
