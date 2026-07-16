##-----------------------------------------------------------------------------
## Visualización de datos
##-----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import os
import config


# Paleta de colores para precipitación
precip_colors = [
    '#ffffff', '#a0f0a0', '#50c050', '#00a000', '#00e000',
    '#50ff00', '#f0ff00', '#ffb000', '#ff7000', '#ff3000',
    '#ff0000', '#c00000', '#a00000', '#a000a0', '#800080',
    '#600060', '#400040'
]

# Crear el mapa de colores
cmap = mcolors.ListedColormap(precip_colors)

def plot_prec2(ds_real, ds_for, time_index, save_path=None, window=config.accum_window_to_mode):
    """
    Grafica comparación de precipitación entre GPM (observado) y WRF (pronóstico)
    """
    # Seleccionar el tiempo a graficar
    ds_pr_real = ds_real.isel(time=time_index)
    ds_pr_forecasted = ds_for.isel(time=time_index)
    
    # Obtener la fecha para el título
    time_str = pd.Timestamp(ds_pr_real.time.values).strftime('%Y-%m-%d %H:%M')
    
    # Crear la figura con diseño controlado
    #fig = plt.figure(figsize=(16, 12))
    
    fig = plt.figure(figsize=(16, 6))

    # Definir la disposición de los subplots y la barra de color
    grid = plt.GridSpec(2, 2, height_ratios=[15, 1], width_ratios=[1, 1], hspace=0.3)
    
    # Primer subplot (real - GPM)
    ax1 = fig.add_subplot(grid[0, 0], projection=ccrs.PlateCarree())
    # Segundo subplot (pronóstico - WRF)
    ax2 = fig.add_subplot(grid[0, 1], projection=ccrs.PlateCarree())
    # Espacio para la barra de color
    cbar_ax = fig.add_subplot(grid[1, :])

    # Ajustar niveles de contorno según la ventana
    if window == '1H':
        levels = np.arange(0, 50, 2)  # Niveles para 1 hora
    elif window == '3H':
        levels = np.arange(0, 100, 2)  # Niveles para 3 horas
    else:
        levels = np.arange(0, 200, 2)  # Niveles para 6 horas o más
    
    # Configuración común para ambos subplots
    for ax in [ax1, ax2]:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
        ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.2)
        
        gl = ax.gridlines(crs=ccrs.PlateCarree(),
                        draw_labels=True,
                        linewidth=0.05, 
                        color='gray', 
                        alpha=0.5,
                        linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

    # Graficar mapa real (GPM)
    contour_real = ax1.contourf(ds_pr_real['lon'], 
                              ds_pr_real['lat'], 
                              ds_pr_real,
                              transform=ccrs.PlateCarree(),
                              #levels=np.arange(0, 50, 2),
                              levels=levels,  
                              cmap=cmap,
                              extend='both')
    ax1.set_title(f'Precipitation GPM')

    # Graficar mapa pronosticado (WRF)
    contour_for = ax2.contourf(ds_pr_forecasted['lon'], 
                             ds_pr_forecasted['lat'], 
                             ds_pr_forecasted,
                             transform=ccrs.PlateCarree(),
                             levels=levels,
                             cmap=cmap,
                             extend='both')
    ax2.set_title(f'Precipitation WRF')

    # Añadir barra de color en la parte inferior
    cbar = plt.colorbar(
        contour_for,
        cax=cbar_ax,
        orientation='horizontal',
        pad=0, #0.5
        shrink=0.5, #0.8
        aspect=120, #40
        fraction=0.02,
        label='Precipitation (mm/h)')

    
       
    ##-----------------------------------------------------------------------------    
    # Dominio de 3 km
    ##-----------------------------------------------------------------------------
    #fig.suptitle(f"Comparación de Precipitación\n"
    #            f"Observación (GPM): 10 km Vs Pronóstico (WRF): 3 km\n"
    #            f"Plazo temporal {time_str}", 
    #            fontsize=16, y=0.25)

    # Dominio de 3 km
    fig.suptitle(f"Precipitation Comparison\n"
                  f"Observation (GPM): 10 km and Forecast (WRF): 3 km\n"
                  f"Temporary Period {time_str}", 
                  fontsize=16) #y=0.22

    
    # Guardar la figura
    if save_path is None:
        save_path = config.path_figures
        
    safe_time_str = time_str.replace(' ', '_').replace(':', '').replace('-', '')
    plt.savefig(os.path.join(save_path, f'Precipitacion_{safe_time_str}_GPM_WRF.png'), 
                dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    #plt.show()
    
    return fig

def generate_comparison_plots(ds_observed, ds_model, max_plots=25):
    """Genera gráficos de comparación para todos los tiempos disponibles"""
    print("Generando gráficos de comparación...")
    
    for i in range(min(max_plots, len(ds_observed.time))):
        try:
            #print(f"Generando gráficos para tiempo índice {i}...")
            plot_prec2(ds_observed, ds_model, i, window=config.accum_window_to_mode)
        except Exception as e:
            print(f"Error generando gráficos para tiempo {i}: {e}")

    print("Gráficos generados exitosamente!")
