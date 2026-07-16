##-----------------------------------------------------------------------------
## Preprocesamiento y alineación de datos - VERSIÓN ADAPTADA PARA ACUMULACIÓN
##-----------------------------------------------------------------------------

import xarray as xr
import pandas as pd
import numpy as np
from data_loader_ import load_wrf_with_correct_times
import config

def extract_wrf_coordinates(ds_wrf):
    """Extrae coordenadas lat/lon reales de WRF"""
    print("Extrayendo coordenadas WRF...")
    
    if 'XLAT' in ds_wrf and 'XLONG' in ds_wrf:
        if 'time' in ds_wrf.XLAT.dims and len(ds_wrf.time) > 0:
            wrf_lat = ds_wrf.XLAT.isel(time=0).values
            wrf_lon = ds_wrf.XLONG.isel(time=0).values
        else:
            wrf_lat = ds_wrf.XLAT.values
            wrf_lon = ds_wrf.XLONG.values
        
        print(f"Rango lat WRF: {wrf_lat.min():.2f} to {wrf_lat.max():.2f}")
        print(f"Rango lon WRF: {wrf_lon.min():.2f} to {wrf_lon.max():.2f}")
        
        return wrf_lat, wrf_lon
    else:
        raise ValueError("No se encontraron coordenadas XLAT/XLONG en WRF")

def create_wrf_dataarray(da_precip, wrf_lat, wrf_lon, wrf_times):
    """Crea DataArray de WRF con coordenadas correctas"""
    print("Creando DataArray WRF con coordenadas...")
    
    # Crear coordenadas para las dimensiones
    lat_coords = wrf_lat.mean(axis=1) if wrf_lat.ndim == 2 else wrf_lat
    lon_coords = wrf_lon.mean(axis=0) if wrf_lon.ndim == 2 else wrf_lon
    
    # Crear DataArray con coordenadas explícitas
    wrf_da = xr.DataArray(
        da_precip.values,
        dims=['time', 'lat', 'lon'],
        coords={
            'time': wrf_times,
            'lat': lat_coords,
            'lon': lon_coords
        },
        attrs=da_precip.attrs
    )
    
    return wrf_da


def calculate_hourly_precipitation(ds_wrf, accumulation_window=config.accum_window_to_mode):
    """
    Calcula precipitación horaria o acumulada según la ventana
    
    Args:
        ds_wrf: Dataset WRF (puede ser original o acumulado)
        accumulation_window: '1H', '3H', o '6H'
    """
    print(f"Calculando precipitación para ventana: {accumulation_window}")
    
    # Verificar qué variables están disponibles
    available_vars = list(ds_wrf.data_vars)
    #print(f"Variables disponibles en WRF: {available_vars}")
    
    # Extraer precipitación total
    if 'RAINNC' in ds_wrf and 'RAINC' in ds_wrf:
        precip_accumulated = ds_wrf['RAINNC'] + ds_wrf['RAINC']
        print("Precipitación total calculada (RAINNC + RAINC)")
    elif 'RAINNC' in ds_wrf:
        precip_accumulated = ds_wrf['RAINNC']
        print("Usando solo RAINNC")
    elif 'RAINC' in ds_wrf:
        precip_accumulated = ds_wrf['RAINC']
        print("Usando solo RAINC")
    else:
        # Buscar cualquier variable de precipitación
        precip_vars = [var for var in ds_wrf.data_vars if 'rain' in var.lower() or 'precip' in var.lower()]
        if precip_vars:
            precip_accumulated = ds_wrf[precip_vars[0]]
            print(f"Usando variable de precipitación: {precip_vars[0]}")
        else:
            raise ValueError("No se encontró variable de precipitación en datos WRF")
    
    if accumulation_window == accumulation_window:
        # Para 1H: calcular diferencia horaria
        print("Calculando precipitación horaria...")
        rain_hourly = xr.zeros_like(precip_accumulated)
        rain_hourly[0] = precip_accumulated[0]

        for i in range(1, len(ds_wrf.time)):
            rain_hourly[i] = precip_accumulated[i] - precip_accumulated[i-1]

        rain_hourly.attrs['description'] = 'Hourly total precipitation'
        rain_hourly.name = 'precipitation'
        return rain_hourly
    
    else:
        # Para 3H y 6H: usar directamente la precipitación acumulada
        print(f"Usando precipitación acumulada cada {accumulation_window}")
        precip_accumulated.name = 'precipitation'
        precip_accumulated.attrs['description'] = f'Accumulated precipitation every {accumulation_window}'
        return precip_accumulated




def align_temporal_data(gpm_data, wrf_data, accumulation_window=config.accum_window_to_mode):
    """
    Alinea datos temporales de GPM y WRF según la ventana de acumulación
    
    Args:
        gpm_data: DataArray GPM
        wrf_data: DataArray WRF  
        accumulation_window: '1H', '3H', o '6H'
    """
    print(f"Alineando datos temporales para ventana: {accumulation_window}")
    
    # Obtener tiempos de ambos datasets
    gpm_times = pd.to_datetime(gpm_data.time.values)
    wrf_times = pd.to_datetime(wrf_data.time.values)
    
    print(f"Tiempos GPM: {len(gpm_times)} - {gpm_times[:3]} ... {gpm_times[-3:]}")
    print(f"Tiempos WRF: {len(wrf_times)} - {wrf_times[:3]} ... {wrf_times[-3:]}")
    
    # Encontrar tiempos comunes
    common_times = sorted(set(gpm_times).intersection(set(wrf_times)))
    print(f"Tiempos comunes encontrados: {len(common_times)}")
    
    if not common_times:
        print("No hay tiempos comunes entre GPM y WRF")
        print("Tiempos GPM disponibles:")
        for t in gpm_times:
            print(f"  {t}")
        print("Tiempos WRF disponibles:")
        for t in wrf_times:
            print(f"  {t}")
        raise ValueError("No hay tiempos comunes entre GPM y WRF")
    
    #print(f"Tiempos comunes: {common_times}")
    
    # Seleccionar solo tiempos comunes
    gpm_aligned = gpm_data.sel(time=common_times)
    wrf_aligned = wrf_data.sel(time=common_times)
    
    #print(f"Dimensiones finales GPM: {gpm_aligned.dims}")
    #print(f"Dimensiones finales WRF: {wrf_aligned.dims}")
    
    return gpm_aligned, wrf_aligned



def preprocess_datasets(ds_gpm, ds_wrf, accumulation_window='1H'):
    """
    Preprocesa y alinea los datasets GPM y WRF con acumulación flexible
    """
    print(f"\n{'='*60}")
    print(f"INICIANDO PREPROCESAMIENTO - Acumulación: {accumulation_window}")
    print(f"{'='*60}")
    
    # Verificar tipos de datos
    #print(f"Tipo de datos GPM: {type(ds_gpm)}")
    #print(f"Tipo de datos WRF: {type(ds_wrf)}")
    #print(f"Variables WRF disponibles: {list(ds_wrf.data_vars) if hasattr(ds_wrf, 'data_vars') else 'DataArray'}")
    
    # Calcular precipitación WRF según la ventana
    wrf_precip = calculate_hourly_precipitation(ds_wrf, accumulation_window)
    
    # Extraer coordenadas WRF
    wrf_lat, wrf_lon = extract_wrf_coordinates(ds_wrf)
    
    # Obtener tiempos de WRF
    wrf_times = pd.to_datetime(ds_wrf.time.values)
    
    # Crear DataArray de WRF con coordenadas correctas
    wrf_da = create_wrf_dataarray(wrf_precip, wrf_lat, wrf_lon, wrf_times)
    
    # Alinear temporalmente
    gpm_temp_aligned, wrf_temp_aligned = align_temporal_data(ds_gpm, wrf_da, accumulation_window)
    
    # Recortar a región común
    lat_min = max(gpm_temp_aligned.lat.min(), wrf_temp_aligned.lat.min())
    lat_max = min(gpm_temp_aligned.lat.max(), wrf_temp_aligned.lat.max())
    lon_min = max(gpm_temp_aligned.lon.min(), wrf_temp_aligned.lon.min())
    lon_max = min(gpm_temp_aligned.lon.max(), wrf_temp_aligned.lon.max())

    print(f"Región común: lat({lat_min:.2f}, {lat_max:.2f}), lon({lon_min:.2f}, {lon_max:.2f})")

    # Recortar ambos datasets a la región común
    gpm_aligned = gpm_temp_aligned.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )

    wrf_aligned = wrf_temp_aligned.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )

    # Preparar datos para MODE
    ds_observed = gpm_aligned  # 10km resolution
    ds_model = wrf_aligned     # 3km resolution

    print(f"\n DATOS FINALES PARA MODE:")
    print(f"   GPM (observado): {ds_observed.shape} - Resolución: ~10km")
    print(f"   WRF (pronóstico): {ds_model.shape} - Resolución: ~3km")
    print(f"   Ventana acumulación: {accumulation_window}")
    print(f"   Tiempos comunes: {len(ds_observed.time)}")
    
    return ds_observed, ds_model


