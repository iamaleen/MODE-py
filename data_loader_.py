##-----------------------------------------------------------------------------
## Carga de datos GPM y WRF - CON VENTANA DE ACUMULACIÓN
##-----------------------------------------------------------------------------

import numpy as np
import xarray as xr
import pandas as pd
import h5py
import glob
import re
import os
from tqdm import tqdm
import config

def read_h5py_data(filepath):
    """Lee un archivo GPM con h5py y devuelve un Dataset de xarray"""
    with h5py.File(filepath, 'r') as f:
        precip = f['Grid/precipitation'][:]  # shape: (1, 3600, 1800)
        lat = f['Grid/lat'][:]              # shape: (1800,)
        lon = f['Grid/lon'][:]              # shape: (3600,)
        
        filename = os.path.basename(filepath)
        datetime_part = filename.split('.')[4]  # '20190501-S010000'
        date_str, time_str = datetime_part.split('-')[:2]
        time = pd.to_datetime(f"{date_str}{time_str[1:]}", format='%Y%m%d%H%M%S')
        
        precip = np.where(precip == -9999.9, np.nan, precip)
        
        return xr.Dataset(
            {
                "precipitation": (('time', 'lon', 'lat'), precip),
            },
            coords={
                'lon': lon,
                'lat': lat,
                'time': [time]
            }
        )


def accumulate_gpm_flexible(gpm_files, accumulation_window=config.accum_window_to_mode):
    """
    Acumula datos GPM según la ventana especificada (1H, 3H, 6H)
    
    Args:
        gpm_files: Lista de archivos GPM
        accumulation_window: '1H', '3H', o '6H'
    """
    print(f"Acumulando datos GPM cada {accumulation_window}...")
    
    # Leer todos los archivos GPM
    datasets = []
    for file in tqdm(gpm_files, desc="Leyendo archivos GPM"):
        try:
            ds = read_h5py_data(file)
            datasets.append(ds)
        except Exception as e:
            print(f"Error procesando {os.path.basename(file)}: {str(e)}")
            continue
    
    if not datasets:
        raise ValueError("No se pudieron cargar archivos GPM válidos")
    
    # Combinar todos los datasets
    ds_gpm = xr.concat(datasets, dim='time')
    ds_gpm = ds_gpm.precipitation.squeeze().transpose('time', 'lat', 'lon')
    
    print(f"Datos GPM originales: {len(ds_gpm.time)} pasos de tiempo")
    
    if accumulation_window == '1H':
        # Usar método original para 1 hora
        return accumulate_gpm_hourly_original(ds_gpm)
    else:
        # Usar resample para 3H y 6H
        return accumulate_gpm_with_resample(ds_gpm, accumulation_window)




def accumulate_gpm_hourly_original(ds_gpm):
    """
    Acumula datos GPM HHR (Half-hourly) en acumulados horarios,
    donde el timestamp del resultado representa el FINAL del intervalo horario.
    
    Ejemplo:
      - Valor etiquetado como '01:00' = precipitación de 00:00–01:00
      - Se construye sumando GPM[00:30] + GPM[01:00]
    
    Esto coincide con la convención usada en WRF y en verificación meteorológica.
    """
    
    # Obtener todos los tiempos
    gpm_times = pd.to_datetime(ds_gpm.time.values)
    
    # Agrupar por hora calendario (inicio del intervalo horario)
    hourly_groups = {}
    
    for i, time_val in enumerate(gpm_times):
        # CORRECCIÓN: restar 30 min para obtener el INICIO del intervalo real
        interval_start = time_val - pd.Timedelta(minutes=30)
        # Asignar al inicio de la hora calendario (ej: 00:30 → 00:00)
        hour_key = interval_start.replace(minute=0, second=0, microsecond=0)
        
        if hour_key not in hourly_groups:
            hourly_groups[hour_key] = []
        hourly_groups[hour_key].append(i)
    
    print(f"Se encontraron {len(hourly_groups)} grupos horarios")
    
    # Acumular precipitación para cada hora calendario
    accumulated_data = []
    accumulated_times_start = []  # Tiempos de INICIO del intervalo
    
    for hour_key, indices in sorted(hourly_groups.items()):
        if len(indices) >= 1:
            hourly_precip = ds_gpm.isel(time=indices).sum(dim='time')
            accumulated_data.append(hourly_precip)
            accumulated_times_start.append(hour_key)
    
    # Crear nuevo DataArray con tiempos de INICIO
    if accumulated_data:
        ds_gpm_hourly_start = xr.concat(accumulated_data, dim='time')
        ds_gpm_hourly_start = ds_gpm_hourly_start.assign_coords(time=accumulated_times_start)
        
        accumulated_times_end = [t + pd.Timedelta(hours=1) for t in accumulated_times_start]
        ds_gpm_hourly = ds_gpm_hourly_start.assign_coords(time=accumulated_times_end)
        return ds_gpm_hourly
    else:
        raise ValueError("No se pudieron acumular datos horarios")











def accumulate_gpm_with_resample(ds_gpm, window):
    """Acumulación usando resample para 3H y 6H"""
    print(f"Aplicando resample cada {window}...")
    
    # Resample y suma
    ds_accumulated = ds_gpm.resample(time=window, origin='start_day').sum(skipna=True)
    
    print(f"Datos GPM acumulados cada {window}: {len(ds_accumulated.time)} pasos de tiempo")
    
    return ds_accumulated


def accumulate_wrf_flexible(ds_wrf, accumulation_window=config.accum_window_to_mode):
    """
    Acumula datos WRF según la ventana especificada MANTENIENDO NOMBRES ORIGINALES
    """
    if accumulation_window == '1H':
        # WRF ya es horario - mantener como está
        print("Datos WRF horarios - sin acumulación adicional")
        return ds_wrf
    else:
        # Acumular WRF para 3H o 6H
        print(f"Acumulando datos WRF cada {accumulation_window}...")
        
        # Crear un nuevo Dataset con todas las variables acumuladas
        ds_accumulated = xr.Dataset()
        
        # Acumular RAINNC si existe
        if 'RAINNC' in ds_wrf:
            rainnc_accumulated = ds_wrf['RAINNC'].resample(time=accumulation_window, origin='start_day').sum(skipna=True)
            ds_accumulated['RAINNC'] = rainnc_accumulated
            print(f"RAINNC acumulado cada {accumulation_window}")
        
        # Acumular RAINC si existe
        if 'RAINC' in ds_wrf:
            rainc_accumulated = ds_wrf['RAINC'].resample(time=accumulation_window, origin='start_day').sum(skipna=True)
            ds_accumulated['RAINC'] = rainc_accumulated
            print(f"RAINC acumulado cada {accumulation_window}")
        
        # Mantener coordenadas originales
        ds_accumulated = ds_accumulated.assign_coords({
            'time': ds_wrf.time.resample(time=accumulation_window, origin='start_day').first(),
            'lat': ds_wrf.lat,
            'lon': ds_wrf.lon
        })
        
        # Mantener otras variables importantes si existen
        if 'XLAT' in ds_wrf:
            ds_accumulated['XLAT'] = ds_wrf['XLAT'].isel(time=0) if 'time' in ds_wrf['XLAT'].dims else ds_wrf['XLAT']
        if 'XLONG' in ds_wrf:
            ds_accumulated['XLONG'] = ds_wrf['XLONG'].isel(time=0) if 'time' in ds_wrf['XLONG'].dims else ds_wrf['XLONG']
        
        print(f"Datos WRF acumulados cada {accumulation_window}: {len(ds_accumulated.time)} pasos de tiempo")
        print(f"Variables disponibles: {list(ds_accumulated.data_vars)}")
        
        return ds_accumulated




def extract_time_from_wrf_filename(filename):
    """Extrae el tiempo real del nombre del archivo WRF"""
    ##-----------------------------------------------------------------------------
    # Dominio de 3 km
    ##-----------------------------------------------------------------------------
    match = re.search(r'wrfout_d03_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', filename)
    
    ##-----------------------------------------------------------------------------
    # Dominio de 1 km
    ##-----------------------------------------------------------------------------
    #match = re.search(r'wrfout_d02_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', filename)
    if match:
        time_str = match.group(1)
        return pd.to_datetime(time_str, format='%Y-%m-%d_%H:%M:%S')
    else:
        raise ValueError(f"No se pudo extraer tiempo del archivo: {filename}")

def preprocess_wrf(ds, file_time):
    """Función de preprocesamiento para archivos WRF"""
    if 'Time' in ds.dims and ds.dims['Time'] == 1:
        ds = ds.isel(Time=0)
    
    if 'south_north' in ds.dims and 'west_east' in ds.dims:
        ds = ds.rename({
            'south_north': 'lat',
            'west_east': 'lon'
        })
    
    ds.attrs['real_time'] = file_time
    return ds

def load_wrf_with_correct_times(wrf_files, accumulation_window=config.accum_window_to_mode):
    """Carga datos WRF con tiempos reales y acumulación flexible"""
    print("Cargando datos WRF con tiempos reales...")
    
    datasets = []
    wrf_times = []
    
    for file in tqdm(wrf_files, desc="Procesando archivos WRF"):
        try:
            file_time = extract_time_from_wrf_filename(file)
            wrf_times.append(file_time)
            
            ds = xr.open_dataset(file)
            ds = preprocess_wrf(ds, file_time)
            datasets.append(ds)
            
        except Exception as e:
            print(f"Error procesando {os.path.basename(file)}: {str(e)}")
            continue
    
    if not datasets:
        raise ValueError("No se pudieron cargar archivos WRF válidos")
    
    # Combinar datasets
    ds_wrf = xr.concat(datasets, dim='time')
    ds_wrf = ds_wrf.assign_coords(time=wrf_times)
    
    print(f"Tiempos WRF reales: {wrf_times[:5]} ... {wrf_times[-5:]}")
    
    # Aplicar acumulación si es necesario
    if accumulation_window != '1H':
        return accumulate_wrf_flexible(ds_wrf, accumulation_window)
    else:
        return ds_wrf

##-----------------------------------------------------------------------------
# FUNCIONES PRINCIPALES 
##-----------------------------------------------------------------------------

def load_gpm_data(accumulation_window=config.accum_window_to_mode):
    """
    Carga datos GPM con acumulación flexible
    
    Args:
        accumulation_window: '1H', '3H', o '6H' (default: '1H')
    """
    #gpm_hourly_files = sorted(glob.glob(config.path_gpm + '3B-HHR.MS.MRG.3IMERG.20240323*.HDF5'))
    gpm_hourly_files = sorted(glob.glob(config.path_gpm + '3B-HHR.MS.MRG.3IMERG*.HDF5'))
    return accumulate_gpm_flexible(gpm_hourly_files, accumulation_window)

def load_wrf_data(accumulation_window=config.accum_window_to_mode):
    """
    Carga datos WRF con acumulación flexible
    
    Args:
        accumulation_window: '1H', '3H', o '6H' (default: '1H')
    """

    ##-----------------------------------------------------------------------------
    # Dominio de 3 km
    ##-----------------------------------------------------------------------------
    #wrf_files = sorted(glob.glob(config.path_wrf + 'wrfout_d03_2024-03-23_*'))
    wrf_files = sorted(glob.glob(config.path_wrf + 'wrfout_*'))

    ##-----------------------------------------------------------------------------
    # Dominio de 1 km
    ##-----------------------------------------------------------------------------
    #wrf_files = sorted(glob.glob(config.path_wrf + 'wrfout_d02_2019-05-20_*'))
    
    # Cargar datos WRF base con acumulación
    return load_wrf_with_correct_times(wrf_files, accumulation_window)


