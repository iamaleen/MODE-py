#=============================================================================
# data_loader_.py
#=============================================================================
# Data Ingestion Module for MODE-py
# This module is responsible for reading raw observational and forecast data 
# from disk. It parses GPM IMERG HDF5 files (handling half-hourly to hourly 
# accumulation) and WRF NetCDF output files (extracting real-time timestamps 
# from filenames).
#
# Key features:
#   - Robust HDF5 parsing for GPM IMERG V07 data.
#   - Regex-based timestamp extraction for WRF output files.
#   - Flexible accumulation logic (1H, 3H, 6H) for both data sources.
#   - Quality control.
#=============================================================================


import numpy as np
import xarray as xr
import pandas as pd
import h5py
import glob
import re
import os
from tqdm import tqdm
import config


##-----------------------------------------------------------------------------
## read_h5py_data function
##-----------------------------------------------------------------------------
def read_h5py_data(filepath):
    """Read a GPM file with h5py and return an xarray Dataset."""

    with h5py.File(filepath, 'r') as f:
        precip = f['Grid/precipitation'][:]  
        lat = f['Grid/lat'][:]              
        lon = f['Grid/lon'][:]              
        
        filename = os.path.basename(filepath)
        datetime_part = filename.split('.')[4]  
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


##-----------------------------------------------------------------------------
## accumulate_gpm_flexible function
##-----------------------------------------------------------------------------
def accumulate_gpm_flexible(gpm_files, accumulation_window=config.accum_window_to_mode):
    """Accumulates GPM data according to the specified window (1H, 3H, 6H)."""

    # Read all GPM files
    datasets = []
    for file in tqdm(gpm_files, desc="Reading GPM files"):
        try:
            ds = read_h5py_data(file)
            datasets.append(ds)
        except Exception as e:
            print(f"Error processing {os.path.basename(file)}: {str(e)}")
            continue
    
    if not datasets:
        raise ValueError("Valid GPM files could not be loaded.")
    
    # Combine all datasets
    ds_gpm = xr.concat(datasets, dim='time')
    ds_gpm = ds_gpm.precipitation.squeeze().transpose('time', 'lat', 'lon')
    
    print(f"Datos GPM originales: {len(ds_gpm.time)} pasos de tiempo")
    
    if accumulation_window == '1H':
        print(f"Accumulating GPM data every {accumulation_window}...")
        return accumulate_gpm_hourly_original(ds_gpm)
    else:
        # Use resample for 3H and 6H
        return accumulate_gpm_with_resample(ds_gpm, accumulation_window)



##-----------------------------------------------------------------------------
## accumulate_gpm_hourly_original function
##-----------------------------------------------------------------------------
def accumulate_gpm_hourly_original(ds_gpm):
    """
    Aggregates half-hourly GPM data into hourly totals,
    where the result timestamp represents the END of the hourly interval.
      - Calculated by summing GPM[00:30] + GPM[01:00]
    """
    
    # Get all times
    gpm_times = pd.to_datetime(ds_gpm.time.values)
    
    # Group by calendar hour
    hourly_groups = {}
    
    for i, time_val in enumerate(gpm_times):
        # Subtract 30 min to obtain the start of the actual interval
        interval_start = time_val - pd.Timedelta(minutes=30)
        # Assign to the start of the clock hour  
        hour_key = interval_start.replace(minute=0, second=0, microsecond=0)
        
        if hour_key not in hourly_groups:
            hourly_groups[hour_key] = []
        hourly_groups[hour_key].append(i)
    
    print(f"{len(hourly_groups)} hourly groups were found.")
    
    # Accumulate precipitation for each calendar hour.
    accumulated_data = []
    accumulated_times_start = []   
    
    for hour_key, indices in sorted(hourly_groups.items()):
        if len(indices) >= 1:
            hourly_precip = ds_gpm.isel(time=indices).sum(dim='time')
            accumulated_data.append(hourly_precip)
            accumulated_times_start.append(hour_key)
    
    # Create new DataArray with start times
    if accumulated_data:
        ds_gpm_hourly_start = xr.concat(accumulated_data, dim='time')
        ds_gpm_hourly_start = ds_gpm_hourly_start.assign_coords(time=accumulated_times_start)
        
        accumulated_times_end = [t + pd.Timedelta(hours=1) for t in accumulated_times_start]
        ds_gpm_hourly = ds_gpm_hourly_start.assign_coords(time=accumulated_times_end)
        return ds_gpm_hourly
    else:
        raise ValueError("Hourly data could not be accumulated.")


##-----------------------------------------------------------------------------
## accumulate_gpm_with_resample function
##-----------------------------------------------------------------------------
def accumulate_gpm_with_resample(ds_gpm, window):
    """Accumulation using resample for 3H and 6H"""

    ds_accumulated = ds_gpm.resample(time=window, origin='start_day').sum(skipna=True)
    
    print(f"Accumulated GPM data every {window}: {len(ds_accumulated.time)} time steps.")
    return ds_accumulated


##-----------------------------------------------------------------------------
## accumulate_wrf_flexible function
##-----------------------------------------------------------------------------
def accumulate_wrf_flexible(ds_wrf, accumulation_window=config.accum_window_to_mode):
    """
    Accumulates WRF data according to the specified window
    """
    if accumulation_window == '1H':
        # WRF schedule: keep as is
        print("Hourly WRF data.")
        return ds_wrf
    else:
        # Accumulate WRF for 3H or 6H
        print(f"Accumulating WRF data every {accumulation_window}...")
        
        # Create a new dataset with all accumulated variables 
        ds_accumulated = xr.Dataset()
        
        # Accumulate RAINNC if it exists
        if 'RAINNC' in ds_wrf:
            rainnc_accumulated = ds_wrf['RAINNC'].resample(time=accumulation_window, origin='start_day').sum(skipna=True)
            ds_accumulated['RAINNC'] = rainnc_accumulated
        
        # Accumulate RAINC if it exists
        if 'RAINC' in ds_wrf:
            rainc_accumulated = ds_wrf['RAINC'].resample(time=accumulation_window, origin='start_day').sum(skipna=True)
            ds_accumulated['RAINC'] = rainc_accumulated
    
        # Keep original coordinates
        ds_accumulated = ds_accumulated.assign_coords({
            'time': ds_wrf.time.resample(time=accumulation_window, origin='start_day').first(),
            'lat': ds_wrf.lat,
            'lon': ds_wrf.lon
        })
        
        # Retain other important variables if they exist
        if 'XLAT' in ds_wrf:
            ds_accumulated['XLAT'] = ds_wrf['XLAT'].isel(time=0) if 'time' in ds_wrf['XLAT'].dims else ds_wrf['XLAT']
        if 'XLONG' in ds_wrf:
            ds_accumulated['XLONG'] = ds_wrf['XLONG'].isel(time=0) if 'time' in ds_wrf['XLONG'].dims else ds_wrf['XLONG']

        return ds_accumulated


##-----------------------------------------------------------------------------
## extract_time_from_wrf_filename function
##-----------------------------------------------------------------------------
def extract_time_from_wrf_filename(filename):
    """Extract the real time from the WRF filename."""

    match = re.search(r'wrfout_d03_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', filename)

    if match:
        time_str = match.group(1)
        return pd.to_datetime(time_str, format='%Y-%m-%d_%H:%M:%S')
    else:
        raise ValueError(f"Could not extract time from the file: {filename}")


##-----------------------------------------------------------------------------
## preprocess_wrf function
##-----------------------------------------------------------------------------
def preprocess_wrf(ds, file_time):
    """Preprocessing function for WRF files"""

    if 'Time' in ds.dims and ds.dims['Time'] == 1:
        ds = ds.isel(Time=0)
    
    if 'south_north' in ds.dims and 'west_east' in ds.dims:
        ds = ds.rename({
            'south_north': 'lat',
            'west_east': 'lon'
        })
    
    ds.attrs['real_time'] = file_time
    return ds


##-----------------------------------------------------------------------------
## load_wrf_with_correct_times function
##-----------------------------------------------------------------------------
def load_wrf_with_correct_times(wrf_files, accumulation_window=config.accum_window_to_mode):
    """Loads WRF data with real-time timestamps and flexible accumulation."""

    datasets = []
    wrf_times = []
    
    for file in tqdm(wrf_files, desc="Processing WRF files"):
        try:
            file_time = extract_time_from_wrf_filename(file)
            wrf_times.append(file_time)
            
            ds = xr.open_dataset(file)
            ds = preprocess_wrf(ds, file_time)
            datasets.append(ds)
            
        except Exception as e:
            print(f"Error processing {os.path.basename(file)}: {str(e)}")
            continue
    
    if not datasets:
        raise ValueError("Valid WRF files could not be loaded.")
    
    # Combine datasets
    ds_wrf = xr.concat(datasets, dim='time')
    ds_wrf = ds_wrf.assign_coords(time=wrf_times)
    
    print(f"Actual WRF timings: {wrf_times[:5]} ... {wrf_times[-5:]}")
    
    # Apply accumulation if necessary
    if accumulation_window != '1H':
        print("Loading WRF data with real-time timestamps...")
        return accumulate_wrf_flexible(ds_wrf, accumulation_window)
    else:
        return ds_wrf






##-----------------------------------------------------------------------------
# MAIN FUNCTIONS
##-----------------------------------------------------------------------------

def load_gpm_data(accumulation_window=config.accum_window_to_mode):
    """Carga datos GPM con acumulación flexible"""

    gpm_hourly_files = sorted(glob.glob(config.path_gpm + '3B-HHR.MS.MRG.3IMERG*.HDF5'))
    return accumulate_gpm_flexible(gpm_hourly_files, accumulation_window)

def load_wrf_data(accumulation_window=config.accum_window_to_mode):
    """Load WRF data with flexible accumulation"""

    wrf_files = sorted(glob.glob(config.path_wrf + 'wrfout_*'))
    return load_wrf_with_correct_times(wrf_files, accumulation_window)


