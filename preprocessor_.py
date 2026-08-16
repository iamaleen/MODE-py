#=============================================================================
# preprocessor_.py
#=============================================================================
# Data Preprocessing and Alignment Module for MODE-py
# This module handles the preparation of raw WRF and GPM datasets for the 
# verification pipeline. It ensures temporal synchronization, calculates 
# flexible precipitation accumulations (1H, 3H, 6H) from WRF variables, and 
# spatially crops both datasets to their common geographic domain.
#
# Key features:
#   - Extraction and standardization of WRF curvilinear coordinates (XLAT/XLONG).
#   - Flexible temporal accumulation routines for both model and satellite data.
#   - Bounding box intersection for spatial alignment.
#=============================================================================


import xarray as xr
import pandas as pd
import numpy as np
from data_loader_ import load_wrf_with_correct_times
import config


def extract_wrf_coordinates(ds_wrf):
    """Extracts actual lat/lon coordinates from WRF."""

    if 'XLAT' in ds_wrf and 'XLONG' in ds_wrf:
        if 'time' in ds_wrf.XLAT.dims and len(ds_wrf.time) > 0:
            wrf_lat = ds_wrf.XLAT.isel(time=0).values
            wrf_lon = ds_wrf.XLONG.isel(time=0).values
        else:
            wrf_lat = ds_wrf.XLAT.values
            wrf_lon = ds_wrf.XLONG.values

        return wrf_lat, wrf_lon
    else:
        raise ValueError("No XLAT/XLONG coordinates found in WRF.")



def create_wrf_dataarray(da_precip, wrf_lat, wrf_lon, wrf_times):
    """Create a WRF DataArray with the correct coordinates."""

    # Create coordinates for the dimensions
    lat_coords = wrf_lat.mean(axis=1) if wrf_lat.ndim == 2 else wrf_lat
    lon_coords = wrf_lon.mean(axis=0) if wrf_lon.ndim == 2 else wrf_lon
    
    # Create a DataArray with explicit coordinates
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
    """Calculates hourly or accumulated precipitation based on the window."""

    # Extract total precipitation
    if 'RAINNC' in ds_wrf and 'RAINC' in ds_wrf:
        precip_accumulated = ds_wrf['RAINNC'] + ds_wrf['RAINC']
        print("Calculated total precipitation RAINNC + RAINC")
    elif 'RAINNC' in ds_wrf:
        precip_accumulated = ds_wrf['RAINNC']
        print("Using only RAINNC")
    elif 'RAINC' in ds_wrf:
        precip_accumulated = ds_wrf['RAINC']
        print("Using only RAINC")
    else:
         raise ValueError("No precipitation variable was found in the WRF data.")
                 
    
    if accumulation_window == accumulation_window:

        # For 1H: calculate the time difference
        rain_hourly = xr.zeros_like(precip_accumulated)
        rain_hourly[0] = precip_accumulated[0]

        for i in range(1, len(ds_wrf.time)):
            rain_hourly[i] = precip_accumulated[i] - precip_accumulated[i-1]

        rain_hourly.attrs['description'] = 'Hourly total precipitation'
        rain_hourly.name = 'precipitation'
        return rain_hourly
    
    else:
        # For 3H and 6H: use accumulated precipitation directly  
        precip_accumulated.name = 'precipitation'
        precip_accumulated.attrs['description'] = f'Accumulated precipitation every {accumulation_window}'
        return precip_accumulated



def align_temporal_data(gpm_data, wrf_data, accumulation_window=config.accum_window_to_mode):
    """Align GPM and WRF time data according to the accumulation window"""

    # Get timestamps from both datasets
    gpm_times = pd.to_datetime(gpm_data.time.values)
    wrf_times = pd.to_datetime(wrf_data.time.values)
    
    print(f"GPM Times: {len(gpm_times)} - {gpm_times[:3]} ... {gpm_times[-3:]}")
    print(f"WRF Times: {len(wrf_times)} - {wrf_times[:3]} ... {wrf_times[-3:]}")
    
    # Finding common times
    common_times = sorted(set(gpm_times).intersection(set(wrf_times)))
    print(f"Common times encountered: {len(common_times)}")
    
    if not common_times:
        print("There are no common time steps between GPM and WRF.")
        print("Available GPM times:")
        for t in gpm_times:
            print(f"  {t}")
        print("Available WRF times:")
        for t in wrf_times:
            print(f"  {t}")
        raise ValueError("There are no common time steps between GPM and WRF.")

    # Select only common time signatures
    gpm_aligned = gpm_data.sel(time=common_times)
    wrf_aligned = wrf_data.sel(time=common_times)

    return gpm_aligned, wrf_aligned



def preprocess_datasets(ds_gpm, ds_wrf, accumulation_window='1H'):
    """Preprocess and align the GPM and WRF datasets with flexible accumulation."""
  
    # Calculate WRF precipitation based on the window
    wrf_precip = calculate_hourly_precipitation(ds_wrf, accumulation_window)
    
    # Extract WRF coordinates
    wrf_lat, wrf_lon = extract_wrf_coordinates(ds_wrf)
    
    # Get WRF times
    wrf_times = pd.to_datetime(ds_wrf.time.values)
    
    # Create WRF DataArray with coordinates
    wrf_da = create_wrf_dataarray(wrf_precip, wrf_lat, wrf_lon, wrf_times)
    
    # Temporarily align
    gpm_temp_aligned, wrf_temp_aligned = align_temporal_data(ds_gpm, wrf_da, accumulation_window)
    
    # Crop to common region
    lat_min = max(gpm_temp_aligned.lat.min(), wrf_temp_aligned.lat.min())
    lat_max = min(gpm_temp_aligned.lat.max(), wrf_temp_aligned.lat.max())
    lon_min = max(gpm_temp_aligned.lon.min(), wrf_temp_aligned.lon.min())
    lon_max = min(gpm_temp_aligned.lon.max(), wrf_temp_aligned.lon.max())

    gpm_aligned = gpm_temp_aligned.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )

    wrf_aligned = wrf_temp_aligned.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )

    # Prepare data for MODE-py
    ds_observed = gpm_aligned  
    ds_model = wrf_aligned     

    print(f"\n Final Data for MODE-py:")
    print(f"   GPM (observed): {ds_observed.shape}")
    print(f"   WRF (forecasted): {ds_model.shape}")
    print(f"   Accumulation window: {accumulation_window}")
    print(f"   Common times: {len(ds_observed.time)}")
    
    return ds_observed, ds_model


