#=============================================================================
# mode_verifier.py
#=============================================================================
# Core Algorithmic Implementation of the MODE-py Framework
# This module contains the MODE3DVerifier class, which encapsulates the 
# complete object-based verification workflow. It handles:
#   - Adaptive multi-resolution Gaussian smoothing and thresholding.
#   - Connected component labeling and recursive spatial merging.
#   - Spatio-temporal graph-based object grouping and persistence tracking.
#   - Fuzzy logic interest matrix calculation and greedy matching.
#   - Extraction of advanced verification metrics (MMI, GSS, GSS_Obj-Based).
#
# Dependencies: numpy, xarray, pandas, scipy, scikit-image, cartopy, matplotlib
#=============================================================================

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter, label, binary_closing
from skimage.measure import regionprops
from matplotlib.patches import Ellipse
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from scipy.spatial import ConvexHull
import os
import config
import field_visualization

##-----------------------------------------------------------------------------
## 3. MODE-py Class MODE3DVerifier (Method for Object-based Diagnostic Evaluation) 
##-----------------------------------------------------------------------------

class MODE3DVerifier:
    """
    Implementation of the MODE method for (Time, Lat, Lon) data,
    handling different spatial resolutions.
    """
    ##-------------------------------------------------------------------------
    ## 3.1.__init__ method 
    ##-------------------------------------------------------------------------
     
    def __init__(self, forecast: xr.DataArray, observed: xr.DataArray, 
             threshold: float = 1.5, conv_radius_forecast: int = 5, 
             conv_radius_observed: int = 3, time_window: int = 1, 
             min_object_size: int = 5, 
             min_object_size_forecast: int = 10,  
             min_object_size_observed: int = 2):  
    
   
        # Validation and standardization of dimensions
        self.forecast = self._standardize_dims(forecast)
        self.observed = self._standardize_dims(observed)
        self._validate_inputs(self.forecast, self.observed)
        
        # Parameter configuration 
        self.threshold = threshold
        self.conv_radius_forecast = conv_radius_forecast  # WRF
        self.conv_radius_observed = conv_radius_observed  # GPM 
        self.time_window = time_window
        self.min_object_size = min_object_size
        

        self.min_object_size_forecast = min_object_size_forecast
        self.min_object_size_observed = min_object_size_observed
        
        # Convert times to pandas.Timestamp
        self.forecast['time'] = self.forecast.time.to_pandas()
        self.observed['time'] = self.observed.time.to_pandas()
        
        # Store original coordinates
        self.forecast_lat = self.forecast.lat.values
        self.forecast_lon = self.forecast.lon.values
        self.observed_lat = self.observed.lat.values
        self.observed_lon = self.observed.lon.values
        
        # Results
        self.forecast_objects = []
        self.observed_objects = []
        self.interest_matrix = None
        self.matches = []
        self.metrics = {}

    ##-------------------------------------------------------------------------
    ## 3.2._standardize_dims method
    ##-------------------------------------------------------------------------
    def _standardize_dims(self, data: xr.DataArray) -> xr.DataArray:
        """Standardizes dimension names to time, lat, and lon."""

        dim_map = {}
        for dim in data.dims:
            lower_dim = dim.lower()
            if 'time' in lower_dim:
                dim_map[dim] = 'time'
            elif 'lat' in lower_dim or 'y' in lower_dim:
                dim_map[dim] = 'lat'
            elif 'lon' in lower_dim or 'x' in lower_dim:
                dim_map[dim] = 'lon'
        
        return data.rename(dim_map)
    
    ##-------------------------------------------------------------------------
    ## 3.3._validate_inputs method 
    ##-------------------------------------------------------------------------
    def _validate_inputs(self, forecast: xr.DataArray, observed: xr.DataArray):
        """Validates input data using standardized dimensions."""

        required_dims = {'time', 'lat', 'lon'}
        
        if not isinstance(forecast, xr.DataArray) or not isinstance(observed, xr.DataArray):
            raise TypeError("The data must be xarray.DataArray.")
            
        if not required_dims.issubset(set(forecast.dims)) or not required_dims.issubset(set(observed.dims)):
            raise ValueError(f"The data must contain dimensions equivalent to {required_dims}.")

        if forecast.time.shape != observed.time.shape:
            raise ValueError("The forecasted and observed fields must have the same times.")

    def _safe_index(self, value: float, max_size: int) -> int:
        """Safely converts a value to an index."""
        return min(max(0, int(round(value))), max_size-1)

    ##-------------------------------------------------------------------------
    ## 3.4.run_verification method
    ##-------------------------------------------------------------------------
    def run_verification(self, interest_threshold: float = 0.5) -> Dict:
        """Execute the complete verification process."""

        print("Iniciando verificación MODE-py...")

        self._preprocess_data()
        self._identify_objects()
        self._match_objects(interest_threshold)
        self._calculate_metrics()

        print("Verification complete!")
        return self.metrics

    ##-------------------------------------------------------------------------
    ## 3.5._preprocess_data method
    ##-------------------------------------------------------------------------
    def _preprocess_data(self):
        """Preprocess the data: smoothing and thresholding with different radii."""

        # Gaussian smoothing for forecasting
        self.smoothed_fcst = xr.apply_ufunc(
            lambda x: gaussian_filter(x, sigma=self.conv_radius_forecast, mode='nearest'),
            self.forecast,
            input_core_dims=[['lat', 'lon']],
            output_core_dims=[['lat', 'lon']],
            vectorize=True
        )
        
        # Gaussian smoothing for observation
        self.smoothed_obs = xr.apply_ufunc(
            lambda x: gaussian_filter(x, sigma=self.conv_radius_observed, mode='nearest'),
            self.observed,
            input_core_dims=[['lat', 'lon']],
            output_core_dims=[['lat', 'lon']],
            vectorize=True
        )
      
        # Thresholding and cleaning
        self.binary_fcst = self.smoothed_fcst > self.threshold
        self.binary_obs = self.smoothed_obs > self.threshold
        
        # Morphological operation to remove small artifacts
        for t in range(len(self.forecast.time)):
            self.binary_fcst[t] = binary_closing(self.binary_fcst[t].values)
            self.binary_obs[t] = binary_closing(self.binary_obs[t].values)

    ##-------------------------------------------------------------------------
    ## 3.6._identify_objects method
    ##-------------------------------------------------------------------------
    def _identify_objects(self):
        """Identifies objects in the forecasted and observed data."""

        time_coords = self.forecast.time.to_pandas().values
        
        # Process forecasts
        self.forecast_objects = self._process_time_slices(
            self.binary_fcst, self.smoothed_fcst, 'forecast', time_coords
        )
        
        # Process observations
        self.observed_objects = self._process_time_slices(
            self.binary_obs, self.smoothed_obs, 'observed', time_coords
        )

    ##-------------------------------------------------------------------------
    ## 3.7._process_time_slices method
    ##-------------------------------------------------------------------------
    def _process_time_slices(self, binary_data: xr.DataArray, intensity_data: xr.DataArray, 
                           obj_type: str, time_coords: np.ndarray) -> List[Dict]:
        """Processes temporal slices to identify objects."""

        objects_3d = []
        
        # Define different thresholds based on resolution.
        if obj_type == 'forecast':
            min_size = self.min_object_size_forecast     
            resolution_info = "WRF"
        else:
            min_size = self.min_object_size_observed    
            resolution_info = "GPM"
        
        
        for t_idx, time_val in enumerate(time_coords):
            binary_slice = binary_data.isel(time=t_idx).values
            intensity_slice = intensity_data.isel(time=t_idx).values
            
            labels, n_objects = label(binary_slice)
            props = regionprops(labels, intensity_image=intensity_slice)
            
            objects_2d = []
            areas_detected = []
            
            for prop in props:
                areas_detected.append(prop.area)
                if prop.area >= min_size: 
                    obj_dict = self._create_object_dict(prop, obj_type, time_val, t_idx)
                    objects_2d.append(obj_dict)
            
            # Space fusion
            objects_2d = self._simple_spatial_merge(objects_2d, spatial_threshold=config.spatial_threshold)
            
            objects_3d.extend(objects_2d)
        
        return self._group_temporal_objects(objects_3d, time_coords)


    ##-------------------------------------------------------------------------
    ## 3.8._simple_spatial_merge method 
    ##-------------------------------------------------------------------------
    def _simple_spatial_merge(self, objects, spatial_threshold=config.spatial_threshold):
        """Spatial fusion combines all properties."""

        if not objects:
            return []
        
        merged = []
        used = set()
        
        for i, obj1 in enumerate(objects):
            if i in used:
                continue
                
            group = [obj1]
            used.add(i)
            
            # Recursive search to find all connected objects 
            changed = True
            while changed:
                changed = False
                for j, obj2 in enumerate(objects):
                    if j in used:
                        continue
                    
                    # Check if obj2 is close to ANY object in the group
                    for existing_obj in group:
                        dist = np.sqrt(
                            (existing_obj['centroid_geo'][0] - obj2['centroid_geo'][0])**2 + 
                            (existing_obj['centroid_geo'][1] - obj2['centroid_geo'][1])**2
                        ) * 111
                        
                        if dist <= spatial_threshold:
                            group.append(obj2)
                            used.add(j)
                            changed = True
                            break
            
            if len(group) > 1:
                # Complete merger of all properties
                merged_obj = self._create_merged_object(group)
                merged.append(merged_obj)
            else:
                merged.append(obj1)

        return merged


    ##-------------------------------------------------------------------------
    ## 3.9._create_merged_object method 
    ##-------------------------------------------------------------------------
    def _create_merged_object(self, objects):
        """Create a new object by combining all the properties."""
        
        # Combine coordinates
        all_coords_pixel = np.vstack([obj['coords_pixel'] for obj in objects])
        all_coords_geo = np.vstack([obj['coords_geo'] for obj in objects])
        
        # Calculate new centroid
        centroid_lat = np.mean([obj['centroid_geo'][0] for obj in objects])
        centroid_lon = np.mean([obj['centroid_geo'][1] for obj in objects])
        
        # Calculate a new bounding box that contains all objects
        all_rows = np.concatenate([obj['coords_pixel'][:, 0] for obj in objects])
        all_cols = np.concatenate([obj['coords_pixel'][:, 1] for obj in objects])
        
        min_row, max_row = np.min(all_rows), np.max(all_rows)
        min_col, max_col = np.min(all_cols), np.max(all_cols)
        
        # Create a completely new object
        merged_obj = {
            'type': objects[0]['type'],
            'time': objects[0]['time'],
            'time_idx': objects[0]['time_idx'],
            'centroid_pixel': (np.mean([obj['centroid_pixel'][0] for obj in objects]),
                              np.mean([obj['centroid_pixel'][1] for obj in objects])),
            'centroid_geo': (centroid_lat, centroid_lon),
            'area': sum(obj['area'] for obj in objects),  # Total area
            'orientation': np.mean([obj['orientation'] for obj in objects]),
            'intensity_mean': np.mean([obj['intensity_mean'] for obj in objects]),
            'intensity_max': max([obj['intensity_max'] for obj in objects]),
            'bbox': (min_row, min_col, max_row, max_col),  # Bounding box containing everything
            'coords_pixel': all_coords_pixel,  # All coordinates combined
            'coords_geo': all_coords_geo,      # All geo-coordinates combined
            'label': f"merged_{objects[0]['label']}",
            'eccentricity': np.mean([obj['eccentricity'] for obj in objects]),
            'resolution': objects[0]['resolution'],
            'was_merged': True,
            'merged_from': len(objects)  # Number of objects merged
        }
        
        return merged_obj


    ##-------------------------------------------------------------------------
    ## 3.10._create_object_dict method
    ##-------------------------------------------------------------------------
    def _create_object_dict(self, prop: regionprops, obj_type: str, 
                          time_val: np.datetime64, t_idx: int) -> Dict:

        """Create a dictionary from an object's properties."""
        # Determine which coordinates to use based on the object type
        if obj_type == 'forecast':
            lat_coords = self.forecast_lat
            lon_coords = self.forecast_lon
        else:
            lat_coords = self.observed_lat
            lon_coords = self.observed_lon
        
        # Convertir coordenadas de píxeles a coordenadas geográficas reales
        centroid_lat_idx = self._safe_index(prop.centroid[0], len(lat_coords))
        centroid_lon_idx = self._safe_index(prop.centroid[1], len(lon_coords))
        
        centroid_lat = lat_coords[centroid_lat_idx]
        centroid_lon = lon_coords[centroid_lon_idx]
        
        # Convert all object coordinates to geographic coordinates
        geo_coords = []
        for coord in prop.coords:
            lat_idx = self._safe_index(coord[0], len(lat_coords))
            lon_idx = self._safe_index(coord[1], len(lon_coords))
            geo_coords.append((lat_coords[lat_idx], lon_coords[lon_idx]))
        
        return {
            'type': obj_type,
            'time': pd.Timestamp(time_val),
            'time_idx': t_idx,
            'centroid_pixel': (prop.centroid[0], prop.centroid[1]),
            'centroid_geo': (centroid_lat, centroid_lon),
            'area': prop.area,
            'orientation': prop.orientation,
            'intensity_mean': prop.mean_intensity,
            'intensity_max': prop.max_intensity,
            'bbox': prop.bbox,
            'coords_pixel': prop.coords,
            'coords_geo': np.array(geo_coords),
            'label': prop.label,
            'eccentricity': prop.eccentricity,
            'resolution': '3km' if obj_type == 'forecast' else '10km'
        }

    ##-------------------------------------------------------------------------
    ## 3.11._group_temporal_objects method
    ##-------------------------------------------------------------------------
    def _group_temporal_objects(self, objects: List[Dict], time_coords: np.ndarray) -> List[Dict]:
        """Groups objects that persist over time."""

        if not objects:
            return []
            
        # Convert time_coords to pandas.Timestamp for consistent comparisons
        time_coords = [pd.Timestamp(t) for t in time_coords]
        
        # Calculate average time differences
        time_diffs = np.diff(time_coords)
        avg_time_diff = np.mean(time_diffs).total_seconds() / 3600 if len(time_diffs) > 0 else 1
        
        # Sort objects by time
        objects_sorted = sorted(objects, key=lambda x: x['time'])
        
        groups = []
        
        for obj in objects_sorted:
            matched = False
            
            for group in groups:
                last_obj = group[-1]
                
                # Calculate normalized time difference
                time_diff = (obj['time'] - last_obj['time']).total_seconds() / 3600 / avg_time_diff
                
                if time_diff <= self.time_window:
                    # Calculate spatial overlap based on geographic coordinates
                    current_centroid = obj['centroid_geo']
                    last_centroid = last_obj['centroid_geo']
                    
                    # Distance between centroids in km
                    distance_km = np.sqrt((current_centroid[0] - last_centroid[0])**2 + 
                                        (current_centroid[1] - last_centroid[1])**2) * 111
                    
                    # Distance threshold for considering the same object
                    min_dist_same_object = config.min_dist_same_object
                    if distance_km <= min_dist_same_object:
                        group.append(obj)
                        matched = True
                        break
            
            if not matched:
                groups.append([obj])
        
        # Convert groups into temporary objects
        temporal_objects = []
        
        for group in groups:
            times = [obj['time'] for obj in group]
            time_idxs = [obj['time_idx'] for obj in group]
            centroids_geo = [obj['centroid_geo'] for obj in group]
            centroids_pixel = [obj['centroid_pixel'] for obj in group]
            areas = [obj['area'] for obj in group]
            intensities = [obj['intensity_mean'] for obj in group]
            
            temporal_obj = {
                'type': group[0]['type'],
                'time_start': min(times),
                'time_end': max(times),
                'duration': len(group),
                'time_points': times,
                'time_indices': time_idxs,
                'centroid_mean_geo': np.mean(centroids_geo, axis=0),
                'centroid_mean_pixel': np.mean(centroids_pixel, axis=0),
                'centroid_trajectory_geo': centroids_geo,
                'centroid_trajectory_pixel': centroids_pixel,
                'area_mean': np.mean(areas),
                'area_max': max(areas),
                'intensity_mean': np.mean(intensities),
                'intensity_max': max(intensities),
                'objects_2d': group,
                'id': len(temporal_objects),
                'resolution': group[0]['resolution']
            }
            
            temporal_objects.append(temporal_obj)
        
        return temporal_objects

    ##-------------------------------------------------------------------------
    ## 3.12._match_objects method
    ##-------------------------------------------------------------------------
    def _match_objects(self, interest_threshold: float):
        """Match predicted objects with observed ones."""

        n_fcst = len(self.forecast_objects)
        n_obs = len(self.observed_objects)
        
        if n_fcst == 0 or n_obs == 0:
            print("No objects were found in one or both fields.")
            return
            
        # Initialize matrix of interest
        self.interest_matrix = np.zeros((n_fcst, n_obs))
        
        # Calculate interest for all pairs
        for i in tqdm(range(n_fcst), desc="Calculating interest"):
            for j in range(n_obs):
                self.interest_matrix[i, j] = self._calculate_interest(
                    self.forecast_objects[i], 
                    self.observed_objects[j]
                )
        
        # Greedy matching
        used_fcst = set()
        used_obs = set()
        
        # Sort possible matches by descending interest
        potential_matches = []
        for i in range(n_fcst):
            for j in range(n_obs):
                potential_matches.append((i, j, self.interest_matrix[i, j]))
        
        potential_matches.sort(key=lambda x: -x[2])
        
        for i, j, interest in potential_matches:
            if i not in used_fcst and j not in used_obs and interest >= interest_threshold:
                self.matches.append({
                    'forecast_id': self.forecast_objects[i]['id'],
                    'observed_id': self.observed_objects[j]['id'],
                    'forecast_idx': i,
                    'observed_idx': j,
                    'interest': interest,
                    'distance': self._calculate_centroid_distance(
                        self.forecast_objects[i], 
                        self.observed_objects[j]
                    ),
                    'time_overlap': self._calculate_time_overlap(
                        self.forecast_objects[i], 
                        self.observed_objects[j]
                    )
                })
                used_fcst.add(i)
                used_obs.add(j)

    ##-------------------------------------------------------------------------
    ## 3.13._calculate_interest method
    ##-------------------------------------------------------------------------
    def _calculate_interest(self, fcst_obj: Dict, obs_obj: Dict) -> float:
        """Calculate the interest between two objects using fuzzy logic."""

        weights = config.INTEREST_WEIGHTS
        
        # 1. Distance between centroids
        distance_km = self._calculate_centroid_distance(fcst_obj, obs_obj)
        max_distance = 500  
        interest_distance = np.exp(-distance_km / (0.2 * max_distance))
        
        # 2. Ratio of areas
        area_ratio = self._calculate_area_ratio(fcst_obj, obs_obj)
        
        # 3. Spatio-temporal overlap
        overlap = self._calculate_spatiotemporal_overlap(fcst_obj, obs_obj)
        
        # 4. Orientation
        orientation_diff = self._calculate_orientation_difference(fcst_obj, obs_obj)
        interest_orientation = 1 - (orientation_diff / 90) if orientation_diff is not None else 0.5
        
        # 5. Temporal coincidence
        temporal_match = self._calculate_time_overlap(fcst_obj, obs_obj)
        
        # Weighted total interest
        total_interest = (
            weights['distance'] * interest_distance +
            weights['area_ratio'] * area_ratio +
            weights['overlap'] * overlap +
            weights['orientation'] * interest_orientation +
            weights['temporal'] * temporal_match
        )
        
        return np.clip(total_interest, 0, 1)

    ##-------------------------------------------------------------------------
    ## 3.14._calculate_centroid_distance method
    ##-------------------------------------------------------------------------
    def _calculate_centroid_distance(self, obj1: Dict, obj2: Dict) -> float:
        """Calculate the Euclidean distance between centroids in geographic coordinates."""

        # Use geographic coordinates instead of pixel indices
        lat1, lon1 = obj1['centroid_mean_geo']
        lat2, lon2 = obj2['centroid_mean_geo']
        
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111  

    ##-------------------------------------------------------------------------
    ## 3.15._calculate_area_ratio method
    ##-------------------------------------------------------------------------
    def _calculate_area_ratio(self, fcst_obj: Dict, obs_obj: Dict) -> float:
        """Calculate the resolution-adjusted area ratio."""

        # Adjust areas for resolution difference
        # This needs to be adjusted!!!! 
        fcst_area = fcst_obj['area_mean'] * 9  # 3km grid cells 
        obs_area = obs_obj['area_mean'] * 100  # 10km grid cells 
        
        # Calculate the ratio of area 
        return min(fcst_area, obs_area) / max(fcst_area, obs_area)

    ##-------------------------------------------------------------------------
    ## 3.16._calculate_spatiotemporal_overlap method
    ##-------------------------------------------------------------------------
    def _calculate_spatiotemporal_overlap(self, fcst_obj: Dict, obs_obj: Dict) -> float:
        """Calculate the maximum spatial overlap at coincident times."""

        max_overlap = 0.0
        
        # Finding common times
        common_times = set(fcst_obj['time_points']).intersection(obs_obj['time_points'])
        
        for t in common_times:
            # Find corresponding 2D objects
            fcst_2d = next((o for o in fcst_obj['objects_2d'] if o['time'] == t), None)
            obs_2d = next((o for o in obs_obj['objects_2d'] if o['time'] == t), None)
            
            if fcst_2d and obs_2d:
                # Use geographic coordinates for the calculation of corresponding overlaps
                overlap = self._calculate_geographic_overlap(
                    fcst_2d['coords_geo'], 
                    obs_2d['coords_geo']
                )
                max_overlap = max(max_overlap, overlap)
        
        return max_overlap

    ##-------------------------------------------------------------------------
    ## 3.17._calculate_geographic_overlap method
    ##-------------------------------------------------------------------------
    def _calculate_geographic_overlap(self, coords1: np.ndarray, coords2: np.ndarray, 
                                   tolerance_km: float = 10.0) -> float:
        """Calculates the geographic overlap between two sets of coordinates."""

        if len(coords1) == 0 or len(coords2) == 0:
            return 0.0
        
        # Convert tolerance from km to degrees 
        tolerance_deg = tolerance_km / 111.0
        
        # Find points that are within the tolerance
        matching_points = 0
        for coord1 in coords1:
            for coord2 in coords2:
                distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                if distance <= tolerance_deg:
                    matching_points += 1
                    break
        
        # Calculate overlap 
        union_size = len(coords1) + len(coords2) - matching_points
        return matching_points / union_size if union_size > 0 else 0.0

    ##-------------------------------------------------------------------------
    ## 3.18._calculate_orientation_difference method
    ##-------------------------------------------------------------------------
    def _calculate_orientation_difference(self, fcst_obj: Dict, obs_obj: Dict) -> Optional[float]:
        """Calculates the average orientation difference in degrees."""

        orientation_diffs = []
        
        for t in set(fcst_obj['time_points']).intersection(obs_obj['time_points']):
            fcst_2d = next((o for o in fcst_obj['objects_2d'] if o['time'] == t), None)
            obs_2d = next((o for o in obs_obj['objects_2d'] if o['time'] == t), None)
            
            if fcst_2d and obs_2d and fcst_2d['eccentricity'] > 0.2 and obs_2d['eccentricity'] > 0.2:
                angle_diff = abs(fcst_2d['orientation'] - obs_2d['orientation'])
                angle_diff = min(angle_diff, 180 - angle_diff)  
                orientation_diffs.append(angle_diff)
        
        return np.mean(orientation_diffs) if orientation_diffs else None

    ##-------------------------------------------------------------------------
    ## 3.19._calculate_time_overlap method
    ##-------------------------------------------------------------------------
    def _calculate_time_overlap(self, obj1: Dict, obj2: Dict) -> float:
        """Calculate the temporal overlap fraction."""

        common_times = set(obj1['time_points']).intersection(obj2['time_points'])
        all_times = set(obj1['time_points']).union(obj2['time_points'])
        return len(common_times) / len(all_times) if all_times else 0.0

    ##-------------------------------------------------------------------------
    ## 3.20._calculate_metrics method
    ##-------------------------------------------------------------------------
    def _calculate_metrics(self):
        """Calcula métricas de evaluación."""

        # Median of Maximum Interest (MMI)
        if self.interest_matrix is not None and self.interest_matrix.size > 0:
            max_interest_fcst = np.max(self.interest_matrix, axis=1)
            max_interest_obs = np.max(self.interest_matrix, axis=0)
            
            self.metrics['MMI_forecast'] = float(np.median(max_interest_fcst))
            self.metrics['MMI_observed'] = float(np.median(max_interest_obs))
            self.metrics['MMI'] = float(np.median(np.concatenate([max_interest_fcst, max_interest_obs])))
        else:
            self.metrics.update({'MMI_forecast': 0.0, 'MMI_observed': 0.0, 'MMI': 0.0})
        
        # Components for  Object-Based Gilbert Skill Score 
        hits = len(self.matches)
        false_alarms = len(self.forecast_objects) - hits
        misses = len(self.observed_objects) - hits

        fcst_count = len(self.forecast_objects)
        obs_count = len(self.observed_objects)


        # Other metrics
        self.metrics['hits'] = hits
        self.metrics['false_alarms'] = false_alarms   
        self.metrics['misses'] = misses
        self.metrics['forecast_objects'] = len(self.forecast_objects)
        self.metrics['observed_objects'] = len(self.observed_objects)

        
        # Object statistics
        if self.forecast_objects:
            self.metrics['forecast_mean_duration'] = np.mean([obj['duration'] for obj in self.forecast_objects])
            self.metrics['forecast_mean_area'] = np.mean([obj['area_mean'] for obj in self.forecast_objects])
        else:
            self.metrics.update({'forecast_mean_duration': 0.0, 'forecast_mean_area': 0.0})
            
        if self.observed_objects:
            self.metrics['observed_mean_duration'] = np.mean([obj['duration'] for obj in self.observed_objects])
            self.metrics['observed_mean_area'] = np.mean([obj['area_mean'] for obj in self.observed_objects])
        else:
            self.metrics.update({'observed_mean_duration': 0.0, 'observed_mean_area': 0.0})



        ##-------------------------------------------------------------------------
        # Gilbert Skill Score 
        ##-------------------------------------------------------------------------

        # Interpolate the forecast to match the spatial grid of the observation
        forecast_interp = self.forecast.interp_like(self.observed, method="nearest")

        # Obtain the binary masks using the new interpolated matrix
        fcst_mask = forecast_interp.values > self.threshold
        obs_mask = self.observed.values > self.threshold

        # Calculate the components of the contingency matrix
        N_pixels = self.observed.sizes['lat'] * self.observed.sizes['lon'] * self.observed.sizes['time']

        # Hits, False Alarms, Misses, and Correct Negatives    
        hits_classic = np.sum(fcst_mask & obs_mask)
        false_alarms_classic = np.sum(fcst_mask & ~obs_mask) 
        misses_classic = np.sum(~fcst_mask & obs_mask)
        correct_negatives_classic = np.sum(~fcst_mask & ~obs_mask)

        # Calculate the number of correct answers expected by chance
        fcst_total_pixels = hits_classic + false_alarms_classic
        obs_total_pixels = hits_classic + misses_classic

        expected_random_hits_classic = (fcst_total_pixels * obs_total_pixels) / N_pixels if N_pixels > 0 else 0

        # Calculate the final Classic GSS
        denominator_classic = hits_classic + false_alarms_classic + misses_classic - expected_random_hits_classic

        if denominator_classic > 0:
            self.metrics['GSS'] = (hits_classic - expected_random_hits_classic) / denominator_classic
        else:
            self.metrics['GSS'] = 0.0

            
        ##-------------------------------------------------------------------------
        # Object-Based Gilbert Skill Score 
        ##-------------------------------------------------------------------------

        # Calculate the total area of ​​the physical domain for each grid
        total_area_fcst = self.forecast.sizes['lat'] * self.forecast.sizes['lon']
        total_area_obs = self.observed.sizes['lat'] * self.observed.sizes['lon']

        # Calculate the areas occupied by objects at their respective resolutions
        obs_area = sum(obj['area_mean'] for obj in self.observed_objects)
        fcst_area = sum(obj['area_mean'] for obj in self.forecast_objects)

        # Calculate area fractions
        obs_area_fraction = obs_area / total_area_obs #if total_area_obs > 0 else 0
        fcst_area_fraction = fcst_area / total_area_fcst #if total_area_fcst > 0 else 0

        # Estimate Correct Negative using the observation fraction.
        D = (((1 - obs_area_fraction) * obs_count) / obs_area_fraction) - fcst_count

        # Calculate epsilon (expected correct matches by chance in the object space)
        denominador_epsilon = fcst_count + misses + D

        if denominador_epsilon > 0:
            epsilon = (fcst_count * obs_count) / denominador_epsilon
            epsilon = min(epsilon, float(hits))  
        else:
            epsilon = 0.0

        # Calculate the Final Object-Based GSS
        denominator_gss = fcst_count + misses - epsilon

        if denominator_gss > 0:
            self.metrics['GSS_Obj-Based'] = (hits - epsilon) / denominator_gss
        else:
            # If the denominator is zero but the model did a perfect job
            if fcst_count == 0 and obs_count == 0:
                self.metrics['GSS_Obj-Based'] = 1.0  
            else:
                self.metrics['GSS_Obj-Based'] = 0.0


        self.metrics['hits_random_classic'] = expected_random_hits_classic
        self.metrics['hits_random_objects'] = epsilon


    
    ##-------------------------------------------------------------------------
    ## 3.21.plot_matched_objects method
    ##-------------------------------------------------------------------------
    def plot_matched_objects(self, time_idx):    
        """
        Displays paired objects as contours overlaid on a precipitation field.
        """

        if not self.matches:
            print("No matches to display")
            return
        
        # Get the specific time
        target_time = self.forecast.time.isel(time=time_idx).values
        
        # Get precipitation fields for this time
        precip_forecast = self.forecast.isel(time=time_idx)
        precip_observed = self.observed.isel(time=time_idx)
        
        # Filter objects for this time
        fcst_objs = [obj for obj in self.forecast_objects 
                    if target_time in obj['time_points']]
        obs_objs = [obj for obj in self.observed_objects 
                   if target_time in obj['time_points']]
        
        if not fcst_objs and not obs_objs:
            print(f"No hay objetos para el tiempo {target_time}")
            return
        
        # Get all matches for this time
        current_matches = []
        for match in self.matches:
            fcst_obj = next((o for o in fcst_objs if o['id'] == match['forecast_id']), None)
            obs_obj = next((o for o in obs_objs if o['id'] == match['observed_id']), None)
            
            if fcst_obj and obs_obj:
                current_matches.append({
                    'forecast': fcst_obj,
                    'observed': obs_obj,
                    'interest': match['interest']
                })
        
        # Configure figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': ccrs.PlateCarree()}, sharex=True, sharey=True) #figsize=(16, 12) figsize=(16, 8)
        
        fig.suptitle(f"Objects Comparison  MODE-py - {pd.Timestamp(target_time).strftime('%Y-%m-%d %H:%M')}\n"
                     f"Threshold: {self.threshold} mm, Convolution Radius WRF: {self.conv_radius_forecast} pixels Convolution Radius GPM: {self.conv_radius_observed} pixels\n"
                     f"Min Dist Object {config.min_dist_same_object}km", 
                     fontsize=16, y=0.8) #y=0.9

        
        # Plot 1: Observed objects
        ax1.set_title("Observed (GPM)")
        
        # Plot the precipitation field
        contour_obs = ax1.contourf(precip_observed['lon'], 
                                   precip_observed['lat'], 
                                   precip_observed,
                                   transform=ccrs.PlateCarree(),
                                   levels=np.arange(0, 50, 2),
                                   cmap=field_visualization.cmap,
                                   extend='both',
                                   alpha=0.8)
        
        # Draw the outlines of the objects
        self._draw_objects_as_contours(ax1, obs_objs, current_matches, 'observed', 
                                      precip_min=0, precip_max=50)
        
        # Plot 2: Predicted objects     
        ax2.set_title("Forecast (WRF)")
        
        # Plot the precipitation field
        contour_fcst = ax2.contourf(precip_forecast['lon'], 
                                    precip_forecast['lat'], 
                                    precip_forecast,
                                    transform=ccrs.PlateCarree(),
                                    levels=np.arange(0, 50, 2),
                                    cmap=field_visualization.cmap,
                                    extend='both',
                                    alpha=0.8)
        
        # Draw the outlines of the objects
        self._draw_objects_as_contours(ax2, fcst_objs, current_matches, 'forecast', 
                                      precip_min=0, precip_max=50)
        
        # Common configuration for both subplots
        for ax in [ax1, ax2]:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
            ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.1)
            
            gl = ax.gridlines(
                draw_labels=True,
                linewidth=0.5, 
                color='gray', 
                alpha=0, 
                linestyle='--'
            )
            gl.top_labels = False
            gl.right_labels = False
            
            ax.set_xlabel('Longitud')
            ax.set_ylabel('Latitud')
        
        # Adjust the limits so that both plots are comparable
        min_lon = min(self.forecast.lon.min(), self.observed.lon.min())
        max_lon = max(self.forecast.lon.max(), self.observed.lon.max())
        min_lat = min(self.forecast.lat.min(), self.observed.lat.min())
        max_lat = max(self.forecast.lat.max(), self.observed.lat.max())
        
        for ax in [ax1, ax2]:
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
        # Add a common color bar for precipitation
        cbar_ax = fig.add_axes([0.08, 0.08, 0.8, 0.025])  # [left, bottom, width, height]
        cbar = plt.colorbar(contour_fcst, 
                            cax=cbar_ax, 
                            orientation='horizontal',
                            pad=0.5, 
                            shrink=0.8, 
                            aspect=60)
        cbar.set_label('Precipitation (mm/h)', fontsize=12)
        
    
        # Add match legend at the top
        if current_matches:
            self._add_match_legend(fig, current_matches)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(config.path_figures, 
                               f'MODE-py_contours_{pd.Timestamp(target_time).strftime("%Y%m%d_%H%M")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    
    ##-------------------------------------------------------------------------
    ## 3.22._draw_objects_as_contours method 
    ##-------------------------------------------------------------------------    
    def _draw_objects_as_contours(self, ax, objects, matches, obj_type, precip_min=0, precip_max=50):
        """
        Draws objects as outlines over the precipitation field.
        
        Differentiation:
        - MATCHED objects: Thick solid line with ID in a white circle
        - UNMATCHED objects: Thin dotted line with ID in a gray circle
        """

        # Identify IDs of paired objects
        matched_ids = [m[obj_type]['id'] for m in matches]
        
        # Colors for paired objects 
        match_colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF',  
            '#FFA500', '#800080', '#008080', '#FF1493', '#32CD32'   
        ]
        
        # Process each object
        for i, obj in enumerate(objects):
            is_matched = obj['id'] in matched_ids
            
            # Determine properties based on whether it is paired or not
            if is_matched:
                # Find the corresponding match
                match_idx = matched_ids.index(obj['id'])
                line_color = match_colors[match_idx % len(match_colors)]
                line_width = 1.5
                line_style = 'solid'
                id_bg_color = 'white'
                id_text_color = 'black'
                line_label = f'Match {match_idx+1}'
            else:
                line_color = 'gray'
                line_width = 1.5
                line_style = 'dotted'
                id_bg_color = 'lightgray'
                id_text_color = 'darkgray'
                line_label = 'No match'
            
            # Draw object outline
            self._draw_object_contour(ax, obj, line_color, line_width, line_style)
            
            # Draw centroid with ID
            self._draw_object_centroid(ax, obj, line_color, id_bg_color, id_text_color, is_matched)
            
            # If paired, draw a line connecting to the paired object
            if is_matched and obj_type == 'forecast':
                self._draw_connection_line(ax, obj, matches[match_idx], line_color)
        

    ##-------------------------------------------------------------------------
    ## 3.23._draw_object_contour method 
    ##------------------------------------------------------------------------- 
    def _draw_object_contour(self, ax, obj, line_color, line_width, line_style):
        """Draw the outline of an object as a line."""

        # Get the object's coordinates for this time
        obj_2d = next((o for o in obj['objects_2d'] 
                      if o['time'] == pd.Timestamp(obj['time_points'][0])), None)
        
        if obj_2d and 'coords_geo' in obj_2d and len(obj_2d['coords_geo']) > 0:

            # Use geographic coordinates directly
            lons = obj_2d['coords_geo'][:, 1]
            lats = obj_2d['coords_geo'][:, 0]
            
            # Create a convex hull for the smooth outline
            if len(lons) >= 3:
                from scipy.spatial import ConvexHull
                points = np.column_stack((lons, lats))
                
                try:
                    hull = ConvexHull(points)
                    
                    # Draw the convex polygon as a line
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])  
                    
                    ax.plot(hull_points[:, 0], hull_points[:, 1], 
                           color=line_color, linewidth=line_width, 
                           linestyle=line_style, alpha=0.9,
                           transform=ccrs.PlateCarree())
                    
                    # Fill to highlight the area
                    ax.fill(hull_points[:, 0], hull_points[:, 1], 
                           color=line_color, alpha=0.05,  
                           transform=ccrs.PlateCarree())
                    
                except Exception as e:
                    # If the convex hull fails, draw simple points to highlight the area
                    print(f" Error in convex hull for object {obj['id']}: {e}")
                    ax.plot(lons, lats, 'o', color=line_color, markersize=2,
                           transform=ccrs.PlateCarree(), alpha=0.7)
    
    ##-------------------------------------------------------------------------
    ## 3.24._draw_object_centroid method 
    ##-------------------------------------------------------------------------
    def _draw_object_centroid(self, ax, obj, line_color, bg_color, text_color, is_matched):
        """Draw the object's centroid with its ID."""

        centroid_lon = np.mean([c[1] for c in obj['centroid_trajectory_geo']])
        centroid_lat = np.mean([c[0] for c in obj['centroid_trajectory_geo']])
        
        # Draw centroid marker
        marker_size = 100 if is_matched else 80
        marker_edge_width = 2 if is_matched else 1
        
        ax.scatter(centroid_lon, centroid_lat, 
                  color=line_color, s=marker_size,
                  edgecolors=bg_color, linewidth=marker_edge_width,
                  zorder=10, alpha=0.9,
                  transform=ccrs.PlateCarree())
      
    ##-------------------------------------------------------------------------
    ## 3.25._draw_connection_line method 
    ##-------------------------------------------------------------------------
    def _draw_connection_line(self, ax, fcst_obj, match, line_color):
        """Draw a line connecting paired objects."""

        paired_obj = match['observed']
        
        fcst_centroid_lon = np.mean([c[1] for c in fcst_obj['centroid_trajectory_geo']])
        fcst_centroid_lat = np.mean([c[0] for c in fcst_obj['centroid_trajectory_geo']])
        
        obs_centroid_lon = np.mean([c[1] for c in paired_obj['centroid_trajectory_geo']])
        obs_centroid_lat = np.mean([c[0] for c in paired_obj['centroid_trajectory_geo']])
        
        # Draw a dotted line between centroids
        ax.plot([fcst_centroid_lon, obs_centroid_lon], 
               [fcst_centroid_lat, obs_centroid_lat], 
               color=line_color, linestyle=':', alpha=0.7, 
               linewidth=1.5, transform=ccrs.PlateCarree())
        
    ##-------------------------------------------------------------------------
    ## 3.26._add_match_legend method 
    ##-------------------------------------------------------------------------
    def _add_match_legend(self, fig, current_matches):
        """Add a custom caption for matches."""

        legend_elements = [
            Line2D([0], [0], marker='o', color='black', 
                   markerfacecolor='white', markersize=10,
                   #markeredgecolor='red', markeredgewidth=2,
                   label=f'GSS Global: {self.metrics.get("GSS", 0):.2f}'),
            
            Line2D([0], [0], marker='o', color='black', 
                   markerfacecolor='white', markersize=10,
                   #markeredgecolor='blue', markeredgewidth=2,
                   label=f'GSS_Obj-Based: {self.metrics.get("GSS_Obj-Based", 0):.3f}')
        ]
        
        # Add specific elements for each match
        match_colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF',
            '#FFA500', '#800080', '#008080', '#FF1493', '#32CD32'
        ]
        
        for i, match in enumerate(current_matches):
            if i < len(match_colors):  
                legend_elements.append(
                    Line2D([0], [0], color=match_colors[i], linewidth=3, linestyle='solid',
                           label=f'Match {i+1}: Int={match["interest"]:.2f}')
                )
        
        # Create a legend at the bottom
        fig.legend(handles=legend_elements, 
                   loc='lower center', 
                   ncol=min(5, len(current_matches) + 2), 
                   bbox_to_anchor=(0.5, -0.05),
                   fontsize=16)
        
    
    ##-------------------------------------------------------------------------
    ## 3.27.save_metrics_to_csv method
    ##-------------------------------------------------------------------------     
    def save_metrics_to_csv(self, csv_path=None, append=False, config_params=None):
        """
        Saves all MODE metrics to a CSV file in long format.

        Args:
            -csv_path: Path to the CSV file (if None, uses the default path)
            -append: If True, appends to the existing file instead of overwriting
            -config_params: Dictionary containing the configuration parameters used
        """
        csv_path=os.path.join(config.path_statistics, f"MODE-py_results T={self.threshold} RO={self.conv_radius_observed} RF={self.conv_radius_forecast} .csv")

        if not self.metrics:
            print("There are no metrics to save. Run run_verification() first.")
            return
        
        # Define default path if not provided
        if csv_path is None:
            csv_path = os.path.join(config.path_statistics, "MODE-py_metrics.csv")
        
        # Create directory if it does not exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Prepare data for CSV in long format
        metrics_data = self.metrics.copy()
        
        # Add metadata and configuration parameters
        metrics_data['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics_data['forecast_resolution'] = '3km'
        metrics_data['observed_resolution'] = '10km'
        
        # Add configuration parameters if provided
        if config_params:
            metrics_data.update(config_params)
        else:
            # Add default instance parameters
            metrics_data.update({
                'threshold': self.threshold,
                'conv_radius_forecast': self.conv_radius_forecast,
                'conv_radius_observed': self.conv_radius_observed,
                'time_window': self.time_window,
                'min_object_size': self.min_object_size,
                'interest_threshold': getattr(self, 'interest_threshold', 0.6)
            })
        
        # Add dimension information
        metrics_data['grid_lat'] = self.forecast.sizes['lat']
        metrics_data['grid_lon'] = self.forecast.sizes['lon']
        metrics_data['time_steps'] = self.forecast.sizes['time']
        
        # Convert to long format
        df_metrics_long = pd.DataFrame(list(metrics_data.items()), columns=['metric', 'value'])
        
        # Add a timestamp as an additional column to identify each execution
        timestamp = metrics_data['timestamp']
        df_metrics_long['timestamp'] = timestamp
        
        # Reorder columns
        df_metrics_long = df_metrics_long[['timestamp', 'metric', 'value']]
        
        # Save as csv
        if append and os.path.exists(csv_path):
                   df_metrics_long.to_csv(csv_path, mode='a', header=False, index=False, float_format='%.4f')
                        
        else:
            # Write mode (overwrite or create new)
            df_metrics_long.to_csv(csv_path, index=False, float_format='%.4f')
            print(f"Metrics saved in: {csv_path}")

        return csv_path


