##-----------------------------------------------------------------------------
## Clase MODE3DVerifier 
##-----------------------------------------------------------------------------

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
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
## 3. Implementacion de MODE (Method for Object-based Diagnostic Evaluation) 
##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------
## 3.1. Definicion de la clase: Class MODE3DVerifier 
##-----------------------------------------------------------------------------
class MODE3DVerifier:
    """
    Implementación del método MODE para datos (Time, Lat, Lon)
    manejando diferentes resoluciones espaciales.
    """
    ##-------------------------------------------------------------------------
    ## 3.2. Metodo __init__
    ##-------------------------------------------------------------------------
     
    def __init__(self, forecast: xr.DataArray, observed: xr.DataArray, 
             threshold: float = 1.5, conv_radius_forecast: int = 5, 
             conv_radius_observed: int = 3, time_window: int = 1, 
             min_object_size: int = 5,  # Mantener para compatibilidad
             min_object_size_forecast: int = 10,  # Específico para WRF
             min_object_size_observed: int = 2):  # Específico para GPM
    
   
        # Validación y estandarización de dimensiones
        self.forecast = self._standardize_dims(forecast)
        self.observed = self._standardize_dims(observed)
        self._validate_inputs(self.forecast, self.observed)
        
        # Configuración de parámetros (diferentes radios para diferentes resoluciones)
        self.threshold = threshold
        self.conv_radius_forecast = conv_radius_forecast  # Para WRF (3km)
        self.conv_radius_observed = conv_radius_observed  # Para GPM (10km)
        self.time_window = time_window
        self.min_object_size = min_object_size
        

        self.min_object_size_forecast = min_object_size_forecast
        self.min_object_size_observed = min_object_size_observed
        
        # Convertir tiempos a pandas.Timestamp
        self.forecast['time'] = self.forecast.time.to_pandas()
        self.observed['time'] = self.observed.time.to_pandas()
        
        # Almacenar coordenadas originales
        self.forecast_lat = self.forecast.lat.values
        self.forecast_lon = self.forecast.lon.values
        self.observed_lat = self.observed.lat.values
        self.observed_lon = self.observed.lon.values
        
        # Resultados
        self.forecast_objects = []
        self.observed_objects = []
        self.interest_matrix = None
        self.matches = []
        self.metrics = {}

    ##-------------------------------------------------------------------------
    ## 3.3. Metodo _standardize_dims
    ##-------------------------------------------------------------------------
    def _standardize_dims(self, data: xr.DataArray) -> xr.DataArray:
        """Estandariza los nombres de dimensiones a time, lat, lon."""
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
    ## 3.4. Metodo _validate_inputs
    ##-------------------------------------------------------------------------
    def _validate_inputs(self, forecast: xr.DataArray, observed: xr.DataArray):
        """Valida los datos de entrada con dimensiones estandarizadas."""
        required_dims = {'time', 'lat', 'lon'}
        
        if not isinstance(forecast, xr.DataArray) or not isinstance(observed, xr.DataArray):
            raise TypeError("Los datos deben ser xarray.DataArray")
            
        if not required_dims.issubset(set(forecast.dims)) or not required_dims.issubset(set(observed.dims)):
            raise ValueError(f"Los datos deben contener dimensiones equivalentes a {required_dims}")
            
        # NO requerimos shapes iguales para permitir diferentes resoluciones
        if forecast.time.shape != observed.time.shape:
            raise ValueError("Los campos pronosticados y observados deben tener los mismos tiempos")

    def _safe_index(self, value: float, max_size: int) -> int:
        """Convierte un valor a índice de forma segura."""
        return min(max(0, int(round(value))), max_size-1)

    ##-------------------------------------------------------------------------
    ## 3.5. Metodo run_verification
    ##-------------------------------------------------------------------------
    def run_verification(self, interest_threshold: float = 0.5) -> Dict:
        """Ejecuta el proceso completo de verificación."""
        print("Iniciando verificación MODE 3D...")
        self._preprocess_data()
        self._identify_objects()
        self._match_objects(interest_threshold)
        self._calculate_metrics()
        print("Verificación completada!")
        return self.metrics

    ##-------------------------------------------------------------------------
    ## 3.6. Metodo _preprocess_data
    ##-------------------------------------------------------------------------
    def _preprocess_data(self):
        """Preprocesa los datos: suavizado y umbralización con diferentes radios."""
        # Suavizado gaussiano para pronóstico (WRF - 3km)
        self.smoothed_fcst = xr.apply_ufunc(
            lambda x: gaussian_filter(x, sigma=self.conv_radius_forecast, mode='nearest'),
            self.forecast,
            input_core_dims=[['lat', 'lon']],
            output_core_dims=[['lat', 'lon']],
            vectorize=True
        )
        
        # Suavizado gaussiano para observación (GPM - 10km)
        self.smoothed_obs = xr.apply_ufunc(
            lambda x: gaussian_filter(x, sigma=self.conv_radius_observed, mode='nearest'),
            self.observed,
            input_core_dims=[['lat', 'lon']],
            output_core_dims=[['lat', 'lon']],
            vectorize=True
        )
      
        # Umbralización y limpieza
        self.binary_fcst = self.smoothed_fcst > self.threshold
        self.binary_obs = self.smoothed_obs > self.threshold
        
        # Operación morfológica para eliminar artefactos pequeños
        for t in range(len(self.forecast.time)):
            self.binary_fcst[t] = binary_closing(self.binary_fcst[t].values)
            self.binary_obs[t] = binary_closing(self.binary_obs[t].values)

    ##-------------------------------------------------------------------------
    ## 3.7. Metodo _identify_objects
    ##-------------------------------------------------------------------------
    def _identify_objects(self):
        """Identifica objetos en los datos pronosticados y observados."""
        time_coords = self.forecast.time.to_pandas().values
        
        # Procesar pronósticos
        self.forecast_objects = self._process_time_slices(
            self.binary_fcst, self.smoothed_fcst, 'forecast', time_coords
        )
        
        # Procesar observaciones
        self.observed_objects = self._process_time_slices(
            self.binary_obs, self.smoothed_obs, 'observed', time_coords
        )

    ##-------------------------------------------------------------------------
    ## 3.8. Metodo _process_time_slices
    ##-------------------------------------------------------------------------
    def _process_time_slices(self, binary_data: xr.DataArray, intensity_data: xr.DataArray, 
                           obj_type: str, time_coords: np.ndarray) -> List[Dict]:
        """Procesa slices temporales para identificar objetos con umbrales diferentes."""
        objects_3d = []
        
        # DEFINIR UMBRALES DIFERENTES SEGÚN RESOLUCIÓN
        if obj_type == 'forecast':
            min_size = self.min_object_size_forecast     
            resolution_info = "WRF (3km)"
        else:
            min_size = self.min_object_size_observed    
            resolution_info = "GPM (10km)"
        
        
        for t_idx, time_val in enumerate(time_coords):
            binary_slice = binary_data.isel(time=t_idx).values
            intensity_slice = intensity_data.isel(time=t_idx).values
            
            labels, n_objects = label(binary_slice)
            props = regionprops(labels, intensity_image=intensity_slice)
            
            objects_2d = []
            areas_detected = []
            
            for prop in props:
                areas_detected.append(prop.area)
                if prop.area >= min_size:  # FILTRO CON UMBRAL ESPECÍFICO
                    obj_dict = self._create_object_dict(prop, obj_type, time_val, t_idx)
                    objects_2d.append(obj_dict)
            
            # Fusión espacial
            objects_2d = self._simple_spatial_merge(objects_2d, spatial_threshold=config.spatial_threshold)
            
            objects_3d.extend(objects_2d)
        
        return self._group_temporal_objects(objects_3d, time_coords)


    ##-------------------------------------------------------------------------
    ## 3.8.1 Metodo _simple_spatial_merge 
    ##-------------------------------------------------------------------------
    def _simple_spatial_merge(self, objects, spatial_threshold=config.spatial_threshold):
        """Fusión espacial COMPLETA - combina todas las propiedades."""
        if not objects:
            return []
        
        merged = []
        used = set()
        
        for i, obj1 in enumerate(objects):
            if i in used:
                continue
                
            group = [obj1]
            used.add(i)
            
            # Búsqueda recursiva para encontrar todos los objetos conectados
            changed = True
            while changed:
                changed = False
                for j, obj2 in enumerate(objects):
                    if j in used:
                        continue
                    
                    # Verificar si obj2 está cerca de CUALQUIER objeto del grupo
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
                # FUSIÓN COMPLETA de todas las propiedades
                merged_obj = self._create_merged_object(group)
                merged.append(merged_obj)
            else:
                merged.append(obj1)

        return merged


    ##-------------------------------------------------------------------------
    ## 3.8.2 Metodo _create_merged_object 
    ##-------------------------------------------------------------------------
    def _create_merged_object(self, objects):
        """Crea un nuevo objeto combinando TODAS las propiedades."""
        # COMBINAR COORDENADAS 
        all_coords_pixel = np.vstack([obj['coords_pixel'] for obj in objects])
        all_coords_geo = np.vstack([obj['coords_geo'] for obj in objects])
        
        # Calcular nuevo centroide
        centroid_lat = np.mean([obj['centroid_geo'][0] for obj in objects])
        centroid_lon = np.mean([obj['centroid_geo'][1] for obj in objects])
        
        # Calcular nuevo bounding box que contenga TODOS los objetos
        all_rows = np.concatenate([obj['coords_pixel'][:, 0] for obj in objects])
        all_cols = np.concatenate([obj['coords_pixel'][:, 1] for obj in objects])
        
        min_row, max_row = np.min(all_rows), np.max(all_rows)
        min_col, max_col = np.min(all_cols), np.max(all_cols)
        
        # Crear objeto completamente nuevo
        merged_obj = {
            'type': objects[0]['type'],
            'time': objects[0]['time'],
            'time_idx': objects[0]['time_idx'],
            'centroid_pixel': (np.mean([obj['centroid_pixel'][0] for obj in objects]),
                              np.mean([obj['centroid_pixel'][1] for obj in objects])),
            'centroid_geo': (centroid_lat, centroid_lon),
            'area': sum(obj['area'] for obj in objects),  # Área total
            'orientation': np.mean([obj['orientation'] for obj in objects]),
            'intensity_mean': np.mean([obj['intensity_mean'] for obj in objects]),
            'intensity_max': max([obj['intensity_max'] for obj in objects]),
            'bbox': (min_row, min_col, max_row, max_col),  # BBox que contiene todo
            'coords_pixel': all_coords_pixel,  # TODAS las coordenadas combinadas
            'coords_geo': all_coords_geo,      # TODAS las coordenadas geo combinadas
            'label': f"merged_{objects[0]['label']}",
            'eccentricity': np.mean([obj['eccentricity'] for obj in objects]),
            'resolution': objects[0]['resolution'],
            'was_merged': True,
            'merged_from': len(objects)  # Cuántos objetos se fusionaron
        }
        
        return merged_obj


    ##-------------------------------------------------------------------------
    ## 3.9. Metodo _create_object_dict
    ##-------------------------------------------------------------------------
    def _create_object_dict(self, prop: regionprops, obj_type: str, 
                          time_val: np.datetime64, t_idx: int) -> Dict:
        """Crea un diccionario con las propiedades de un objeto."""
        # Determinar qué coordenadas usar según el tipo de objeto
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
        
        # Convertir todas las coordenadas del objeto a coordenadas geográficas
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
    ## 3.10. Metodo _group_temporal_objects
    ##-------------------------------------------------------------------------
    def _group_temporal_objects(self, objects: List[Dict], time_coords: np.ndarray) -> List[Dict]:
        """Agrupa objetos que persisten en el tiempo."""
        if not objects:
            return []
            
        # Convertir time_coords to pandas.Timestamp para comparaciones consistentes
        time_coords = [pd.Timestamp(t) for t in time_coords]
        
        # Calcular diferencias temporales promedio
        time_diffs = np.diff(time_coords)
        avg_time_diff = np.mean(time_diffs).total_seconds() / 3600 if len(time_diffs) > 0 else 1
        
        # Ordenar objetos por tiempo
        objects_sorted = sorted(objects, key=lambda x: x['time'])
        
        groups = []
        
        for obj in objects_sorted:
            matched = False
            
            for group in groups:
                last_obj = group[-1]
                
                # Calcular diferencia de tiempo normalizada
                time_diff = (obj['time'] - last_obj['time']).total_seconds() / 3600 / avg_time_diff
                
                if time_diff <= self.time_window:
                    # Calcular superposición espacial basada en coordenadas geográficas
                    current_centroid = obj['centroid_geo']
                    last_centroid = last_obj['centroid_geo']
                    
                    # Distancia entre centroides en km
                    distance_km = np.sqrt((current_centroid[0] - last_centroid[0])**2 + 
                                        (current_centroid[1] - last_centroid[1])**2) * 111
                    
                    # Umbral de distancia para considerar el mismo objeto (20 km)
                    min_dist_same_object = config.min_dist_same_object
                    if distance_km <= min_dist_same_object:
                        group.append(obj)
                        matched = True
                        break
            
            if not matched:
                groups.append([obj])
        
        # Convertir grupos en objetos temporales
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
    ## 3.11. Metodo _match_objects
    ##-------------------------------------------------------------------------
    def _match_objects(self, interest_threshold: float):
        """Empareja objetos pronosticados con observados."""
        n_fcst = len(self.forecast_objects)
        n_obs = len(self.observed_objects)
        
        if n_fcst == 0 or n_obs == 0:
            print("Advertencia: No se encontraron objetos en uno o ambos campos")
            return
            
        # Inicializar matriz de interés
        self.interest_matrix = np.zeros((n_fcst, n_obs))
        
        # Calcular interés para todos los pares
        for i in tqdm(range(n_fcst), desc="Calculando interés"):
            for j in range(n_obs):
                self.interest_matrix[i, j] = self._calculate_interest(
                    self.forecast_objects[i], 
                    self.observed_objects[j]
                )
        
        # Emparejamiento greedy
        used_fcst = set()
        used_obs = set()
        
        # Ordenar posibles emparejamientos por interés descendente
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
    ## 3.12. Metodo _calculate_interest
    ##-------------------------------------------------------------------------
    def _calculate_interest(self, fcst_obj: Dict, obs_obj: Dict) -> float:
        """Calcula el interés entre dos objetos usando lógica difusa."""
        weights = {
            'distance': 0.25,
            'area_ratio': 0.2,
            'overlap': 0.25,
            'orientation': 0.1,
            'temporal': 0.2
        }
        
        # 1. Distancia entre centroides (en km)
        distance_km = self._calculate_centroid_distance(fcst_obj, obs_obj)
        max_distance = 500  # Distancia máxima razonable en km
        interest_distance = np.exp(-distance_km / (0.2 * max_distance))
        
        # 2. Razón de áreas (ajustada por resolución)
        area_ratio = self._calculate_area_ratio(fcst_obj, obs_obj)
        
        # 3. Solapamiento espaciotemporal
        overlap = self._calculate_spatiotemporal_overlap(fcst_obj, obs_obj)
        
        # 4. Orientación (solo para objetos no circulares)
        orientation_diff = self._calculate_orientation_difference(fcst_obj, obs_obj)
        interest_orientation = 1 - (orientation_diff / 90) if orientation_diff is not None else 0.5
        
        # 5. Coincidencia temporal
        temporal_match = self._calculate_time_overlap(fcst_obj, obs_obj)
        
        # Interés total ponderado
        total_interest = (
            weights['distance'] * interest_distance +
            weights['area_ratio'] * area_ratio +
            weights['overlap'] * overlap +
            weights['orientation'] * interest_orientation +
            weights['temporal'] * temporal_match
        )
        
        return np.clip(total_interest, 0, 1)

    ##-------------------------------------------------------------------------
    ## 3.13. Metodo _calculate_centroid_distance
    ##-------------------------------------------------------------------------
    def _calculate_centroid_distance(self, obj1: Dict, obj2: Dict) -> float:
        """Calcula distancia euclidiana entre centroides en coordenadas geográficas (km)."""
        # Usar coordenadas geográficas en lugar de índices de píxeles
        lat1, lon1 = obj1['centroid_mean_geo']
        lat2, lon2 = obj2['centroid_mean_geo']
        
        # Distancia aproximada en km (simplificada para pequeñas distancias)
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111  # 1 grado ≈ 111 km

    ##-------------------------------------------------------------------------
    ## 3.14. Metodo _calculate_area_ratio
    ##-------------------------------------------------------------------------
    def _calculate_area_ratio(self, fcst_obj: Dict, obs_obj: Dict) -> float:
        """Calcula la razón de áreas ajustada por resolución."""
        # Ajustar áreas por diferencia de resolución (3km vs 10km)
        fcst_area = fcst_obj['area_mean'] * 9  # 3km grid cells (3x3 = 9 km² por celda)
        obs_area = obs_obj['area_mean'] * 100  # 10km grid cells (10x10 = 100 km² por celda)
        
        # Calcular razón de áreas
        return min(fcst_area, obs_area) / max(fcst_area, obs_area)

    ##-------------------------------------------------------------------------
    ## 3.15. Metodo _calculate_spatiotemporal_overlap
    ##-------------------------------------------------------------------------
    def _calculate_spatiotemporal_overlap(self, fcst_obj: Dict, obs_obj: Dict) -> float:
        """Calcula la máxima superposición espacial en tiempos coincidentes."""
        max_overlap = 0.0
        
        # Encontrar tiempos comunes
        common_times = set(fcst_obj['time_points']).intersection(obs_obj['time_points'])
        
        for t in common_times:
            # Encontrar objetos 2D correspondientes
            fcst_2d = next((o for o in fcst_obj['objects_2d'] if o['time'] == t), None)
            obs_2d = next((o for o in obs_obj['objects_2d'] if o['time'] == t), None)
            
            if fcst_2d and obs_2d:
                # Usar coordenadas geográficas para el cálculo de superposición
                overlap = self._calculate_geographic_overlap(
                    fcst_2d['coords_geo'], 
                    obs_2d['coords_geo']
                )
                max_overlap = max(max_overlap, overlap)
        
        return max_overlap

    ##-------------------------------------------------------------------------
    ## 3.16. Metodo _calculate_geographic_overlap
    ##-------------------------------------------------------------------------
    def _calculate_geographic_overlap(self, coords1: np.ndarray, coords2: np.ndarray, 
                                   tolerance_km: float = 10.0) -> float:
        """Calcula superposición geográfica entre dos conjuntos de coordenadas."""
        if len(coords1) == 0 or len(coords2) == 0:
            return 0.0
        
        # Convertir tolerancia de km a grados (aproximadamente)
        tolerance_deg = tolerance_km / 111.0
        
        # Encontrar puntos que están dentro de la tolerancia
        matching_points = 0
        for coord1 in coords1:
            for coord2 in coords2:
                distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                if distance <= tolerance_deg:
                    matching_points += 1
                    break
        
        # Calcular superposición como fracción
        union_size = len(coords1) + len(coords2) - matching_points
        return matching_points / union_size if union_size > 0 else 0.0

    ##-------------------------------------------------------------------------
    ## 3.17. Metodo _calculate_orientation_difference
    ##-------------------------------------------------------------------------
    def _calculate_orientation_difference(self, fcst_obj: Dict, obs_obj: Dict) -> Optional[float]:
        """Calcula diferencia de orientación promedio (en grados)."""
        orientation_diffs = []
        
        for t in set(fcst_obj['time_points']).intersection(obs_obj['time_points']):
            fcst_2d = next((o for o in fcst_obj['objects_2d'] if o['time'] == t), None)
            obs_2d = next((o for o in obs_obj['objects_2d'] if o['time'] == t), None)
            
            if fcst_2d and obs_2d and fcst_2d['eccentricity'] > 0.2 and obs_2d['eccentricity'] > 0.2:
                angle_diff = abs(fcst_2d['orientation'] - obs_2d['orientation'])
                angle_diff = min(angle_diff, 180 - angle_diff)  # Considerar simetría
                orientation_diffs.append(angle_diff)
        
        return np.mean(orientation_diffs) if orientation_diffs else None

    ##-------------------------------------------------------------------------
    ## 3.18. Metodo _calculate_time_overlap
    ##-------------------------------------------------------------------------
    def _calculate_time_overlap(self, obj1: Dict, obj2: Dict) -> float:
        """Calcula la fracción de superposición temporal."""
        common_times = set(obj1['time_points']).intersection(obj2['time_points'])
        all_times = set(obj1['time_points']).union(obj2['time_points'])
        return len(common_times) / len(all_times) if all_times else 0.0

    ##-------------------------------------------------------------------------
    ## 3.19. Metodo _calculate_metrics
    ##-------------------------------------------------------------------------
    def _calculate_metrics(self):
        """Calcula métricas de evaluación."""
        # Mediana del Máximo Interés (MMI)
        if self.interest_matrix is not None and self.interest_matrix.size > 0:
            max_interest_fcst = np.max(self.interest_matrix, axis=1)
            max_interest_obs = np.max(self.interest_matrix, axis=0)
            
            self.metrics['MMI_forecast'] = float(np.median(max_interest_fcst))
            self.metrics['MMI_observed'] = float(np.median(max_interest_obs))
            self.metrics['MMI'] = float(np.median(np.concatenate([max_interest_fcst, max_interest_obs])))
        else:
            self.metrics.update({'MMI_forecast': 0.0, 'MMI_observed': 0.0, 'MMI': 0.0})
        
        # Gilbert Skill Score (GSS)
        hits = len(self.matches)
        false_alarms = len(self.forecast_objects) - hits
        misses = len(self.observed_objects) - hits

        fcst_count = len(self.forecast_objects)
        obs_count = len(self.observed_objects)


        # Otras métricas
        self.metrics['hits'] = hits
        self.metrics['false_alarms'] = false_alarms   
        self.metrics['misses'] = misses
        self.metrics['forecast_objects'] = len(self.forecast_objects)
        self.metrics['observed_objects'] = len(self.observed_objects)

        
        # Estadísticas de objetos
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



        # =============================================================================
        # Gilbert Skill Score CLÁSICO (Corrección de resolución mediante interpolación) 
        # =============================================================================

        # 1. Interpolar el pronóstico para que tenga la misma rejilla espacial que la observación
        # Usamos interpolación lineal ("linear") o de vecino más próximo ("nearest")
        forecast_interp = self.forecast.interp_like(self.observed, method="nearest")

        # 2. Obtener las máscaras binarias usando la nueva matriz interpolada
        # Ahora ambas tendrán la dimensión de la observación: (24, 67, 134)
        fcst_mask = forecast_interp.values > self.threshold
        obs_mask = self.observed.values > self.threshold

        # 3. Calcular los componentes de la matriz de contingencia 
        # El número total de píxeles se basa en la rejilla de destino (la observación)
        N_pixels = self.observed.sizes['lat'] * self.observed.sizes['lon'] * self.observed.sizes['time']

        # Aciertos (Píxeles donde ambos son True)
        hits_classic = np.sum(fcst_mask & obs_mask)

        # Falsas Alarmas (Píxeles donde Pronóstico es True pero Observación es False)
        false_alarms_classic = np.sum(fcst_mask & ~obs_mask)

        # Fallos (Píxeles donde Pronóstico es False pero Observación es True)
        misses_classic = np.sum(~fcst_mask & obs_mask)

        # Negativos Correctos (Píxeles donde ambos son False)
        correct_negatives_classic = np.sum(~fcst_mask & ~obs_mask)

        # 4. Calcular los aciertos esperados por azar
        fcst_total_pixels = hits_classic + false_alarms_classic
        obs_total_pixels = hits_classic + misses_classic

        expected_random_hits_classic = (fcst_total_pixels * obs_total_pixels) / N_pixels if N_pixels > 0 else 0

        # 5. Calcular el GSS Clásico final
        denominator_classic = hits_classic + false_alarms_classic + misses_classic - expected_random_hits_classic

        if denominator_classic > 0:
            self.metrics['GSS'] = (hits_classic - expected_random_hits_classic) / denominator_classic
        else:
            self.metrics['GSS'] = 0.0

            
        # =============================================================================
        # # Object-Based Gilbert Skill Score 
        # =============================================================================

        # 1. Calcular el área total del dominio físico para CADA rejilla
        total_area_fcst = self.forecast.sizes['lat'] * self.forecast.sizes['lon']
        total_area_obs = self.observed.sizes['lat'] * self.observed.sizes['lon']

        # 2. Calcular las áreas ocupadas por objetos en sus respectivas resoluciones
        obs_area = sum(obj['area_mean'] for obj in self.observed_objects)
        fcst_area = sum(obj['area_mean'] for obj in self.forecast_objects)

        # 3. Calcular fracciones de área de manera independiente y correcta
        obs_area_fraction = obs_area / total_area_obs #if total_area_obs > 0 else 0
        fcst_area_fraction = fcst_area / total_area_fcst #if total_area_fcst > 0 else 0

        # 4. Estimar D (Negativos Correctos) usando la fracción de la observación
        D = (((1 - obs_area_fraction) * obs_count) / obs_area_fraction) - fcst_count

        # 5. Calcular epsilon (Aciertos esperados por azar en el espacio de objetos)
        # Salvaguarda: el denominador debe ser mayor a cero y epsilon no puede superar a los objetos existentes
        denominador_epsilon = fcst_count + misses + D

        if denominador_epsilon > 0:
            epsilon = (fcst_count * obs_count) / denominador_epsilon
            epsilon = min(epsilon, float(hits))  # El azar nunca puede ser mayor que los aciertos reales
        else:
            epsilon = 0.0

        # 6. Calcular el GSS Basado en Objetos Final
        denominator_gss = fcst_count + misses - epsilon

        if denominator_gss > 0:
            self.metrics['GSS_Obj-Based'] = (hits - epsilon) / denominator_gss
        else:
            # Si el denominador es cero pero el modelo hizo un trabajo perfecto (0 fcst, 0 obs)
            if fcst_count == 0 and obs_count == 0:
                self.metrics['GSS_Obj-Based'] = 1.0  # Castigo o premio por evento nulo perfecto
            else:
                self.metrics['GSS_Obj-Based'] = 0.0


        self.metrics['hits_random_classic'] = expected_random_hits_classic
        self.metrics['hits_random_objects'] = epsilon


    
    ##-------------------------------------------------------------------------
    ## 3.20. Metodo plot_matched_objects (MODIFICADO)
    ##-------------------------------------------------------------------------
    def plot_matched_objects(self, time_idx):    
        """
        Visualiza objetos emparejados como contornos sobre campo de precipitación.
        """
        if not self.matches:
            print("No hay matches para visualizar")
            return
        
        # Obtener el tiempo específico
        target_time = self.forecast.time.isel(time=time_idx).values
        
        # Obtener campos de precipitación para este tiempo
        precip_forecast = self.forecast.isel(time=time_idx)
        precip_observed = self.observed.isel(time=time_idx)
        
        # Filtrar objetos para este tiempo
        fcst_objs = [obj for obj in self.forecast_objects 
                    if target_time in obj['time_points']]
        obs_objs = [obj for obj in self.observed_objects 
                   if target_time in obj['time_points']]
        
        if not fcst_objs and not obs_objs:
            print(f"No hay objetos para el tiempo {target_time}")
            return
        
        # Obtener todos los matches para este tiempo
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
        
        # Configurar figura - 2 columnas: Obs y Forecast
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': ccrs.PlateCarree()}, sharex=True, sharey=True) #figsize=(16, 12) figsize=(16, 8)
        
        # Título principal
        fig.suptitle(f"Objects Comparison  MODE - {pd.Timestamp(target_time).strftime('%Y-%m-%d %H:%M')}\n"
                     f"Threshold: {self.threshold} mm, Convolution Radius WRF: {self.conv_radius_forecast} pixels Convolution Radius GPM: {self.conv_radius_observed} pixels\n"
                     f"Min Dist Object {config.min_dist_same_object}km", 
                     fontsize=16, y=0.8) #y=0.9

        
        # Plot 1: Objetos observados 
        ax1.set_title("Observed (GPM)")
        
        # 1. Primero pintar el campo de precipitación
        contour_obs = ax1.contourf(precip_observed['lon'], 
                                   precip_observed['lat'], 
                                   precip_observed,
                                   transform=ccrs.PlateCarree(),
                                   levels=np.arange(0, 50, 2),
                                   cmap=field_visualization.cmap,
                                   extend='both',
                                   alpha=0.8)
        
        # 2. Luego dibujar los contornos de los objetos
        self._draw_objects_as_contours(ax1, obs_objs, current_matches, 'observed', 
                                      precip_min=0, precip_max=50)
        
        # Plot 2: Objetos pronosticados
        # Dominio 3km o 1km
        ax2.set_title("Forecast (WRF)")
        
        # 1. Primero pintar el campo de precipitación
        contour_fcst = ax2.contourf(precip_forecast['lon'], 
                                    precip_forecast['lat'], 
                                    precip_forecast,
                                    transform=ccrs.PlateCarree(),
                                    levels=np.arange(0, 50, 2),
                                    cmap=field_visualization.cmap,
                                    extend='both',
                                    alpha=0.8)
        
        # 2. Luego dibujar los contornos de los objetos
        self._draw_objects_as_contours(ax2, fcst_objs, current_matches, 'forecast', 
                                      precip_min=0, precip_max=50)
        
        # Configuración común para ambos subplots
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
        
        # Ajustar límites para que ambos plots sean comparables
        min_lon = min(self.forecast.lon.min(), self.observed.lon.min())
        max_lon = max(self.forecast.lon.max(), self.observed.lon.max())
        min_lat = min(self.forecast.lat.min(), self.observed.lat.min())
        max_lat = max(self.forecast.lat.max(), self.observed.lat.max())
        
        for ax in [ax1, ax2]:
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
        # Añadir barra de color común para precipitación
        cbar_ax = fig.add_axes([0.08, 0.08, 0.8, 0.025])  # [left, bottom, width, height]
        cbar = plt.colorbar(contour_fcst, 
                            cax=cbar_ax, 
                            orientation='horizontal',
                            pad=0.5, #0.5
                            shrink=0.8, #0.8
                            aspect=60)
        cbar.set_label('Precipitation (mm/h)', fontsize=12)
        
    
        # Añadir leyenda de matches en la parte superior
        if current_matches:
            self._add_match_legend(fig, current_matches)
        
        plt.tight_layout()
        
        # Guardar figura
        save_path = os.path.join(config.path_figures, 
                               f'MODE_contours_{pd.Timestamp(target_time).strftime("%Y%m%d_%H%M")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #plt.show()
    
    ##-------------------------------------------------------------------------
    ## 3.21. Metodo _draw_objects_as_contours 
    ##-------------------------------------------------------------------------    
    def _draw_objects_as_contours(self, ax, objects, matches, obj_type, precip_min=0, precip_max=50):
        """
        Dibuja objetos como contornos sobre el campo de precipitación.
        
        Diferenciación:
        - Objetos MATCHED: Línea sólida gruesa con ID en círculo blanco
        - Objetos UNMATCHED: Línea punteada fina con ID en círculo gris
        """
        # Identificar IDs de objetos emparejados
        matched_ids = [m[obj_type]['id'] for m in matches]
        
        # Colores para objetos emparejados (uno por match)
        match_colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF',  # Colores brillantes
            '#FFA500', '#800080', '#008080', '#FF1493', '#32CD32'   # Más colores
        ]
        
        # Procesar cada objeto
        for i, obj in enumerate(objects):
            is_matched = obj['id'] in matched_ids
            
            # Determinar propiedades según si está emparejado o no
            if is_matched:
                # Encontrar el match correspondiente
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
            
            # Dibujar contorno del objeto
            self._draw_object_contour(ax, obj, line_color, line_width, line_style)
            
            # Dibujar centroide con ID
            self._draw_object_centroid(ax, obj, line_color, id_bg_color, id_text_color, is_matched)
            
            # Si está emparejado, dibujar línea conectando con el objeto emparejado
            if is_matched and obj_type == 'forecast':
                self._draw_connection_line(ax, obj, matches[match_idx], line_color)
        
        # Añadir texto informativo en la esquina
        #total_objects = len(objects)
        #matched_objects = len(matched_ids)
        
        #info_text = f"Objetos: {total_objects} ({matched_objects} matched)"
        #ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
        #        fontsize=10, verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ##-------------------------------------------------------------------------
    ## 3.22. Metodo _draw_object_contour 
    ##------------------------------------------------------------------------- 
    def _draw_object_contour(self, ax, obj, line_color, line_width, line_style):
        """Dibuja el contorno de un objeto como línea."""
        # Obtener las coordenadas del objeto para este tiempo
        obj_2d = next((o for o in obj['objects_2d'] 
                      if o['time'] == pd.Timestamp(obj['time_points'][0])), None)
        
        if obj_2d and 'coords_geo' in obj_2d and len(obj_2d['coords_geo']) > 0:
            # Usar coordenadas geográficas directamente
            lons = obj_2d['coords_geo'][:, 1]
            lats = obj_2d['coords_geo'][:, 0]
            
            # Crear convex hull para el contorno suave
            if len(lons) >= 3:
                from scipy.spatial import ConvexHull
                points = np.column_stack((lons, lats))
                
                try:
                    hull = ConvexHull(points)
                    
                    # Dibujar el polígono convexo como línea
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])  # Cerrar el polígono
                    
                    ax.plot(hull_points[:, 0], hull_points[:, 1], 
                           color=line_color, linewidth=line_width, 
                           linestyle=line_style, alpha=0.9,
                           transform=ccrs.PlateCarree())
                    
                    # Relleno muy sutil para destacar el área
                    ax.fill(hull_points[:, 0], hull_points[:, 1], 
                           color=line_color, alpha=0.05,  # Muy transparente
                           transform=ccrs.PlateCarree())
                    
                except Exception as e:
                    # Si falla el convex hull, dibujar puntos simples
                    print(f" Error en convex hull para objeto {obj['id']}: {e}")
                    ax.plot(lons, lats, 'o', color=line_color, markersize=2,
                           transform=ccrs.PlateCarree(), alpha=0.7)
    
    ##-------------------------------------------------------------------------
    ## 3.23. Metodo _draw_object_centroid 
    ##-------------------------------------------------------------------------
    def _draw_object_centroid(self, ax, obj, line_color, bg_color, text_color, is_matched):
        """Dibuja el centroide del objeto con su ID."""
        centroid_lon = np.mean([c[1] for c in obj['centroid_trajectory_geo']])
        centroid_lat = np.mean([c[0] for c in obj['centroid_trajectory_geo']])
        
        # Dibujar marcador del centroide
        marker_size = 100 if is_matched else 80
        marker_edge_width = 2 if is_matched else 1
        
        ax.scatter(centroid_lon, centroid_lat, 
                  color=line_color, s=marker_size,
                  edgecolors=bg_color, linewidth=marker_edge_width,
                  zorder=10, alpha=0.9,
                  transform=ccrs.PlateCarree())
        
        # Añadir texto con ID
        #ax.text(centroid_lon, centroid_lat, str(obj['id']), 
        #       ha='center', va='center', fontsize=9 if is_matched else 8,
        #       fontweight='bold' if is_matched else 'normal',
        #       color=text_color,
        #       bbox=dict(boxstyle='circle,pad=0.2', 
        #                facecolor=bg_color, 
        #                edgecolor=line_color if is_matched else 'gray',
        #                linewidth=1.5 if is_matched else 1,
        #                alpha=0.9),
        #       transform=ccrs.PlateCarree())
    
    ##-------------------------------------------------------------------------
    ## 3.24. Metodo _draw_connection_line 
    ##-------------------------------------------------------------------------
    def _draw_connection_line(self, ax, fcst_obj, match, line_color):
        """Dibuja línea conectando objetos emparejados."""
        paired_obj = match['observed']
        
        fcst_centroid_lon = np.mean([c[1] for c in fcst_obj['centroid_trajectory_geo']])
        fcst_centroid_lat = np.mean([c[0] for c in fcst_obj['centroid_trajectory_geo']])
        
        obs_centroid_lon = np.mean([c[1] for c in paired_obj['centroid_trajectory_geo']])
        obs_centroid_lat = np.mean([c[0] for c in paired_obj['centroid_trajectory_geo']])
        
        # Dibujar línea punteada entre centroides
        ax.plot([fcst_centroid_lon, obs_centroid_lon], 
               [fcst_centroid_lat, obs_centroid_lat], 
               color=line_color, linestyle=':', alpha=0.7, 
               linewidth=1.5, transform=ccrs.PlateCarree())
        
        # Añadir texto con interés del match a mitad de la línea
        #mid_lon = (fcst_centroid_lon + obs_centroid_lon) / 2
        #mid_lat = (fcst_centroid_lat + obs_centroid_lat) / 2
        
        #ax.text(mid_lon, mid_lat, f"{match['interest']:.2f}", 
        #      ha='center', va='center', fontsize=8,
        #       color=line_color, fontweight='bold',
        #       bbox=dict(boxstyle='round,pad=0.2', 
        #                facecolor='white', alpha=0.8),
        #       transform=ccrs.PlateCarree())
    
    ##-------------------------------------------------------------------------
    ## 3.25. Metodo _add_match_legend 
    ##-------------------------------------------------------------------------
    def _add_match_legend(self, fig, current_matches):
        """Añade leyenda personalizada para los matches."""
        from matplotlib.lines import Line2D
        
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
        
        # Añadir elementos específicos para cada match
        match_colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF',
            '#FFA500', '#800080', '#008080', '#FF1493', '#32CD32'
        ]
        
        for i, match in enumerate(current_matches):
            if i < len(match_colors):  # Limitar a 10 matches en la leyenda
                legend_elements.append(
                    Line2D([0], [0], color=match_colors[i], linewidth=3, linestyle='solid',
                           label=f'Match {i+1}: Int={match["interest"]:.2f}')
                )
        
        # Crear leyenda en la parte superior
        fig.legend(handles=legend_elements, 
                   loc='lower center', 
                   ncol=min(5, len(current_matches) + 2), 
                   bbox_to_anchor=(0.5, -0.05),
                   fontsize=16)
        
    
    ##-------------------------------------------------------------------------
    ## 3.23. Metodo save_metrics_to_csv
    ##-------------------------------------------------------------------------     
    def save_metrics_to_csv(self, csv_path=None, append=False, config_params=None):
        """
        Guarda todas las métricas de MODE en un archivo CSV en formato largo.
        
        Args:
            csv_path: Ruta del archivo CSV (si None, usa path por defecto)
            append: Si True, añade al archivo existente en lugar de sobrescribir
            config_params: Diccionario con parámetros de configuración usados
        """
        csv_path=os.path.join(config.path_statistics, f"MODE_results T={self.threshold} RO={self.conv_radius_observed} RF={self.conv_radius_forecast} .csv")
        if not self.metrics:
            print("No hay métricas para guardar. Ejecuta primero run_verification().")
            return
        
        # Definir path por defecto si no se proporciona
        if csv_path is None:
            csv_path = os.path.join(config.path_statistics, "MODE_metrics_long.csv")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Preparar datos para CSV en formato largo
        metrics_data = self.metrics.copy()
        
        # Añadir metadatos y parámetros de configuración
        metrics_data['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        #metrics_data['forecast_resolution'] = '1km'
        metrics_data['forecast_resolution'] = '3km'
        metrics_data['observed_resolution'] = '10km'
        
        # Añadir parámetros de configuración si se proporcionan
        if config_params:
            metrics_data.update(config_params)
        else:
            # Añadir parámetros por defecto de la instancia
            metrics_data.update({
                'threshold': self.threshold,
                'conv_radius_forecast': self.conv_radius_forecast,
                'conv_radius_observed': self.conv_radius_observed,
                'time_window': self.time_window,
                'min_object_size': self.min_object_size,
                'interest_threshold': getattr(self, 'interest_threshold', 0.5)
            })
        
        # Añadir información de dimensiones
        metrics_data['grid_lat'] = self.forecast.sizes['lat']
        metrics_data['grid_lon'] = self.forecast.sizes['lon']
        metrics_data['time_steps'] = self.forecast.sizes['time']
        
        # Convertir a formato largo (melt)
        df_metrics_long = pd.DataFrame(list(metrics_data.items()), columns=['metric', 'value'])
        
        # Añadir timestamp como columna adicional para identificar cada ejecución
        timestamp = metrics_data['timestamp']
        df_metrics_long['timestamp'] = timestamp
        
        # Reordenar columnas
        df_metrics_long = df_metrics_long[['timestamp', 'metric', 'value']]
        
        # Guardar en CSV
        if append and os.path.exists(csv_path):
            # Modo append
            df_metrics_long.to_csv(csv_path, mode='a', header=False, index=False, float_format='%.4f')
            
            print(f"Métricas añadidas al archivo: {csv_path}")
        else:
            # Modo write (sobrescribir o crear nuevo)
            df_metrics_long.to_csv(csv_path, index=False, float_format='%.4f')
            print(f"Métricas guardadas en: {csv_path}")
        
        
        return csv_path


