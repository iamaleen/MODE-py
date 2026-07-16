#!/usr/bin/env python3
##-----------------------------------------------------------------------------
## Análisis Estadístico Independiente para MODE
##-----------------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from typing import Dict, List, Optional
from mode_verifier import MODE3DVerifier
import config

class MODEStatisticalAnalyzer:
    """
    Clase independiente para análisis estadístico de resultados MODE
    Puede trabajar con objetos MODE3DVerifier existentes o cargar resultados guardados
    """
    
    def __init__(self, verifier: Optional[MODE3DVerifier] = None, results_path: Optional[str] = None):
        """
        Inicializa el analizador con un verificador existente o cargando resultados
        
        Args:
            verifier: Instancia de MODE3DVerifier con resultados calculados
            results_path: Ruta para cargar resultados guardados
        """
        if verifier:
            self.verifier = verifier
            self.results_loaded = True
        elif results_path:
            self.load_verification_results(results_path)
        else:
            raise ValueError("Debe proporcionar un verificador o una ruta de resultados")
    
    def load_verification_results(self, results_path: str):
        """Carga resultados de verificación desde archivo"""
        try:
            with open(results_path, 'rb') as f:
                self.verifier = pickle.load(f)
            self.results_loaded = True
            print(f"Resultados cargados desde: {results_path}")
        except Exception as e:
            raise ValueError(f"Error cargando resultados: {e}")
    
    def save_verification_results(self, results_path: str):
        """Guarda resultados de verificación para análisis posterior"""
        if hasattr(self, 'verifier'):
            with open(results_path, 'wb') as f:
                pickle.dump(self.verifier, f)
            print(f"Resultados guardados en: {results_path}")
    
    ##-------------------------------------------------------------------------
    ## 1. Método _calculate_object_intensity
    ##-------------------------------------------------------------------------
    def _calculate_object_intensity(self, time_obj: Dict, obj_type: str):
        """Calcula los valores de intensidad para un objeto 2D."""
        coords = time_obj['coords_pixel']
        time_idx = time_obj['time_idx']
        
        if obj_type == 'forecast':
            intensity_data = self.verifier.smoothed_fcst.isel(time=time_idx).values
        else:
            intensity_data = self.verifier.smoothed_obs.isel(time=time_idx).values
        
        # Extraer valores de intensidad para todas las coordenadas del objeto
        intensity_vals = []
        for y, x in coords:
            y_idx = self._safe_index(y, intensity_data.shape[0])
            x_idx = self._safe_index(x, intensity_data.shape[1])
            intensity_vals.append(intensity_data[y_idx, x_idx])
        
        time_obj['intensity_values'] = intensity_vals 

    def _safe_index(self, value: float, max_size: int) -> int:
        """Convierte un valor a índice de forma segura."""
        return min(max(0, int(round(value))), max_size-1)

    ##-------------------------------------------------------------------------
    ## 2. Método analyze_precipitation_quantiles
    ##------------------------------------------------------------------------- 
    def analyze_precipitation_quantiles(self):
        """Analiza la distribución de precipitación por cuartiles dentro de los objetos."""
        
        self.quantile_results = {
            'forecast': {'Q1': [], 'Q2': [], 'Q3': [], 'max': []},
            'observed': {'Q1': [], 'Q2': [], 'Q3': [], 'max': []},
            'matched_pairs': []
        }
        
        # Analizar objetos individuales
        for obj_type in ['forecast', 'observed']:
            objects = getattr(self.verifier, f'{obj_type}_objects')
            
            for obj in objects:
                intensities = []
                for time_obj in obj['objects_2d']:
                    # Obtener intensidades de todos los píxeles del objeto
                    if 'intensity_values' not in time_obj:
                        self._calculate_object_intensity(time_obj, obj_type)
                    
                    intensities.extend(time_obj['intensity_values'])
                
                if intensities:
                    q1, q2, q3 = np.percentile(intensities, [25, 50, 75])
                    self.quantile_results[obj_type]['Q1'].append(q1)
                    self.quantile_results[obj_type]['Q2'].append(q2)
                    self.quantile_results[obj_type]['Q3'].append(q3)
                    self.quantile_results[obj_type]['max'].append(max(intensities))
        
        # Analizar pares emparejados
        for match in self.verifier.matches:
            fcst_obj = next(o for o in self.verifier.forecast_objects if o['id'] == match['forecast_id'])
            obs_obj = next(o for o in self.verifier.observed_objects if o['id'] == match['observed_id'])
            
            # Calcular diferencias en cuartiles entre objetos emparejados
            fcst_intensities = []
            obs_intensities = []
            
            for time_obj in fcst_obj['objects_2d']:
                if 'intensity_values' not in time_obj:
                    self._calculate_object_intensity(time_obj, 'forecast')
                fcst_intensities.extend(time_obj['intensity_values'])
            
            for time_obj in obs_obj['objects_2d']:
                if 'intensity_values' not in time_obj:
                    self._calculate_object_intensity(time_obj, 'observed')
                obs_intensities.extend(time_obj['intensity_values'])
            
            if fcst_intensities and obs_intensities:
                fcst_q1, fcst_q2, fcst_q3 = np.percentile(fcst_intensities, [25, 50, 75])
                obs_q1, obs_q2, obs_q3 = np.percentile(obs_intensities, [25, 50, 75])
                
                self.quantile_results['matched_pairs'].append({
                    'interest': match['interest'],
                    'q1_diff': fcst_q1 - obs_q1,
                    'q2_diff': fcst_q2 - obs_q2,
                    'q3_diff': fcst_q3 - obs_q3,
                    'max_diff': max(fcst_intensities) - max(obs_intensities),
                    'area_ratio': fcst_obj['area_mean'] / obs_obj['area_mean']
                })
        
        return self.quantile_results
    
    ##-------------------------------------------------------------------------
    ## 2.1. Método plot_quantile_analysis
    ##-------------------------------------------------------------------------
    def plot_quantile_analysis(self, save_path: str = None):
        """Visualiza el análisis por cuartiles en gráficos combinados"""
        
        if not hasattr(self, 'quantile_results'):
            self.analyze_precipitation_quantiles()
        
        if save_path is None:
            save_path = config.path_statistics
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Boxplot de cuartiles por tipo de objeto
        data_to_plot = [self.quantile_results['forecast']['Q1'], self.quantile_results['observed']['Q1'],
                       self.quantile_results['forecast']['Q2'], self.quantile_results['observed']['Q2'],
                       self.quantile_results['forecast']['Q3'], self.quantile_results['observed']['Q3']]
        
        axes[0, 0].boxplot(data_to_plot, positions=[1, 2, 4, 5, 7, 8])
        axes[0, 0].set_xticks([1.5, 4.5, 7.5])
        axes[0, 0].set_xticklabels(['Q1', 'Q2', 'Q3'])
        axes[0, 0].set_title('Distribución de Cuartiles por Tipo de Objeto')
        axes[0, 0].set_ylabel('Precipitación (mm)')
        axes[0, 0].legend(['WRF', 'GPM'], loc='upper right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Diferencias de cuartiles en pares emparejados
        if self.quantile_results['matched_pairs']:
            pairs = self.quantile_results['matched_pairs']
            interest_values = [p['interest'] for p in pairs]
            q1_diffs = [p['q1_diff'] for p in pairs]
            q2_diffs = [p['q2_diff'] for p in pairs]
            q3_diffs = [p['q3_diff'] for p in pairs]
            
            axes[0, 1].scatter(interest_values, q1_diffs, alpha=0.6, label='Q1', s=50)
            axes[0, 1].scatter(interest_values, q2_diffs, alpha=0.6, label='Q2', s=50)
            axes[0, 1].scatter(interest_values, q3_diffs, alpha=0.6, label='Q3', s=50)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Interés del Match')
            axes[0, 1].set_ylabel('Diferencia (WRF - GPM)')
            axes[0, 1].set_title('Diferencias de Cuartiles vs. Interés del Match')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Distribución de diferencias
        if self.quantile_results['matched_pairs']:
            axes[1, 0].hist(q1_diffs, alpha=0.5, label='Q1', bins=20, edgecolor='black')
            axes[1, 0].hist(q2_diffs, alpha=0.5, label='Q2', bins=20, edgecolor='black')
            axes[1, 0].hist(q3_diffs, alpha=0.5, label='Q3', bins=20, edgecolor='black')
            axes[1, 0].axvline(x=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Diferencia (WRF - GPM)')
            axes[1, 0].set_ylabel('Frecuencia')
            axes[1, 0].set_title('Distribución de Diferencias por Cuartil')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Relación entre diferencia de máximos y área
        if self.quantile_results['matched_pairs']:
            max_diffs = [p['max_diff'] for p in pairs]
            area_ratios = [p['area_ratio'] for p in pairs]
            
            scatter = axes[1, 1].scatter(area_ratios, max_diffs, c=interest_values, 
                                       cmap='viridis', alpha=0.7, s=50)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].axvline(x=1, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Ratio de Área (WRF/GPM)')
            axes[1, 1].set_ylabel('Diferencia de Máximos (WRF - GPM)')
            axes[1, 1].set_title('Diferencia de Máximos vs. Ratio de Área')
            plt.colorbar(scatter, ax=axes[1, 1], label='Interés del Match')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'quantile_analysis_combined.png'), 
                   dpi=300, bbox_inches='tight')
        #plt.show()

    ##-------------------------------------------------------------------------
    ## 2.2. Método plot_individual_quantile_analysis
    ##-------------------------------------------------------------------------
    def plot_individual_quantile_analysis(self, save_path: str = None):
        """Visualiza el análisis por cuartiles en gráficos independientes"""
        
        if not hasattr(self, 'quantile_results'):
            self.analyze_precipitation_quantiles()
        
        if save_path is None:
            save_path = config.path_statistics
        
        # Crear directorio si no existe
        os.makedirs(save_path, exist_ok=True)
        
        # Gráfico 1: Boxplot de cuartiles por tipo de objeto
        plt.figure(figsize=(10, 6))
        data_to_plot = [self.quantile_results['forecast']['Q1'], self.quantile_results['observed']['Q1'],
                       self.quantile_results['forecast']['Q2'], self.quantile_results['observed']['Q2'],
                       self.quantile_results['forecast']['Q3'], self.quantile_results['observed']['Q3']]
        
        plt.boxplot(data_to_plot, positions=[1, 2, 4, 5, 7, 8])
        plt.xticks([1.5, 4.5, 7.5], ['Q1', 'Q2', 'Q3'])
        plt.title('Distribución de Cuartiles por Tipo de Objeto')
        plt.ylabel('Precipitación (mm)')
        plt.legend(['WRF', 'GPM'], loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'quantile_boxplot.png'), dpi=300, bbox_inches='tight')
        #plt.show()
        
        # Gráfico 2: Diferencias de cuartiles en pares emparejados
        if self.quantile_results['matched_pairs']:
            plt.figure(figsize=(10, 6))
            pairs = self.quantile_results['matched_pairs']
            interest_values = [p['interest'] for p in pairs]
            q1_diffs = [p['q1_diff'] for p in pairs]
            q2_diffs = [p['q2_diff'] for p in pairs]
            q3_diffs = [p['q3_diff'] for p in pairs]
            
            plt.scatter(interest_values, q1_diffs, alpha=0.6, label='Q1', s=50)
            plt.scatter(interest_values, q2_diffs, alpha=0.6, label='Q2', s=50)
            plt.scatter(interest_values, q3_diffs, alpha=0.6, label='Q3', s=50)
            plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
            plt.xlabel('Interés del Match')
            plt.ylabel('Diferencia (WRF - GPM)')
            plt.title('Diferencias de Cuartiles vs. Interés del Match')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'quantile_differences_scatter.png'), 
                       dpi=300, bbox_inches='tight')
            #plt.show()
        
        # Gráfico 3: Relación entre diferencia de máximos y área
        if self.quantile_results['matched_pairs']:
            plt.figure(figsize=(10, 6))
            pairs = self.quantile_results['matched_pairs']
            max_diffs = [p['max_diff'] for p in pairs]
            area_ratios = [p['area_ratio'] for p in pairs]
            interest_values = [p['interest'] for p in pairs]
            
            scatter = plt.scatter(area_ratios, max_diffs, c=interest_values, 
                               cmap='viridis', alpha=0.7, s=50)
            plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
            plt.axvline(x=1, color='r', linestyle='--', linewidth=1)
            plt.xlabel('Ratio de Área (WRF/GPM)')
            plt.ylabel('Diferencia de Máximos (WRF - GPM)')
            plt.title('Diferencia de Máximos vs. Ratio de Área')
            plt.colorbar(scatter, label='Interés del Match')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'max_difference_vs_area.png'), 
                       dpi=300, bbox_inches='tight')
            #plt.show()
        
        # Gráfico 4: Distribución de diferencias
        if self.quantile_results['matched_pairs']:
            plt.figure(figsize=(10, 6))
            pairs = self.quantile_results['matched_pairs']
            q1_diffs = [p['q1_diff'] for p in pairs]
            q2_diffs = [p['q2_diff'] for p in pairs]
            q3_diffs = [p['q3_diff'] for p in pairs]
            
            plt.hist(q1_diffs, alpha=0.5, label='Q1', bins=20, edgecolor='black')
            plt.hist(q2_diffs, alpha=0.5, label='Q2', bins=20, edgecolor='black')
            plt.hist(q3_diffs, alpha=0.5, label='Q3', bins=20, edgecolor='black')
            plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
            plt.xlabel('Diferencia (WRF - GPM)')
            plt.ylabel('Frecuencia')
            plt.title('Distribución de Diferencias por Cuartil')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'differences_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            #plt.show()

    ##-------------------------------------------------------------------------
    ## 3. Método analyze_temporal_persistence
    ##-------------------------------------------------------------------------
    def analyze_temporal_persistence(self):
        """Analiza la persistencia temporal de los objetos"""
        
        fcst_durations = [obj['duration'] for obj in self.verifier.forecast_objects]
        obs_durations = [obj['duration'] for obj in self.verifier.observed_objects]
        
        matched_fcst_durations = []
        matched_obs_durations = []
        unmatched_fcst_durations = []
        unmatched_obs_durations = []
        
        matched_ids = {m['forecast_id'] for m in self.verifier.matches}
        
        for obj in self.verifier.forecast_objects:
            if obj['id'] in matched_ids:
                matched_fcst_durations.append(obj['duration'])
            else:
                unmatched_fcst_durations.append(obj['duration'])
        
        matched_ids = {m['observed_id'] for m in self.verifier.matches}
        
        for obj in self.verifier.observed_objects:
            if obj['id'] in matched_ids:
                matched_obs_durations.append(obj['duration'])
            else:
                unmatched_obs_durations.append(obj['duration'])
        
        self.persistence_metrics = {
            'fcst_mean_duration': np.mean(fcst_durations) if fcst_durations else 0,
            'obs_mean_duration': np.mean(obs_durations) if obs_durations else 0,
            'matched_fcst_mean_duration': np.mean(matched_fcst_durations) if matched_fcst_durations else 0,
            'matched_obs_mean_duration': np.mean(matched_obs_durations) if matched_obs_durations else 0,
        }
        
        # Calcular correlación de duración para pares emparejados
        if self.verifier.matches:
            interests = []
            fcst_durs = []
            obs_durs = []
            
            for match in self.verifier.matches:
                fcst_obj = next(o for o in self.verifier.forecast_objects if o['id'] == match['forecast_id'])
                obs_obj = next(o for o in self.verifier.observed_objects if o['id'] == match['observed_id'])
                
                interests.append(match['interest'])
                fcst_durs.append(fcst_obj['duration'])
                obs_durs.append(obs_obj['duration'])
            
            if fcst_durs and obs_durs:
                self.persistence_metrics['duration_correlation'] = np.corrcoef(fcst_durs, obs_durs)[0, 1]
            else:
                self.persistence_metrics['duration_correlation'] = 0
        
        return self.persistence_metrics

    ##-------------------------------------------------------------------------
    ## 3.1. Método plot_temporal_persistence
    ##-------------------------------------------------------------------------
    def plot_temporal_persistence(self, save_path: str = None):
        """Visualiza la persistencia temporal de los objetos"""
        
        if not hasattr(self, 'persistence_metrics'):
            self.analyze_temporal_persistence()
        
        if save_path is None:
            save_path = config.path_statistics
        
        fcst_durations = [obj['duration'] for obj in self.verifier.forecast_objects]
        obs_durations = [obj['duration'] for obj in self.verifier.observed_objects]
        
        matched_fcst_durations = []
        matched_obs_durations = []
        unmatched_fcst_durations = []
        unmatched_obs_durations = []
        
        matched_ids = {m['forecast_id'] for m in self.verifier.matches}
        
        for obj in self.verifier.forecast_objects:
            if obj['id'] in matched_ids:
                matched_fcst_durations.append(obj['duration'])
            else:
                unmatched_fcst_durations.append(obj['duration'])
        
        matched_ids = {m['observed_id'] for m in self.verifier.matches}
        
        for obj in self.verifier.observed_objects:
            if obj['id'] in matched_ids:
                matched_obs_durations.append(obj['duration'])
            else:
                unmatched_obs_durations.append(obj['duration'])
        
        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Boxplot de duraciones
        data = [fcst_durations, obs_durations, 
                matched_fcst_durations, matched_obs_durations,
                unmatched_fcst_durations, unmatched_obs_durations]
        
        axes[0].boxplot(data)
        axes[0].set_xticks([1, 2, 3, 4, 5, 6])
        axes[0].set_xticklabels(['WRF Total', 'GPM Total', 'WRF Matched', 'GPM Matched', 
                               'WRF Unmatched', 'GPM Unmatched'], rotation=45)
        axes[0].set_ylabel('Duración (pasos temporales)')
        axes[0].set_title('Persistencia Temporal de Objetos')
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot: duración vs interés para objetos emparejados
        if self.verifier.matches:
            interests = []
            fcst_durs = []
            obs_durs = []
            
            for match in self.verifier.matches:
                fcst_obj = next(o for o in self.verifier.forecast_objects if o['id'] == match['forecast_id'])
                obs_obj = next(o for o in self.verifier.observed_objects if o['id'] == match['observed_id'])
                
                interests.append(match['interest'])
                fcst_durs.append(fcst_obj['duration'])
                obs_durs.append(obs_obj['duration'])
            
            scatter = axes[1].scatter(fcst_durs, obs_durs, c=interests, cmap='viridis', alpha=0.7, s=50)
            max_dur = max(max(fcst_durs), max(obs_durs)) if fcst_durs and obs_durs else 10
            axes[1].plot([0, max_dur], [0, max_dur], 'r--')
            axes[1].set_xlabel('Duración WRF (pasos)')
            axes[1].set_ylabel('Duración GPM (pasos)')
            axes[1].set_title('Duración de Pares Emparejados')
            plt.colorbar(scatter, ax=axes[1], label='Interés')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'temporal_persistence.png'), dpi=300, bbox_inches='tight')
        #plt.show()

    ##-------------------------------------------------------------------------
    ## 4. Método run_complete_statistical_analysis
    ##-------------------------------------------------------------------------
    def run_complete_statistical_analysis(self, save_path: str = None):
        """Ejecuta el análisis estadístico completo con todas las visualizaciones"""
        
        if save_path is None:
            save_path = config.path_statistics
        
        print("="*60)
        print("ANÁLISIS ESTADÍSTICO COMPLETO MODE")
        print("="*60)
        
        # Ejecutar análisis avanzados
        print("\n1. Ejecutando análisis de cuartiles...")
        quantile_results = self.analyze_precipitation_quantiles()
        
        print("2. Ejecutando análisis de persistencia temporal...")
        persistence_metrics = self.analyze_temporal_persistence()
        
        # Generar visualizaciones
        print("3. Generando visualizaciones...")
        self.plot_quantile_analysis(save_path)
        self.plot_individual_quantile_analysis(save_path)
        self.plot_temporal_persistence(save_path)
        
        # Mostrar métricas resumen
        print("\n" + "="*60)
        print("MÉTRICAS ESTADÍSTICAS RESUMEN")
        print("="*60)
        
        print(f"\nANÁLISIS DE CUARTILES:")
        print(f"Objetos WRF analizados: {len(quantile_results['forecast']['Q1'])}")
        print(f"Objetos GPM analizados: {len(quantile_results['observed']['Q1'])}")
        print(f"Pares emparejados analizados: {len(quantile_results['matched_pairs'])}")
        
        if quantile_results['matched_pairs']:
            q1_diffs = [p['q1_diff'] for p in quantile_results['matched_pairs']]
            q2_diffs = [p['q2_diff'] for p in quantile_results['matched_pairs']]
            q3_diffs = [p['q3_diff'] for p in quantile_results['matched_pairs']]
            
            print(f"Diferencia media Q1 (WRF-GPM): {np.mean(q1_diffs):.3f} mm")
            print(f"Diferencia media Q2 (WRF-GPM): {np.mean(q2_diffs):.3f} mm")
            print(f"Diferencia media Q3 (WRF-GPM): {np.mean(q3_diffs):.3f} mm")
        
        print(f"\nPERSISTENCIA TEMPORAL:")
        print(f"Duración media objetos WRF: {persistence_metrics['fcst_mean_duration']:.2f} pasos")
        print(f"Duración media objetos GPM: {persistence_metrics['obs_mean_duration']:.2f} pasos")
        print(f"Duración media WRF emparejados: {persistence_metrics['matched_fcst_mean_duration']:.2f} pasos")
        print(f"Duración media GPM emparejados: {persistence_metrics['matched_obs_mean_duration']:.2f} pasos")
        
        if 'duration_correlation' in persistence_metrics:
            print(f"Correlación duraciones emparejadas: {persistence_metrics['duration_correlation']:.3f}")
        
        print(f"\nResultados guardados en: {save_path}")
        print("Análisis estadístico completado!")

def main():
    """Función principal para análisis estadístico independiente"""
    
    # Opción 1: Cargar resultados guardados
    analyzer = MODEStatisticalAnalyzer(results_path=config.path_statistics + 'mode_verification_results.pkl')
    
    # Opción 2: Usar verificación existente 
    #print("Este script requiere una instancia de MODE3DVerifier con resultados calculados")
    #print("Use desde otro script o proporcione la ruta de resultados guardados")
    
    # Ejemplo de uso:
    #from run_mode_verification import main as run_mode_verification
    #from mode_verifier import MODE3DVerifier
    
    # Primero ejecutar verificación MODE
    #verifier = run_mode_verification()  # Esto debería retornar el verificador
    # 
    # Luego análisis estadístico
    #analyzer = MODEStatisticalAnalyzer(verifier=verifier)
    analyzer.run_complete_statistical_analysis()

if __name__ == "__main__":
    main()
