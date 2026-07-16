#!/usr/bin/env python3
##-----------------------------------------------------------------------------
## Análisis de Sensibilidad Independiente para MODE
##-----------------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from tqdm import tqdm
from typing import Dict, List, Optional

# Importar los módulos necesarios para cargar datos
import sys
import os

# Añadir la ruta para poder importar los otros módulos
sys.path.append(os.path.dirname(__file__))

from data_loader_ import load_gpm_data, load_wrf_data
from preprocessor_ import preprocess_datasets
from mode_verifier import MODE3DVerifier
import config

class MODESensitivityAnalyzer:
    """
    Clase independiente para análisis de sensibilidad de parámetros MODE
    Incluye GSS_Obj-Based y guarda resultados en CSV
    """
    
    def __init__(self, verifier: Optional[MODE3DVerifier] = None, 
                 results_path: Optional[str] = None,
                 forecast_data: Optional = None,
                 observed_data: Optional = None):
        """
        Inicializa el analizador de sensibilidad de múltiples formas
        """
        if verifier:
            self.verifier = verifier
            self.forecast = verifier.forecast
            self.observed = verifier.observed
            self.data_loaded = True
            print("✓ Usando verificador MODE existente")
            
        elif results_path:
            self.load_verification_results(results_path)
            
        elif forecast_data is not None and observed_data is not None:
            self.forecast = forecast_data
            self.observed = observed_data
            self.data_loaded = True
            print("✓ Usando datos de entrada directamente")
            
        else:
            print("Cargando datos GPM y WRF...")
            self.load_fresh_data()
    
    def load_fresh_data(self):
        """Carga datos GPM y WRF frescos"""
        try:
            ds_gpm_hourly = load_gpm_data()
            ds_wrf = load_wrf_data()
            
            self.observed, self.forecast = preprocess_datasets(ds_gpm_hourly, ds_wrf)
            self.data_loaded = True
            print("✓ Datos GPM y WRF cargados y preprocesados exitosamente")
            
        except Exception as e:
            raise ValueError(f"Error cargando datos: {e}")
    
    def load_verification_results(self, results_path: str):
        """Carga resultados de verificación desde archivo .pkl"""
        try:
            with open(results_path, 'rb') as f:
                self.verifier = pickle.load(f)
            self.forecast = self.verifier.forecast
            self.observed = self.verifier.observed
            self.data_loaded = True
            print(f"✓ Resultados cargados desde: {results_path}")
        except Exception as e:
            print(f"Error cargando resultados .pkl: {e}")
            print("Intentando cargar datos frescos...")
            self.load_fresh_data()
    
    ##-------------------------------------------------------------------------
    ## 1. Método plot_parameter_sensitivity (ACTUALIZADO CON GSS_Obj-Based)
    ##-------------------------------------------------------------------------
    def plot_parameter_sensitivity(self, 
                                 conv_radio_range=None, 
                                 thresholds_range=None,
                                 figsize=(16, 12), 
                                 cmap='coolwarm'):
        """
        Visualiza la sensibilidad de las métricas a diferentes parámetros
        AHORA INCLUYE GSS_Obj-Based EN EL HEATMAP 4
        """
        # Verificar que los datos estén cargados
        if not hasattr(self, 'data_loaded') or not self.data_loaded:
            self.load_fresh_data()
        
        # Valores por defecto IDÉNTICOS al script original
        if conv_radio_range is None:
            conv_radio_range = {
                'fcst': [6, 8, 10, 12, 14, 16],
                'obs': [3, 4, 5, 6, 7, 8]
            }
        
        if thresholds_range is None:
            #thresholds_range = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
            thresholds_range = [10.0, 15.0, 20.0]
        
        print("="*60)
        print("CALCULANDO SENSIBILIDAD A PARÁMETROS...")
        print("="*60)
        print(f"Radios WRF: {conv_radio_range['fcst']}")
        print(f"Radios GPM: {conv_radio_range['obs']}")
        print(f"Umbrales: {thresholds_range}")
        
        # Matrices para almacenar resultados (AHORA CON GSS_Obj-Based)
        n_radio = len(conv_radio_range['fcst'])
        n_thresholds = len(thresholds_range)
        
        gss_matrix = np.zeros((n_radio, n_thresholds))
        mmi_matrix = np.zeros((n_radio, n_thresholds))
        objects_count = np.zeros((n_radio, n_thresholds))
        gss_obj_based_matrix = np.zeros((n_radio, n_thresholds))  # NUEVA MATRIZ
        
        # Lista para almacenar todos los resultados para CSV
        all_results = []
        
        # Ejecutar MODE para cada combinación de parámetros
        total_combinations = n_radio * n_thresholds
        current_combination = 0
        
        print(f"\nEvaluando {total_combinations} combinaciones de parámetros...")
        
        for i, fcst_radius in enumerate(conv_radio_range['fcst']):
            for j, threshold in enumerate(thresholds_range):
                current_combination += 1
                print(f"Procesando combinación {current_combination}/{total_combinations}: "
                      f"radio_WRF={fcst_radius}, umbral={threshold}mm/h")
                
                try:
                    # Usar radio proporcional para GPM
                    obs_radius = conv_radio_range['obs'][i] if i < len(conv_radio_range['obs']) else conv_radio_range['obs'][-1]
                    
                    # Crear nueva instancia con parámetros diferentes
                    temp_verifier = MODE3DVerifier(
                        forecast=self.forecast,
                        observed=self.observed,
                        threshold=threshold,
                        conv_radius_forecast=fcst_radius,
                        conv_radius_observed=obs_radius,
                        time_window=1,
                        min_object_size=15
                    )
                    
                    # Ejecutar verificación
                    metrics = temp_verifier.run_verification(interest_threshold=0.6)
                    
                    # Almacenar resultados en matrices
                    gss_matrix[i, j] = metrics.get('GSS', 0)
                    mmi_matrix[i, j] = metrics.get('MMI', 0)
                    objects_count[i, j] = metrics.get('forecast_objects', 0) + metrics.get('observed_objects', 0)
                    gss_obj_based_matrix[i, j] = metrics.get('GSS_Obj-Based', 0)  # NUEVA MÉTRICA
                    
                    # Guardar resultados para CSV
                    result_dict = {
                        'radio_wrf': fcst_radius,
                        'radio_gpm': obs_radius,
                        'threshold': threshold,
                        'GSS': gss_matrix[i, j],
                        'MMI': mmi_matrix[i, j],
                        'GSS_Obj_Based': gss_obj_based_matrix[i, j],
                        'total_objects': objects_count[i, j],
                        'forecast_objects': metrics.get('forecast_objects', 0),
                        'observed_objects': metrics.get('observed_objects', 0),
                        'hits': metrics.get('hits', 0),
                        'false_alarms': metrics.get('false_alarms', 0),
                        'misses': metrics.get('misses', 0)
                    }
                    all_results.append(result_dict)
                    
                    print(f"  ✓ GSS: {gss_matrix[i, j]:.3f}, MMI: {mmi_matrix[i, j]:.3f}, "
                          f"GSS_Obj: {gss_obj_based_matrix[i, j]:.3f}, Objetos: {objects_count[i, j]}")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    gss_matrix[i, j] = np.nan
                    mmi_matrix[i, j] = np.nan
                    objects_count[i, j] = np.nan
                    gss_obj_based_matrix[i, j] = np.nan
        
        # Guardar resultados en CSV
        self._save_results_to_csv(all_results, conv_radio_range, thresholds_range)
        
        # Crear visualización (AHORA CON 4 HEATMAPS)
        print("\nGenerando visualizaciones...")
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Heatmap de GSS
        im1 = axes[0, 0].imshow(gss_matrix, cmap='viridis', aspect='auto', origin='lower',
                               extent=[min(thresholds_range), max(thresholds_range), 
                                       min(conv_radio_range['fcst']), max(conv_radio_range['fcst'])])
        axes[0, 0].set_xlabel('Umbral de Precipitación (mm/h)')
        axes[0, 0].set_ylabel('Radio de Convolución WRF (píxeles)')
        axes[0, 0].set_title('Gilbert Skill Score (GSS)')
        plt.colorbar(im1, ax=axes[0, 0], label='GSS')
        
        # 2. Heatmap de MMI
        im2 = axes[0, 1].imshow(mmi_matrix, cmap='plasma', aspect='auto', origin='lower',
                               extent=[min(thresholds_range), max(thresholds_range), 
                                       min(conv_radio_range['fcst']), max(conv_radio_range['fcst'])])
        axes[0, 1].set_xlabel('Umbral de Precipitación (mm/h)')
        axes[0, 1].set_ylabel('Radio de Convolución WRF (píxeles)')
        axes[0, 1].set_title('Mediana del Máximo Interés (MMI)')
        plt.colorbar(im2, ax=axes[0, 1], label='MMI')
        
        # 3. Heatmap de número de objetos
        im3 = axes[1, 0].imshow(objects_count, cmap='RdYlGn', aspect='auto', origin='lower',
                               extent=[min(thresholds_range), max(thresholds_range), 
                                       min(conv_radio_range['fcst']), max(conv_radio_range['fcst'])])
        axes[1, 0].set_xlabel('Umbral de Precipitación (mm/h)')
        axes[1, 0].set_ylabel('Radio de Convolución WRF (píxeles)')
        axes[1, 0].set_title('Número Total de Objetos')
        plt.colorbar(im3, ax=axes[1, 0], label='Número de Objetos')
        
        # 4. Heatmap de GSS_Obj-Based 
        im4 = axes[1, 1].imshow(gss_obj_based_matrix, cmap='coolwarm', aspect='auto', origin='lower',
                               extent=[min(thresholds_range), max(thresholds_range), 
                                       min(conv_radio_range['fcst']), max(conv_radio_range['fcst'])])
        axes[1, 1].set_xlabel('Umbral de Precipitación (mm/h)')
        axes[1, 1].set_ylabel('Radio de Convolución WRF (píxeles)')
        axes[1, 1].set_title('GSS Object-Based')
        plt.colorbar(im4, ax=axes[1, 1], label='GSS_Obj-Based')
        
        plt.suptitle('Sensibilidad de Métricas MODE a Parámetros\n')
                    #f'Radios GPM: {conv_radio_range["obs"]}', fontsize=16)
        
        # Guardar gráfico
        sensitivity_path = os.path.join(config.path_statistics, "MODE_parameter_sensitivity.png")
        plt.savefig(sensitivity_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {sensitivity_path}")
        
        plt.tight_layout()
        plt.show()
        
        return {
            'GSS': gss_matrix, 
            'MMI': mmi_matrix, 
            'objects': objects_count,
            'GSS_Obj_Based': gss_obj_based_matrix,
            'all_results': all_results
        }

    ##-------------------------------------------------------------------------
    ## 2. Método _save_results_to_csv (NUEVO)
    ##-------------------------------------------------------------------------
    def _save_results_to_csv(self, all_results, conv_radio_range, thresholds_range):
        """Guarda todos los resultados en un archivo CSV"""
        
        # Crear DataFrame con todos los resultados
        df = pd.DataFrame(all_results)
        
        # Encontrar mejores combinaciones
        best_gss_idx = df['GSS'].idxmax()
        best_mmi_idx = df['MMI'].idxmax()
        best_gss_obj_idx = df['GSS_Obj_Based'].idxmax()
        
        # Crear resumen de mejores parámetros
        summary_data = {
            'Métrica': ['Mejor GSS', 'Mejor MMI', 'Mejor GSS_Obj-Based'],
            'Radio_WRF': [
                df.loc[best_gss_idx, 'radio_wrf'],
                df.loc[best_mmi_idx, 'radio_wrf'],
                df.loc[best_gss_obj_idx, 'radio_wrf']
            ],
            'Radio_GPM': [
                df.loc[best_gss_idx, 'radio_gpm'],
                df.loc[best_mmi_idx, 'radio_gpm'],
                df.loc[best_gss_obj_idx, 'radio_gpm']
            ],
            'Umbral': [
                df.loc[best_gss_idx, 'threshold'],
                df.loc[best_mmi_idx, 'threshold'],
                df.loc[best_gss_obj_idx, 'threshold']
            ],
            'Valor': [
                df.loc[best_gss_idx, 'GSS'],
                df.loc[best_mmi_idx, 'MMI'],
                df.loc[best_gss_obj_idx, 'GSS_Obj_Based']
            ],
            'Total_Objetos': [
                df.loc[best_gss_idx, 'total_objects'],
                df.loc[best_mmi_idx, 'total_objects'],
                df.loc[best_gss_obj_idx, 'total_objects']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Guardar archivos CSV
        results_csv_path = os.path.join(config.path_statistics, "MODE_sensitivity_results.csv")
        summary_csv_path = os.path.join(config.path_statistics, "MODE_best_parameters.csv")
        
        df.to_csv(results_csv_path, index=False, encoding='utf-8')
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        
        print(f"✓ Resultados completos guardados en: {results_csv_path}")
        print(f"✓ Mejores parámetros guardados en: {summary_csv_path}")
        
        # Mostrar resumen en consola
        print("\n" + "="*60)
        print("MEJORES COMBINACIONES DE PARÁMETROS")
        print("="*60)
        print(summary_df.to_string(index=False))
        print("="*60)

    ##-------------------------------------------------------------------------
    ## 3. Método run_sensitivity_analysis
    ##-------------------------------------------------------------------------
    def run_sensitivity_analysis(self, 
                               conv_radio_range=None,
                               thresholds_range=None):
        """
        Ejecuta el análisis de sensibilidad completo
        """
        if conv_radio_range is None:
            conv_radio_range = {
                'fcst': [6, 8, 10, 12, 14, 16],
                'obs': [3, 4, 5, 6, 7, 8]
            }
        
        if thresholds_range is None:
            #thresholds_range = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
            thresholds_range = [10.0, 15.0, 20.0]
        # Ejecutar análisis de sensibilidad
        results = self.plot_parameter_sensitivity(
            conv_radio_range=conv_radio_range,
            thresholds_range=thresholds_range
        )
        
        print("="*60)
        print("ANÁLISIS DE SENSIBILIDAD COMPLETADO EXITOSAMENTE!")
        print("="*60)
        
        return results

def main():
    """Función principal para análisis de sensibilidad independiente"""
    
    print("="*60)
    print("ANÁLISIS DE SENSIBILIDAD MODE - EJECUCIÓN INDEPENDIENTE")
    print("="*60)
    print("Este script cargará los datos GPM y WRF automáticamente")
    print("Incluye GSS_Obj-Based y guarda resultados en CSV")
    print("="*60)
    
    try:
        # Crear analizador - cargará datos automáticamente
        analyzer = MODESensitivityAnalyzer()
        
        # Ejecutar análisis completo
        results = analyzer.run_sensitivity_analysis()
        
    except Exception as e:
        print(f"Error ejecutando análisis de sensibilidad: {e}")

if __name__ == "__main__":
    main()
