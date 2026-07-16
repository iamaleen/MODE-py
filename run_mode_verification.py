#!/usr/bin/env python3
##-----------------------------------------------------------------------------
## Script Principal - Verificación MODE
##-----------------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')

from data_loader_ import load_gpm_data, load_wrf_data
from preprocessor_ import preprocess_datasets
from field_visualization import generate_comparison_plots
from mode_verifier import MODE3DVerifier
import config
import pickle 
import sys

def run_sensitivity_only():
    """Ejecuta SOLO el análisis de sensibilidad, sin verificación MODE completa"""
    print("="*60)
    print("EJECUTANDO SOLO ANÁLISIS DE SENSIBILIDAD")
    print("="*60)
    
    from sensitivity_analysis import MODESensitivityAnalyzer
    
    # Cargar datos mínimos necesarios para sensibilidad
    print("\n1. CARGANDO DATOS PARA SENSIBILIDAD...")
    ds_gpm_hourly = load_gpm_data()
    ds_wrf = load_wrf_data()
    ds_observed, ds_model = preprocess_datasets(ds_gpm_hourly, ds_wrf)
    
    # Crear analizador con datos directos (más eficiente)
    sensitivity_analyzer = MODESensitivityAnalyzer(
        forecast_data=ds_model, 
        observed_data=ds_observed
    )
    
    # Ejecutar análisis de sensibilidad
    print("\n2. EJECUTANDO ANÁLISIS DE SENSIBILIDAD...")
    results = sensitivity_analyzer.run_sensitivity_analysis()
      
    return results

def run_full_verification():
    """Ejecuta la verificación MODE completa (sin sensibilidad)"""
    print("="*60)
    print("VERIFICACIÓN MODE COMPLETA - WRF vs GPM")
    print("="*60)
    
    # 1. Cargar datos
    print("\n1. CARGANDO DATOS...")
    ds_gpm_hourly = load_gpm_data()
    ds_wrf = load_wrf_data()
    
    # 2. Preprocesar y alinear
    print("\n2. PREPROCESANDO Y ALINEANDO DATOS...")
    ds_observed, ds_model = preprocess_datasets(ds_gpm_hourly, ds_wrf)
    
    # 3. Generar gráficos de comparación
    print("\n3. GENERANDO GRÁFICOS DE COMPARACIÓN...")
    generate_comparison_plots(ds_observed, ds_model, max_plots=25)
    
    # 4. Ejecutar verificación MODE
    print("\n4. EJECUTANDO VERIFICACIÓN MODE...")
    
    # Crear copia de parámetros excluyendo interest_threshold
    init_params = config.MODE_PARAMS.copy()
    interest_threshold = init_params.pop('interest_threshold')
    
    verifier = MODE3DVerifier(
        forecast=ds_model,
        observed=ds_observed,
        **init_params
    )
    
    metrics = verifier.run_verification(interest_threshold=interest_threshold)

    # Guardar en CSV
    csv_path = verifier.save_metrics_to_csv()
    
    # 5. Mostrar resultados
    print("\nMétricas de Evaluación MODE:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>25}: {v:.3f}")
        else:
            print(f"{k:>25}: {v}")
    
    # 6. Visualizar resultados MODE
    print("\n5. GENERANDO GRÁFICOS MODE...")
    for time_idx in range(min(25, len(ds_model.time))):
        verifier.plot_matched_objects(time_idx=time_idx)
    
    # 7. Guardar resultados para análisis posterior
    print("\n6. GUARDANDO RESULTADOS PARA ANÁLISIS ESTADÍSTICO...")
    results_path = config.path_statistics + 'mode_verification_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(verifier, f)
    print(f"Resultados guardados en: {results_path}")
    
    print("\n" + "="*60)
    print("VERIFICACIÓN MODE COMPLETADA EXITOSAMENTE!")
    print("="*60)
    
    return verifier

def run_verification_with_sensitivity():
    """Ejecuta verificación MODE completa + análisis de sensibilidad"""
    print("="*60)
    print("VERIFICACIÓN MODE COMPLETA + ANÁLISIS DE SENSIBILIDAD")
    print("="*60)
    
    # Primero ejecutar verificación completa
    verifier = run_full_verification()
    
    # Luego ejecutar sensibilidad
    print("\n7. EJECUTANDO ANÁLISIS DE SENSIBILIDAD...")
    from sensitivity_analysis import MODESensitivityAnalyzer
    
    sensitivity_analyzer = MODESensitivityAnalyzer(verifier=verifier)
    results = sensitivity_analyzer.run_sensitivity_analysis()
    
    print("\n" + "="*60)
    print("PROCESO COMPLETO TERMINADO EXITOSAMENTE!")
    print("="*60)
    
    return verifier, results

def main():
    """Función principal que decide qué ejecutar basado en los argumentos"""
    
    # Verificar argumentos
    if '--sensitivity-only' in sys.argv:
        # Modo 1: Solo sensibilidad
        return run_sensitivity_only()
    
    elif '--sensitivity' in sys.argv:
        # Modo 2: Verificación completa + sensibilidad
        return run_verification_with_sensitivity()
    
    else:
        # Modo 3: Solo verificación (por defecto)
        return run_full_verification()

if __name__ == "__main__":
    result = main()
