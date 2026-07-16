#!/usr/bin/env python3
##-----------------------------------------------------------------------------
## Script para ejecutar análisis de sensibilidad independiente
##-----------------------------------------------------------------------------

# Incluye GSS_Obj-Based en heatmaps
# Genera archivos CSV con resultados completos
# Guarda mejores parámetros en CSV separado

from sensitivity_analysis import MODESensitivityAnalyzer

def main():
    """Función principal - ejecuta análisis de sensibilidad independiente"""
    
    print("="*60)
    print("ANÁLISIS DE SENSIBILIDAD MODE")
    print("="*60)
  
    print("="*60)
    print("Cargando datos GPM y WRF automáticamente...")
    print("="*60)
    
    try:
        # Crear analizador - cargará datos automáticamente
        analyzer = MODESensitivityAnalyzer()
        
        # Ejecutar análisis completo
        print("Iniciando análisis de sensibilidad...")
        results = analyzer.run_sensitivity_analysis()
        
        print("\n" + "="*60)
        print("¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
        print("="*60)
        print("Archivos generados:")
        print("MODE_parameter_sensitivity.png")
        print("MODE_sensitivity_results.csv")
        print("MODE_best_parameters.csv")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPosibles soluciones:")
        print("1. Verifique que los datos GPM y WRF estén en las rutas correctas")
        print("2. Verifique los archivos de configuración en config.py")

if __name__ == "__main__":
    main()
