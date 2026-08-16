#=============================================================================
# run_sensitivity_analysis.py
#=============================================================================
# Standalone Execution Script for Sensitivity Analysis
# This script provides a direct entry point to run the parameter sensitivity 
# analysis independently of the full MODE-py verification workflow. It is 
# optimized for iterative testing of different parameter spaces.
#
# Usage:
#   python run_sensitivity_analysis.py
#=============================================================================

from sensitivity_analysis import MODESensitivityAnalyzer

def main():
    """Main function. Performs independent sensitivity analysis."""
    
    print("="*60)
    print("MODE-py SENSITIVITY ANALYSIS")
    print("="*60)
  
    try:
        # Create parser
        analyzer = MODESensitivityAnalyzer()
        
        # Run full scan
        results = analyzer.run_sensitivity_analysis()
        
        print("\n" + "="*60)
        print("ANALYSIS SUCESSFULLY COMPLETED!")
        print("="*60)
        print("Generated files:")
        print("MODE_parameter_sensitivity.png")
        print("MODE_sensitivity_results.csv")
        print("MODE_best_parameters.csv")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("1. Verify that the GPM and WRF data are in the correct paths.")
        print("2. 2. Check the configuration files in config.py")

if __name__ == "__main__":
    main()
