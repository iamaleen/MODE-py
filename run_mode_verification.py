#=============================================================================
# run_mode_verification.py
#=============================================================================
# Main Execution for MODE-py
# This script orchestrates the complete verification workflow. It loads 
# forcasted and observed data, preprocesses and aligns the datasets, executes the 
# MODE3DVerifier, generates diagnostic visualizations, and saves the 
# evaluation metrics to csv.
#
# Usage:
#   python run_mode_verification.py                    # Run full verification
#   python run_mode_verification.py --sensitivity      # Verification plus Sensitivity
#   python run_mode_verification.py --sensitivity-only # Run sensitivity analysis only
#=============================================================================

import warnings
warnings.filterwarnings('ignore')

from data_loader_ import load_gpm_data, load_wrf_data
from preprocessor_ import preprocess_datasets
from field_visualization import generate_comparison_plots
from mode_verifier import MODE3DVerifier
import config
import pickle 
import sys

from sensitivity_analysis import MODESensitivityAnalyzer

def run_sensitivity_only():
    """Run only the sensitivity analysis, without full MODE-py verification."""

    print("="*60)
    print("RUNNING SENSITIVITY ANALYSIS ONLY")
    print("="*60)

    # Load minimum data required for sensitivity
    print("\n1. Loading data for sensitivity...")
    ds_gpm_hourly = load_gpm_data()
    ds_wrf = load_wrf_data()
    ds_observed, ds_model = preprocess_datasets(ds_gpm_hourly, ds_wrf)
    
    # Create analyzer with direct data
    sensitivity_analyzer = MODESensitivityAnalyzer(
        forecast_data=ds_model, 
        observed_data=ds_observed
    )
    
    # Run sensitivity analysis
    print("\n2. Running sensitivity analysis...")
    results = sensitivity_analyzer.run_sensitivity_analysis()
      
    return results

def run_full_verification():
    """Run the MODE-py verification."""

    print("="*60)
    print("COMPLETE MODE-py VERIFICATION")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    ds_gpm_hourly = load_gpm_data()
    ds_wrf = load_wrf_data()
    
    # Preprocess and align
    print("\n2. Preprocessig and aligning data...")
    ds_observed, ds_model = preprocess_datasets(ds_gpm_hourly, ds_wrf)
    
    # Generate comparison charts
    print("\n3. Generating comparison charts...")
    generate_comparison_plots(ds_observed, ds_model, max_plots=25)
    
    # Run MODE-py verification
    print("\n4. Running MODE-py...")
    
    # Create a copy of parameters 
    init_params = config.MODE_PARAMS.copy()
    interest_threshold = init_params.pop('interest_threshold')
    
    verifier = MODE3DVerifier(
        forecast=ds_model,
        observed=ds_observed,
        **init_params
    )
    
    metrics = verifier.run_verification(interest_threshold=interest_threshold)

    # Save as csv
    csv_path = verifier.save_metrics_to_csv()
    
    # Show results
    print("\nMODE-py Evaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>25}: {v:.3f}")
        else:
            print(f"{k:>25}: {v}")
    
    # View results
    print("\n5. Generating MODE-py graphs...")
    for time_idx in range(min(25, len(ds_model.time))):
        verifier.plot_matched_objects(time_idx=time_idx)
    
    # Save results
    print("\n6. Saving results...")
    results_path = config.path_statistics + 'mode_verification_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(verifier, f)
    print(f"Results saved in: {results_path}")
    
    print("\n" + "="*60)
    print("MODE-py VERIFICATION SUCCESFULLY COMPLETED!")
    print("="*60)
    
    return verifier

def run_verification_with_sensitivity():
    """Performs a full MODE-py verification plus sensitivity analysis."""

    print("="*60)
    print("FULL MODE-py VERIFICATION & SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Run a full check 
    verifier = run_full_verification()
    
    # Run sensitivity
    print("\n7. Running sensitivity analysis...")
    
    
    sensitivity_analyzer = MODESensitivityAnalyzer(verifier=verifier)
    results = sensitivity_analyzer.run_sensitivity_analysis()
    
    print("\n" + "="*60)
    print("ENTIRE PROCESS SUCCESSFULLY COMPLETED!")
    print("="*60)
    
    return verifier, results

def main():
    """Main function."""
    
    # Verify arguments
    if '--sensitivity-only' in sys.argv:
        return run_sensitivity_only()
    
    elif '--sensitivity' in sys.argv:
        return run_verification_with_sensitivity()
    
    else:
        return run_full_verification()

if __name__ == "__main__":
    result = main()
