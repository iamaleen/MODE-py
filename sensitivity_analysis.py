#=============================================================================
# sensitivity_analysis.py
#=============================================================================
# Parameter Sensitivity Analysis Module for MODE-py
# This independent module evaluates the impact of user-defined parameters 
# (e.g., convolution radii, precipitation thresholds) on verification metrics.
# It performs parametric sweeps, generates 2D sensitivity heatmaps, and 
# exports the optimal parameter configurations to csv files.
#
# Note: Can be executed standalone or integrated into the main workflow.
#=============================================================================


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from tqdm import tqdm
from typing import Dict, List, Optional
import sys
import os

# Add the path to allow importing the other modules
sys.path.append(os.path.dirname(__file__))

from data_loader_ import load_gpm_data, load_wrf_data
from preprocessor_ import preprocess_datasets
from mode_verifier import MODE3DVerifier
import config


##-----------------------------------------------------------------------------
## MODESensitivityAnalyzer Class for MODE-py parameter sensitivity analysis
##-----------------------------------------------------------------------------
class MODESensitivityAnalyzer:
    """Standalone class for MODE-py parameter sensitivity analysis."""
    
    def __init__(self, verifier: Optional[MODE3DVerifier] = None, 
                 results_path: Optional[str] = None,
                 forecast_data: Optional = None,
                 observed_data: Optional = None):
      

        # Initializes the sensitivity analyzer in multiple ways
        # Using existing MODE-py checker
        if verifier:
            self.verifier = verifier
            self.forecast = verifier.forecast
            self.observed = verifier.observed
            self.data_loaded = True
            
        elif results_path:
            self.load_verification_results(results_path)
            
        # Using input data directly
        elif forecast_data is not None and observed_data is not None:
            self.forecast = forecast_data
            self.observed = observed_data
            self.data_loaded = True
        else:
            self.load_fresh_data()
    
    def load_fresh_data(self):
        """Load GPM and WRF data"""
        try:
            ds_gpm_hourly = load_gpm_data()
            ds_wrf = load_wrf_data()
            
            self.observed, self.forecast = preprocess_datasets(ds_gpm_hourly, ds_wrf)
            self.data_loaded = True
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")
    
    def load_verification_results(self, results_path: str):
        """Load verification results from a .pkl file."""
        try:
            with open(results_path, 'rb') as f:
                self.verifier = pickle.load(f)
            self.forecast = self.verifier.forecast
            self.observed = self.verifier.observed
            self.data_loaded = True
            
        except Exception as e:
            print(f"Error loading .pkl results: {e}")
            self.load_fresh_data()
    
    ##-------------------------------------------------------------------------
    ## 1. plot_parameter_sensitivity method 
    ##-------------------------------------------------------------------------
    def plot_parameter_sensitivity(self, 
                                 conv_radio_range=None, 
                                 thresholds_range=None,
                                 figsize=(16, 12), 
                                 cmap='coolwarm'):
        """Visualize the sensitivity of the metrics to different parameters. """

        # Verify that the data is loaded
        if not hasattr(self, 'data_loaded') or not self.data_loaded:
            self.load_fresh_data()
        
        # Default values
        if conv_radio_range is None:
            conv_radio_range = {
                'fcst': [6, 8, 10, 12, 14, 16],
                'obs': [3, 4, 5, 6, 7, 8]
            }
        
        if thresholds_range is None:
            #thresholds_range = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
            thresholds_range = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
        
        print("="*60)
        print("CALCULATING PARAMETER SENSITIVITY...")
        print("="*60)
        print(f"WRF radii: {conv_radio_range['fcst']}")
        print(f"GPM radii: {conv_radio_range['obs']}")
        print(f"Thresholds: {thresholds_range}")
        
        # Matrices for storing results
        n_radio = len(conv_radio_range['fcst'])
        n_thresholds = len(thresholds_range)
        
        gss_matrix = np.zeros((n_radio, n_thresholds))
        mmi_matrix = np.zeros((n_radio, n_thresholds))
        objects_count = np.zeros((n_radio, n_thresholds))
        gss_obj_based_matrix = np.zeros((n_radio, n_thresholds))  
       
        all_results = []
        
        # Run MODE-py for each parameter combination
        total_combinations = n_radio * n_thresholds
        current_combination = 0
        
        print(f"\nEvaluating {total_combinations} parameter combinations...")
        
        for i, fcst_radius in enumerate(conv_radio_range['fcst']):
            for j, threshold in enumerate(thresholds_range):
                current_combination += 1
                print(f"Processing combination {current_combination}/{total_combinations}: "
                      f"WRF_radii={fcst_radius}, threshold={threshold}mm/h")
                
                try:
                    # Use proportional radius for GPM
                    obs_radius = conv_radio_range['obs'][i] if i < len(conv_radio_range['obs']) else conv_radio_range['obs'][-1]
                    
                    # Create a new instance with different parameters
                    temp_verifier = MODE3DVerifier(
                        forecast=self.forecast,
                        observed=self.observed,
                        threshold=threshold,
                        conv_radius_forecast=fcst_radius,
                        conv_radius_observed=obs_radius,
                        time_window=1,
                        min_object_size=15
                    )
                    
                    # Run verification
                    metrics = temp_verifier.run_verification(interest_threshold=0.6)
                    
                    # Store results in arrays
                    gss_matrix[i, j] = metrics.get('GSS', 0)
                    mmi_matrix[i, j] = metrics.get('MMI', 0)
                    objects_count[i, j] = metrics.get('forecast_objects', 0) + metrics.get('observed_objects', 0)
                    gss_obj_based_matrix[i, j] = metrics.get('GSS_Obj-Based', 0)  
                    
                    # Save results to csv
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
                    
                    print(f" GSS: {gss_matrix[i, j]:.3f}, MMI: {mmi_matrix[i, j]:.3f}, "
                          f"GSS_Obj: {gss_obj_based_matrix[i, j]:.3f}, Objetos: {objects_count[i, j]}")
                    
                except Exception as e:
                    print(f"Error: {e}")
                    gss_matrix[i, j] = np.nan
                    mmi_matrix[i, j] = np.nan
                    objects_count[i, j] = np.nan
                    gss_obj_based_matrix[i, j] = np.nan
        
        # Save results to csv
        self._save_results_to_csv(all_results, conv_radio_range, thresholds_range)
        
        # Create visualization 
        print("\nGenerating visualizations...")
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. GSS Heatmap
        im1 = axes[0, 0].imshow(gss_matrix, cmap='viridis', aspect='auto', origin='lower',
                               extent=[min(thresholds_range), max(thresholds_range), 
                                       min(conv_radio_range['fcst']), max(conv_radio_range['fcst'])])
        axes[0, 0].set_xlabel('Precipitation Threshold (mm/h)')
        axes[0, 0].set_ylabel('WRF Convolution Radius (píxeles)')
        axes[0, 0].set_title('Gilbert Skill Score (GSS)')
        plt.colorbar(im1, ax=axes[0, 0], label='GSS')
        
        # 2. MMI Heatmap
        im2 = axes[0, 1].imshow(mmi_matrix, cmap='plasma', aspect='auto', origin='lower',
                               extent=[min(thresholds_range), max(thresholds_range), 
                                       min(conv_radio_range['fcst']), max(conv_radio_range['fcst'])])
        axes[0, 1].set_xlabel('Precipitation Threshold (mm/h)')
        axes[0, 1].set_ylabel('WRF Convolution Radius (píxeles)')
        axes[0, 1].set_title('Median of Maximum Interest (MMI)')
        plt.colorbar(im2, ax=axes[0, 1], label='MMI')
        
        # 3. Heatmap of the number of objects
        im3 = axes[1, 0].imshow(objects_count, cmap='RdYlGn', aspect='auto', origin='lower',
                               extent=[min(thresholds_range), max(thresholds_range), 
                                       min(conv_radio_range['fcst']), max(conv_radio_range['fcst'])])
        axes[1, 0].set_xlabel('Precipitation Threshold (mm/h)')
        axes[1, 0].set_ylabel('WRF Convolution Radius (píxeles)')
        axes[1, 0].set_title('Total Number of Objects')
        plt.colorbar(im3, ax=axes[1, 0], label='Number of Objects')
        
        # 4. GSS_Obj-Based Heatmap 
        im4 = axes[1, 1].imshow(gss_obj_based_matrix, cmap='coolwarm', aspect='auto', origin='lower',
                               extent=[min(thresholds_range), max(thresholds_range), 
                                       min(conv_radio_range['fcst']), max(conv_radio_range['fcst'])])
        axes[1, 1].set_xlabel('Umbral de Precipitación (mm/h)')
        axes[1, 1].set_ylabel('Radio de Convolución WRF (píxeles)')
        axes[1, 1].set_title('GSS Object-Based')
        plt.colorbar(im4, ax=axes[1, 1], label='GSS_Obj-Based')
        
        plt.suptitle('Sensitivity of MODE-py Metrics to Parameters\n')

        # Guardar gráfico
        sensitivity_path = os.path.join(config.path_statistics, "MODE-py_parameter_sensitivity.png")
        plt.savefig(sensitivity_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {sensitivity_path}")
        
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
    ## 2._save_results_to_csv method 
    ##-------------------------------------------------------------------------
    def _save_results_to_csv(self, all_results, conv_radio_range, thresholds_range):
        """Save all the results to a csv file."""
        
        # Create a DataFrame with all the results.
        df = pd.DataFrame(all_results)
        
        # Find the best combinations
        best_gss_idx = df['GSS'].idxmax()
        best_mmi_idx = df['MMI'].idxmax()
        best_gss_obj_idx = df['GSS_Obj_Based'].idxmax()
        
        # Create a summary of the best parameters
        summary_data = {
            'Métrica': ['Best GSS', 'Best MMI', 'Best GSS_Obj-Based'],
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
            'Threshold': [
                df.loc[best_gss_idx, 'threshold'],
                df.loc[best_mmi_idx, 'threshold'],
                df.loc[best_gss_obj_idx, 'threshold']
            ],
            'Value': [
                df.loc[best_gss_idx, 'GSS'],
                df.loc[best_mmi_idx, 'MMI'],
                df.loc[best_gss_obj_idx, 'GSS_Obj_Based']
            ],
            'Total_Objects': [
                df.loc[best_gss_idx, 'total_objects'],
                df.loc[best_mmi_idx, 'total_objects'],
                df.loc[best_gss_obj_idx, 'total_objects']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save csv files
        results_csv_path = os.path.join(config.path_statistics, "MODE-py_sensitivity_results.csv")
        summary_csv_path = os.path.join(config.path_statistics, "MODE-py_best_parameters.csv")
        
        df.to_csv(results_csv_path, index=False, encoding='utf-8')
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        
        print(f"Full results saved to: {results_csv_path}")
        print(f"Best parameters saved to: {summary_csv_path}")
        
        # Show summary in console
        print("\n" + "="*60)
        print("BEST PARAMETER COMBINATIONS")
        print("="*60)
        print(summary_df.to_string(index=False))
        print("="*60)

    ##-------------------------------------------------------------------------
    ## 3.run_sensitivity_analysis method
    ##-------------------------------------------------------------------------
    def run_sensitivity_analysis(self, 
                               conv_radio_range=None,
                               thresholds_range=None):
        """Run the full sensitivity analysis."""

        if conv_radio_range is None:
            conv_radio_range = {
                'fcst': [6, 8, 10, 12, 14, 16],
                'obs': [3, 4, 5, 6, 7, 8]
            }
        
        if thresholds_range is None:
            thresholds_range = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]


        # Run sensitivity analysis
        results = self.plot_parameter_sensitivity(
            conv_radio_range=conv_radio_range,
            thresholds_range=thresholds_range
        )
        
        print("Sensitivity analysis successfully completed!")
        
        return results

def main():
    """Main function for independent sensitivity analysis."""
    
    print("="*60)
    print("MODE-py Sensitivity Analysis")
    print("="*60)
          
    try:
        # Create analyzer - will load data automatically
        analyzer = MODESensitivityAnalyzer()
        
        # Run full scan
        results = analyzer.run_sensitivity_analysis()
        
    except Exception as e:
        print(f"Error running sensitivity analysis: {e}")

if __name__ == "__main__":
    main()
