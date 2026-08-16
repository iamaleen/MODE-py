#=============================================================================
# statistical_analysis.py
#=============================================================================
# Advanced Statistical Diagnostics Module for MODE-py
# This module provides in-depth statistical analysis of verified objects.
# It calculates precipitation intensity quantiles (Q1, Q2, Q3) within 
# matched objects, evaluates temporal persistence and duration correlations, 
# and generates comprehensive diagnostic plots for scientific interpretation.
#
# Note: Requires a pre-computed MODE3DVerifier instance or a saved .pkl file.
#=============================================================================


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


##-----------------------------------------------------------------------------
## MODEStatisticalAnalyzer Class for MODE-py statistical analysis
##-----------------------------------------------------------------------------
class MODEStatisticalAnalyzer:
    """
    Standalone class for statistical analysis of MODE-py results
    Can work with existing MODE3DVerifier objects or load saved results.
    """
    
    def __init__(self, verifier: Optional[MODE3DVerifier] = None, results_path: Optional[str] = None):

        if verifier:
            self.verifier = verifier
            self.results_loaded = True
        elif results_path:
            self.load_verification_results(results_path)
        else:
            raise ValueError("You must provide a validator or a results path.")
    
    def load_verification_results(self, results_path: str):
        """Load verification results from file"""

        try:
            with open(results_path, 'rb') as f:
                self.verifier = pickle.load(f)
            self.results_loaded = True
            print(f"Results loaded from: {results_path}")
        except Exception as e:
            raise ValueError(f"Error loading results: {e}")
    
    def save_verification_results(self, results_path: str):
        """Save verification results for later analysis."""

        if hasattr(self, 'verifier'):
            with open(results_path, 'wb') as f:
                pickle.dump(self.verifier, f)
            print(f"Results saved to: {results_path}")
    
    ##-------------------------------------------------------------------------
    ## 1._calculate_object_intensity method
    ##-------------------------------------------------------------------------
    def _calculate_object_intensity(self, time_obj: Dict, obj_type: str):
        """Calculate the intensity values ​​for a 2D object."""

        coords = time_obj['coords_pixel']
        time_idx = time_obj['time_idx']
        
        if obj_type == 'forecast':
            intensity_data = self.verifier.smoothed_fcst.isel(time=time_idx).values
        else:
            intensity_data = self.verifier.smoothed_obs.isel(time=time_idx).values
        
        # Extract intensity values ​​for all object coordinates
        intensity_vals = []
        for y, x in coords:
            y_idx = self._safe_index(y, intensity_data.shape[0])
            x_idx = self._safe_index(x, intensity_data.shape[1])
            intensity_vals.append(intensity_data[y_idx, x_idx])
        
        time_obj['intensity_values'] = intensity_vals 

    def _safe_index(self, value: float, max_size: int) -> int:
        """Safely converts a value to an index."""
        return min(max(0, int(round(value))), max_size-1)

    ##-------------------------------------------------------------------------
    ## 2.analyze_precipitation_quantiles method
    ##------------------------------------------------------------------------- 
    def analyze_precipitation_quantiles(self):
        """Analyze the distribution of precipitation by quartiles within the objects."""
        
        self.quantile_results = {
            'forecast': {'Q1': [], 'Q2': [], 'Q3': [], 'max': []},
            'observed': {'Q1': [], 'Q2': [], 'Q3': [], 'max': []},
            'matched_pairs': []
        }
        
        # Analyze individual objects
        for obj_type in ['forecast', 'observed']:
            objects = getattr(self.verifier, f'{obj_type}_objects')
            
            for obj in objects:
                intensities = []
                for time_obj in obj['objects_2d']:
                    # Get intensities of all pixels of the object
                    if 'intensity_values' not in time_obj:
                        self._calculate_object_intensity(time_obj, obj_type)
                    
                    intensities.extend(time_obj['intensity_values'])
                
                if intensities:
                    q1, q2, q3 = np.percentile(intensities, [25, 50, 75])
                    self.quantile_results[obj_type]['Q1'].append(q1)
                    self.quantile_results[obj_type]['Q2'].append(q2)
                    self.quantile_results[obj_type]['Q3'].append(q3)
                    self.quantile_results[obj_type]['max'].append(max(intensities))
        
        # Analyze paired samples
        for match in self.verifier.matches:
            fcst_obj = next(o for o in self.verifier.forecast_objects if o['id'] == match['forecast_id'])
            obs_obj = next(o for o in self.verifier.observed_objects if o['id'] == match['observed_id'])
            
            # Calculate quartile differences between paired objects
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
    ## 3.plot_quantile_analysis method
    ##-------------------------------------------------------------------------
    def plot_quantile_analysis(self, save_path: str = None):
        """Visualize the quartile analysis in combination charts."""
        
        if not hasattr(self, 'quantile_results'):
            self.analyze_precipitation_quantiles()
        
        if save_path is None:
            save_path = config.path_statistics
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Quartile boxplot by object type
        data_to_plot = [self.quantile_results['forecast']['Q1'], self.quantile_results['observed']['Q1'],
                       self.quantile_results['forecast']['Q2'], self.quantile_results['observed']['Q2'],
                       self.quantile_results['forecast']['Q3'], self.quantile_results['observed']['Q3']]
        
        axes[0, 0].boxplot(data_to_plot, positions=[1, 2, 4, 5, 7, 8])
        axes[0, 0].set_xticks([1.5, 4.5, 7.5])
        axes[0, 0].set_xticklabels(['Q1', 'Q2', 'Q3'])
        axes[0, 0].set_title('Quartile Distribution by Object Type')
        axes[0, 0].set_ylabel('Precipitation (mm)')
        axes[0, 0].legend(['WRF', 'GPM'], loc='upper right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Quartile differences in matched pairs
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
            axes[0, 1].set_xlabel('Match Interest')
            axes[0, 1].set_ylabel('Difference (WRF - GPM)')
            axes[0, 1].set_title('Quartile Differences and Match Interest')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution of differences
        if self.quantile_results['matched_pairs']:
            axes[1, 0].hist(q1_diffs, alpha=0.5, label='Q1', bins=20, edgecolor='black')
            axes[1, 0].hist(q2_diffs, alpha=0.5, label='Q2', bins=20, edgecolor='black')
            axes[1, 0].hist(q3_diffs, alpha=0.5, label='Q3', bins=20, edgecolor='black')
            axes[1, 0].axvline(x=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Difference (WRF - GPM)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Differences by Quartile')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Relationship between the difference of maxima and area
        if self.quantile_results['matched_pairs']:
            max_diffs = [p['max_diff'] for p in pairs]
            area_ratios = [p['area_ratio'] for p in pairs]
            
            scatter = axes[1, 1].scatter(area_ratios, max_diffs, c=interest_values, 
                                       cmap='viridis', alpha=0.7, s=50)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].axvline(x=1, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Area Ratio (WRF/GPM)')
            axes[1, 1].set_ylabel('Difference of Maxima (WRF - GPM)')
            axes[1, 1].set_title('Difference of Maxima and Area Ratio')
            plt.colorbar(scatter, ax=axes[1, 1], label='Match Interest')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'quantile_analysis_combined.png'), 
                   dpi=300, bbox_inches='tight')


    ##-------------------------------------------------------------------------
    ## 4.plot_individual_quantile_analysis method
    ##-------------------------------------------------------------------------
    def plot_individual_quantile_analysis(self, save_path: str = None):
        """View the quartile analysis in separate charts."""
        
        if not hasattr(self, 'quantile_results'):
            self.analyze_precipitation_quantiles()
        
        if save_path is None:
            save_path = config.path_statistics
        
        # Create directory if it does not exist
        os.makedirs(save_path, exist_ok=True)
        
        # Chart 1: Quartile boxplot by object type
        plt.figure(figsize=(10, 6))
        data_to_plot = [self.quantile_results['forecast']['Q1'], self.quantile_results['observed']['Q1'],
                       self.quantile_results['forecast']['Q2'], self.quantile_results['observed']['Q2'],
                       self.quantile_results['forecast']['Q3'], self.quantile_results['observed']['Q3']]
        
        plt.boxplot(data_to_plot, positions=[1, 2, 4, 5, 7, 8])
        plt.xticks([1.5, 4.5, 7.5], ['Q1', 'Q2', 'Q3'])
        plt.title('Quartile Distribution by Object Type')
        plt.ylabel('Precipitation (mm)')
        plt.legend(['WRF', 'GPM'], loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'quantile_boxplot.png'), dpi=300, bbox_inches='tight')

        # Chart 2: Quartile differences in matched pairs
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
            plt.xlabel('Match Interest')
            plt.ylabel('Difference (WRF - GPM)')
            plt.title('Quartile Differences and Match Interest')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'quantile_differences_scatter.png'), 
                       dpi=300, bbox_inches='tight')
            
        
        # Graph 3: Relationship between the difference in maxima and area
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
            plt.xlabel('Area Ratio (WRF/GPM)')
            plt.ylabel('Difference of Maxima (WRF - GPM)')
            plt.title('Difference of Maxima and Area Ratio')
            plt.colorbar(scatter, label='Match Interest')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'max_difference_vs_area.png'), 
                       dpi=300, bbox_inches='tight')
  
        
        # Chart 4: Distribution of differences
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
            plt.xlabel('Difference (WRF - GPM)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Differences by Quartile')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'differences_distribution.png'), 
                       dpi=300, bbox_inches='tight')


    ##-------------------------------------------------------------------------
    ## 5.analyze_temporal_persistence method
    ##-------------------------------------------------------------------------
    def analyze_temporal_persistence(self):
        """Analyzes the temporal persistence of objects."""
        
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
        
        # Calculate duration correlation for matched pairs
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
    ## 6.plot_temporal_persistence method
    ##-------------------------------------------------------------------------
    def plot_temporal_persistence(self, save_path: str = None):
        """Visualize the temporal persistence of objects."""
        
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
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Boxplot of durations
        data = [fcst_durations, obs_durations, 
                matched_fcst_durations, matched_obs_durations,
                unmatched_fcst_durations, unmatched_obs_durations]
        
        axes[0].boxplot(data)
        axes[0].set_xticks([1, 2, 3, 4, 5, 6])
        axes[0].set_xticklabels(['WRF Total', 'GPM Total', 'WRF Matched', 'GPM Matched', 
                               'WRF Unmatched', 'GPM Unmatched'], rotation=45)
        axes[0].set_ylabel('Duration (time steps)')
        axes[0].set_title('Temporal Object Persistence')
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot: duration and interest for paired objects
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
            axes[1].set_xlabel('WRF Duration (time steps)')
            axes[1].set_ylabel('Duración GPM (time steps)')
            axes[1].set_title('Duration of Paired Pairs')
            plt.colorbar(scatter, ax=axes[1], label='Interest')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'temporal_persistence.png'), dpi=300, bbox_inches='tight')


    ##-------------------------------------------------------------------------
    ## 7.run_complete_statistical_analysis method
    ##-------------------------------------------------------------------------
    def run_complete_statistical_analysis(self, save_path: str = None):
        """Run the complete statistical analysis with all visualizations."""
        
        if save_path is None:
            save_path = config.path_statistics
        
        print("="*60)
        print("COMPREHENSIVE STATISTICAL ANALYSIS OF MODE-py")
        print("="*60)
        
        # Run advanced analyses
        print("\n1. Running quartile analysis...")
        quantile_results = self.analyze_precipitation_quantiles()
        
        print("2. Running temporal persistence analysis...")
        persistence_metrics = self.analyze_temporal_persistence()
        
        # Generate visualizations
        print("3. Generating visualizations...")
        self.plot_quantile_analysis(save_path)
        self.plot_individual_quantile_analysis(save_path)
        self.plot_temporal_persistence(save_path)
        
        # Show summary metrics
        print("\n" + "="*60)
        print("SUMMARY STATISTICAL METRICS")
        print("="*60)
        
        print(f"\nQuartile Analysis:")
        print(f"Analyzed WRF objects: {len(quantile_results['forecast']['Q1'])}")
        print(f"GPM objects analyzed: {len(quantile_results['observed']['Q1'])}")
        print(f"Matched pairs analyzed: {len(quantile_results['matched_pairs'])}")
        
        if quantile_results['matched_pairs']:
            q1_diffs = [p['q1_diff'] for p in quantile_results['matched_pairs']]
            q2_diffs = [p['q2_diff'] for p in quantile_results['matched_pairs']]
            q3_diffs = [p['q3_diff'] for p in quantile_results['matched_pairs']]
            
            print(f"Mean difference Q1 (WRF-GPM): {np.mean(q1_diffs):.3f} mm")
            print(f"Mean difference Q2 (WRF-GPM): {np.mean(q2_diffs):.3f} mm")
            print(f"Mean difference Q3 (WRF-GPM): {np.mean(q3_diffs):.3f} mm")
        
        print(f"\nTemporal Persistence:")
        print(f"Average duration of WRF objects: {persistence_metrics['fcst_mean_duration']:.2f} steps")
        print(f"Average duration of GPM objects: {persistence_metrics['obs_mean_duration']:.2f} steps")
        print(f"Average duration of paired WRFs: {persistence_metrics['matched_fcst_mean_duration']:.2f} steps")
        print(f"Average duration of paired GPMs: {persistence_metrics['matched_obs_mean_duration']:.2f} steps")

        print(f"\nResults saved in: {save_path}")
        print("Statistical analysis completed!")

def main():
    """Main function for independent statistical analysis"""
    
    # Load saved results
    analyzer = MODEStatisticalAnalyzer(results_path=config.path_statistics + 'mode_verification_results.pkl')

    analyzer.run_complete_statistical_analysis()

if __name__ == "__main__":
    main()
