# MODE-py: Spatio-Temporal Object-Based Verification Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: XXX](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/XXX)

**MODE-py** is a modular, extensible, and reproducible Python framework for the spatio-temporal object-based verification of high-resolution precipitation forecasts. It implements the Method for Object-based Diagnostic Evaluation (MODE) with specific extensions to handle heterogeneous spatial resolutions (e.g., 3 km WRF vs. 10 km GPM IMERG) and temporal persistence analysis.

This repository contains the source code, synthetic benchmark suites, and documentation associated with the manuscript: 
*"MODE-py: A Python framework for spatio-temporal object-based verification of high resolution precipitation forecasts"* (Submitted to *Computers & Geosciences*).

## Key Features

*   **Adaptive Multi-Resolution Preprocessing:** Handles different grid resolutions and flexible temporal accumulation windows (1H, 3H, 6H).
*   **Spatio-Temporal Graph Grouping:** Transforms isolated 2D objects into persistent 3D entities using graph-based connectivity and trajectory tracking.
*   **Fuzzy Logic Matching:** Calculates a composite interest function (distance, area, overlap, orientation, temporal persistence) and resolves assignments via a greedy matching algorithm.
*   **Advanced Metrics:** Computes Median of Maximum Interest (MMI), classical Grid-Point Gilbert Skill Score (GSS), and Object-Based GSS.
*   **Integrated Sensitivity Analysis:** Automates parametric sweeps (convolution radii, thresholds) and generates diagnostic heatmaps.
*   **Synthetic Benchmark Suite:** Includes static (geometric perturbations) and dynamic (splitting, merging, translation) test cases for controlled algorithm validation.

## Requirements & Installation

MODE-py is developed in **Python 3.10+**. It is highly recommended to use a virtual environment.


## Repository Structure

MODE_Verification/
|-- config.py                  # Centralized configuration (paths, parameters, weights)

|-- data_loader_.py            # Ingestion of GPM (HDF5) and WRF (NetCDF) data

|-- preprocessor_.py           # Temporal alignment and spatial cropping

|-- mode_verifier.py           # Core algorithmic implementation (MODE3DVerifier class)

|-- field_visualization.py     # Geospatial plotting utilities (Cartopy/Matplotlib)

|-- statistical_analysis.py    # Quartile analysis and temporal persistence diagnostics

|-- sensitivity_analysis.py    # Parametric sweeps and heatmap generation

|-- run_mode_verification.py   # Main execution pipeline

    synthetic_benchmark/
    |-- synthetic_generator.py     # Generation of controlled geometric/temporal perturbations

    |-- synthetic_visualization.py # Plotting tools for synthetic cases

    |-- run_synthetic_benchmark.py # Execution script for the benchmark suite


```bash
# Clone the repository
git clone https://github.com/iamaleen/MODE-py.git
cd MODE-py

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt



