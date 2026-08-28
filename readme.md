# DRO in Portfolio Optimization

This repository compares different portfolio optimization strategies, focusing on a Second-Order Cone Programming (SOCP) approach. The main SOCP model integrates distributional robustness using Wasserstein ambiguity sets while accounting for transaction and holding costs.

## Included Models
The project evaluates six different models:
* **SOCP**: Our proposed model with robustness and trading frictions.
* **Classic Mean-Variance (MV)**: The standard Markowitz approach.
* **Classic Mean-Absolute Deviation (MAD)**: A linear risk-measure alternative.
* **Moehle**: A model focused on trading frictions and turnover.
* **DRO-MAD**: Distributionally robust version of the MAD model.
* **1/N**: A simple equally-weighted benchmark.

## Setup and Configuration
Details regarding configurations, solver settings, and the specific implementations for each model can be found within the project directory.