# Fairness Metrics and Analysis

This repository contains scripts and data for generating synthetic datasets, calculating fairness metrics, and analyzing temporal trends. It also includes processing scripts for real-world data from the UNSW dataset.

## Directory Structure

- **Data Generators**
  - `biased_data_generator.py`: Generates a synthetic dataset with intentional biases (full bias: alert, confirmation, and score perturbations).
  - `balanced_data_generator.py`: Generates a synthetic dataset with balanced, unbiased data.
  - `alert_only_bias_generator.py`: **(NEW)** Generates a synthetic dataset with alert-generation bias only (platform-dependent alert rates), while confirmation rates and score assignments remain uniform. Used for metric isolation analysis (targets Ï†_ind).
  - `calibration_only_bias_generator.py`: **(NEW)** Generates a synthetic dataset with calibration bias only (platform-dependent confirmation rates and score drift), while alert rates remain uniform. Used for metric isolation analysis (targets Î´_cal).

- **Generated Data**
  - `synthetic_fairness_data_biased.csv`: Synthetic data with biases in platform visibility, alert generation, and scoring (full bias scenario).
  - `synthetic_fairness_data_neutral.csv`: Synthetic data with balanced, unbiased characteristics (neutral baseline).
  - `synthetic_fairness_data_alert_only.csv`: **(NEW)** Synthetic data with alert-generation bias only.
  - `synthetic_fairness_data_calibration_only.csv`: **(NEW)** Synthetic data with calibration bias only.

- **Metrics Calculation and Analysis**
  - `metrics_calculator.py`: Calculates and compares fairness metrics for biased and neutral datasets (2-scenario comparison).
  - `metrics_calculator_extended.py`: **(NEW)** Calculates fairness metrics across all four scenarios (neutral, alert-only, calibration-only, full bias), generates a comparison table and per-metric visualizations for metric isolation analysis.
  - `temporal_analysis.py`: Analyzes the temporal evolution of fairness metrics.

- **Real-World Data Processing**
  - `realworld.py`: Processes the UNSW dataset, applies robust preprocessing, and computes fairness metrics using bootstrapping.
 
- **Statistical Significance Testing**
  - `test_hip.py`: Performs statistical significance testing on the fairness metrics using the Mann-Whitney U test, including **rank-biserial correlation as effect size** **(UPDATED)** to complement p-values under large sample sizes. Outputs results to `hypothesis_test_results.csv`.
 
- **Classical Fairness Metrics**
  - `classic_metrics.py`: Computes classical fairness metrics (SPD and EOD) using TCP as the reference group and outputs results in CSV and LaTeX formats.
 
- **Empirical compatibility of new operational fairness metrics**
  - `compatibility.py`: Analyzes the joint behavior of fairness metrics across platforms, flags significant deviations, and computes metric correlations.

## Usage

### Generating Synthetic Data

1. **Biased Data (full bias)**:
   ```bash
   python biased_data_generator.py
   ```
   This will generate `synthetic_fairness_data_biased.csv`.

2. **Balanced Data (neutral baseline)**:
   ```bash
   python balanced_data_generator.py
   ```
   This will generate `synthetic_fairness_data_neutral.csv`.

3. **Alert-only Bias** **(NEW)**:
   ```bash
   python alert_only_bias_generator.py
   ```
   This will generate `synthetic_fairness_data_alert_only.csv`. Only alert generation rates are platform-dependent; confirmation rates and scores remain uniform. This scenario isolates the effect on Ï†_ind.

4. **Calibration-only Bias** **(NEW)**:
   ```bash
   python calibration_only_bias_generator.py
   ```
   This will generate `synthetic_fairness_data_calibration_only.csv`. Alert rates are uniform; confirmation rates and score drift vary by platform. This scenario isolates the effect on Î´_cal.

### Calculating Fairness Metrics

Run the `metrics_calculator.py` script to calculate and compare fairness metrics for the neutral and biased datasets (original 2-scenario comparison):
```bash
python metrics_calculator.py
```

### Metric Isolation Analysis (4-scenario comparison) **(NEW)**

Run the `metrics_calculator_extended.py` script to calculate fairness metrics across all four scenarios and generate comparison plots:
```bash
python metrics_calculator_extended.py
```

This script:

- Loads all four synthetic datasets (neutral, alert-only bias, calibration-only bias, full bias).
- Computes Ï†_ind, Ï†_sep, and Î´_cal for each scenario and platform/score group.
- Generates a summary table: `metric_table_all_scenarios.csv`.
- Produces three comparison plots:
  - `phi_ind_all_scenarios.png`: Operational Independence across all scenarios.
  - `phi_sep_all_scenarios.png`: Detection Separation across all scenarios.
  - `delta_cal_all_scenarios.png`: Calibration Sufficiency across all scenarios.

The metric isolation analysis verifies that each proposed metric responds selectively to its intended fairness dimension:
- **Ï†_ind** activates under alert-only bias (IoT reaches 2.37) but remains near baseline (~1.0) under calibration-only bias.
- **Î´_cal** activates under calibration-only bias (values 0.18â€“0.22) but remains near baseline (~0.02) under alert-only bias.
- **Ï†_sep** shows expected cross-sensitivity under alert-only bias, consistent with the operational semantics of detection quality.

### Temporal Analysis

Run the `temporal_analysis.py` script to analyze the temporal evolution of fairness metrics:
```bash
python temporal_analysis.py
```

### Processing Real-World Data (UNSW Dataset)
This script processes network traffic data from the UNSW dataset and computes fairness metrics across different protocols. Ensure the dataset is available locally. Update the data_dir variable in realworld.py to point to the directory containing the .csv files.

Then run:
```bash
python realworld.py
```

The script will:

Load selected columns from all .csv files in the directory.

Preprocess and clean the data.

Generate fairness metrics:

**Ï†_ind**: Alert disparity by protocol.  
**Ï†_sep**: F1-score by protocol.  
**Î´_cal**: Calibration gap by score.  

Save results as .csv and .png files:

phi_ind_bootstrap.csv, phi_ind_raw.csv, phi_ind_unsw.png  
phi_sep_bootstrap.csv, phi_sep_raw.csv, phi_sep_unsw.png  
delta_cal.csv, delta_cal_unsw.png

To ensure the robustness of the fairness metrics, the script uses bootstrapping, a statistical resampling technique. Each metric is computed multiple times (e.g., 1000 iterations) on randomly sampled subsets of the data (with replacement). This allows the estimation of:

Mean: The average value of the metric across all resamples.

Standard deviation (std): A measure of variability, indicating how much the metric fluctuates across different samples.

This approach provides confidence intervals and helps assess the stability and reliability of the fairness metrics across different groups (e.g., protocols).


### Statistical Significance Testing
After generating the fairness metrics, you can run the following script to test whether the observed differences are statistically significant. Run:

```bash
python test_hip.py
```
This script performs Mann-Whitney U tests on:

- **Ï†_sep**: Compares detection quality (F1-score) between `ospf` and `tcp`.  
- **Ï†_ind**: Compares alert distribution between `tcp` and `ospf`.  
- **Î´_cal**: Compares calibration consistency between the lowest and highest score levels.  

The output includes descriptive statistics, p-values, **and rank-biserial correlation as effect size** **(UPDATED)** to assess both the statistical significance and practical magnitude of the observed differences. Effect sizes are interpreted as: |r| < 0.1 negligible, < 0.3 small, < 0.5 medium, â‰¥ 0.5 large. Results are saved to `hypothesis_test_results.csv`.


### Classical Fairness Metrics
To compute classical fairness metrics such as Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD), run:

```bash
python classic_metrics.py
```
This script loads and preprocesses the UNSW dataset, and then calculates:

SPD: Difference in alert rates between each protocol and the reference group (tcp).  
EOD: Difference in true positive rates (recall) between each protocol and tcp.  

and saves the results in:

fairness_classic_metrics.csv: A CSV summary of SPD and EOD by platform.  
fairness_classic_metrics.tex: A LaTeX-formatted table for inclusion in reports or publications.

### Fairness Metric Compatibility Analysis
To analyze how new operational fairness metrics behave jointly across different platforms, run:

```bash
python compatibility.py
```
This script:

Loads and preprocesses the UNSW dataset.

Computes three fairness metrics per platform:

**Ï†_ind**: Alert rate relative to platform prevalence.  
**Ï†_sep**: F1-score of alert vs. confirmation.  
**Î´_cal**: Calibration deviation across score levels.  

Flags platforms where metrics exceed empirical thresholds.

Computes correlations between the metrics.

Saves the results in:

fairness_metric_compatibility.csv: CSV with metric values and activation flags.  
fairness_metric_compatibility.tex: LaTeX-formatted table for reports.

## Dependencies

Make sure you have the following Python libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

You can install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Acknowledgments

- The UNSW dataset was created by the Cyber Range Lab at UNSW Canberra and is provided for academic use by Nour Moustafa and Jill Slay.
