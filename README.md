# Fairness Metrics and Analysis

This repository contains scripts and data for generating synthetic datasets, calculating fairness metrics, and analyzing temporal trends. It also includes processing scripts for real-world data from the UNSW dataset.

## Directory Structure

- **Data Generators**
  - `biased_data_generator.py`: Generates a synthetic dataset with intentional biases.
  - `balanced_data_generator.py`: Generates a synthetic dataset with balanced, unbiased data.

- **Generated Data**
  - `synthetic_fairness_data_biased.csv`: Synthetic data with biases in platform visibility, alert generation, and scoring.
  - `synthetic_fairness_data_neutral.csv`: Synthetic data with balanced, unbiased characteristics.

- **Metrics Calculation and Analysis**
  - `metrics_calculator.py`: Calculates and compares fairness metrics for biased and neutral datasets.
  - `temporal_analysis.py`: Analyzes the temporal evolution of fairness metrics.

- **Real-World Data Processing**
  - `realworld.py`: Processes the UNSW dataset, applies robust preprocessing, and computes fairness metrics using bootstrapping.
 
- **Statistical Significance Testing**
  - `test.py`: Performs statistical significance testing on the fairness metrics using the Mann-Whitney U test to assess whether observed differences between groups (e.g., protocols) are statistically robust.
 
- **Classical Fairness Metrics**
  - `classic_metrics.py`: Computes classical fairness metrics (SPD and EOD) using TCP as the reference group and outputs results in CSV and LaTeX formats.
 
- **Empirical compatibility of new operational fairness metricks**
  - `compatibility.py`: Analyzes the joint behavior of fairness metrics across platforms, flags significant deviations, and computes metric correlations.

## Usage

### Generating Synthetic Data

1. **Biased Data**:
   ```bash
   python biased_data_generator.py
   ```
   This will generate `synthetic_fairness_data_biased.csv`.

2. **Balanced Data**:
   ```bash
   python balanced_data_generator.py
   ```
   This will generate `synthetic_fairness_data_neutral.csv`.

### Calculating Fairness Metrics

Run the `metrics_calculator.py` script to calculate and compare fairness metrics for the generated datasets:
```bash
python metrics_calculator.py
```

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

**φ_ind**: Alert disparity by protocol.  
**φ_sep**: F1-score by protocol.  
**δ_cal**: Calibration gap by score.  

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
python test.py
```
This script performs Mann-Whitney U tests on:

- **φ_sep**: Compares detection quality (F1-score) between `ospf` and `tcp`.  
- **φ_ind**: Compares alert distribution between `tcp` and `ospf`.  
- **δ_cal**: Compares calibration consistency between the lowest and highest score levels.  

The output includes descriptive statistics and p-values indicating whether the differences are statistically significant.


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

**φ_ind**: Alert rate relative to platform prevalence.  
**φ_sep**: F1-score of alert vs. confirmation.  
**δ_cal**: Calibration deviation across score levels.  

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
