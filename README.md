# Fairness Metrics and Analysis

This repository contains scripts and data for generating synthetic datasets, calculating fairness metrics, and analyzing temporal trends. It also includes processing scripts for real-world data from the CIC IDS 2017 dataset.

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
  - `realworld.py`: Processes network traffic data from the CIC IDS 2017 dataset and calculates fairness metrics.

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

### Processing Real-World Data

Ensure you have the CIC IDS 2017 dataset available. Modify the `traffic_labeling_dir` variable in `realworld.py` to point to the directory containing the dataset files. Then run:
```bash
python realworld.py
```

## Dependencies

Make sure you have the following Python libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Acknowledgments

- The CIC IDS 2017 dataset is provided by the Canadian Institute for Cybersecurity.