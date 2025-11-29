# PID-IMSDE

PID-Optimized Multi-scale Symbolic Dynamic Entropy for Unsupervised Cross-Domain Fault Diagnosis.

## Data
CWRU bearing dataset available at: https://csegroups.case.edu/bearingdatacenter

## Code
- `main.py`: Main experiment script
- `data_loader.py`: CWRU/PU data loading
- `pid_optimizer.py`: PID parameter optimization
- `elm_classifier.py`: ELM implementation

## Usage
python main.py --source_domain cwru_0hp --target_domain cwru_3hp --random_seed 42

## Results
Table 2 accuracy: 62.0±8.7% (peak: 69.7%)