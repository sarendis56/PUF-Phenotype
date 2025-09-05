# DPAN (DRAM Phenotype-based Authentication Network) Reproduction

This project reproduces the experimental results from the paper:

**"PUF-Phenotype: A Robust and Noise-Resilient Approach to Aid Intra-Group-based Authentication with DRAM-PUFs Using Machine Learning"**

## Overview

The DPAN approach uses machine learning to classify noisy DRAM-PUF responses without requiring helper data or error correction. The key innovation is treating PUF responses as "phenotypes" - complete observable characteristics including noise.

## Key Features

- **Modified VGG16 Feature Extractor**: Uses pre-trained VGG16 with 1x1 average pooling instead of fully connected layers
- **Multiple Classifiers**: Implements 6 different classifiers (SVM, Logistic Regression, Decision Tree, KNN, Random Forest, XGBoost)
- **Confidence Analysis**: Analyzes confidence scores for fraudulent detection
- **Multi-Device Support**: Tests with 3, 4, and 5 device groups
- **Comprehensive Evaluation**: Includes accuracy, F1-score, cross-validation, and confidence metrics

## Dataset Structure

The dataset contains DRAM PUF data from 5 devices (alpha, beta, delta, epsilon, gamma) with:
- 8 locations per device (a, b, c, d, e, f, g, h)
- 4 temperatures (20°C, 30°C, 40°C, 50°C)
- 2 voltages (1.3V, 1.5V)
- 3 challenge patterns (0x00, 0x55, 0xFF)
- 10 measurements per condition

## Installation

1. Make sure you have Python 3.10+ and uv installed
2. Clone/navigate to the project directory
3. Install dependencies:
   ```bash
   python3 -m uv sync
   ```

## Usage

### Basic Usage

Run with default settings (5 devices):
```bash
python3 -m uv run main.py --data_path ../data
```

### Advanced Usage

```bash
python3 -m uv run main.py \
    --data_path ../data \
    --num_devices 5 \
    --batch_size 32 \
    --test_size 0.2 \
    --save_path results \
    --no_plots
```

### Parameters

- `--data_path`: Path to the dataset directory (default: ../data)
- `--num_devices`: Number of devices to use - 3, 4, or 5 (default: 5)
- `--batch_size`: Batch size for feature extraction (default: 32)
- `--test_size`: Test set size as fraction (default: 0.2)
- `--save_path`: Directory to save results (default: results)
- `--no_plots`: Skip generating plots

## Expected Results

Based on the paper, the expected performance for the lightweight VGG16 approach:

### 3 Devices
- Logistic Regression: ~98.3% accuracy
- XGBoost: ~99.1% accuracy

### 4 Devices
- Logistic Regression: ~98.1% accuracy
- Random Forest: ~98.1% accuracy

### 5 Devices
- SVM: ~97.9% accuracy
- Random Forest: ~98.4% accuracy

## Output Files

The script generates several output files in the results directory:

1. `dpan_summary_{num_devices}_devices.csv`: Performance metrics for all classifiers
2. `dpan_confidence_{num_devices}_devices.csv`: Confidence analysis results
3. `dpan_results_{num_devices}_devices.png`: Visualization plots

## Architecture Details

### Modified VGG16 Feature Extractor
- Uses pre-trained VGG16 backbone
- Replaces fully connected layers with 6x6 adaptive average pooling
- Outputs 18,432-dimensional feature vectors (6×6×512)
- Reduces model size from ~500MB to ~57MB

### Classifiers
All classifiers use hyperparameters optimized in the original paper:
- **SVM**: C=10, gamma=0.1
- **Logistic Regression**: C=1, max_iter=100
- **KNN**: n_neighbors=9, leaf_size=1
- **Random Forest**: n_estimators=396
- **XGBoost**: learning_rate=0.02, n_estimators=64, max_depth=3
- **Decision Tree**: criterion='gini', max_depth=10

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for faster feature extraction
- **RAM**: At least 8GB RAM (16GB+ recommended for larger datasets)
- **Storage**: ~2GB for dependencies, ~1GB for dataset

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch_size parameter
2. **Dataset not found**: Check data_path parameter points to correct directory
3. **Missing images**: Ensure dataset is properly extracted

### Performance Tips

1. Use GPU for faster feature extraction
2. Increase batch_size if you have sufficient GPU memory
3. Use organized dataset structure for faster loading

## Citation

If you use this reproduction in your research, please cite the original paper:

```bibtex
@article{millwood2023puf,
  title={PUF-Phenotype: A Robust and Noise-Resilient Approach to Aid Group-Based Authentication With DRAM-PUFs Using Machine Learning},
  author={Millwood, Owen and Miskelly, James and Yang, Bingxin and Gope, Prosanta and Kavun, Elif Bilge and Lin, Chenghua},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={18},
  pages={2451--2465},
  year={2023},
  publisher={IEEE}
}
```
