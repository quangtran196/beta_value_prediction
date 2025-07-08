# Beta Oscillation Prediction Model for Deep Brain Stimulation

A deep learning model for predicting beta oscillations in Parkinson's disease under different Deep Brain Stimulation (DBS) parameters. This model uses LSTM networks to predict future beta oscillation values based on DBS amplitude, frequency, and current beta state.

## Overview

This project implements a sophisticated neural network model that:
- Predicts beta oscillations for various DBS parameter combinations
- Handles both open-loop and closed-loop prediction scenarios
- Emphasizes accurate prediction of oscillation characteristics
- Provides comprehensive evaluation and visualization tools

## Features

### Core Capabilities
- **Multi-step ahead prediction** of beta oscillations (default: 10 steps)
- **Oscillation-aware training** with custom loss functions
- **Temporal data splitting** to prevent data leakage
- **Comprehensive evaluation** including parameter-specific analysis
- **GPU support** for faster training

### Advanced Features
- **Hyperparameter tuning** using Bayesian optimization (Optuna)
- **Data augmentation** for high oscillation samples
- **Progressive weight scheduling** during training
- **Closed-loop refinement** phase for improved recursive predictions
- **Extensive visualization** including heatmaps and parameter grids

## Requirements

### Software Dependencies
```
Python >= 3.8
TensorFlow >= 2.10.0
NumPy >= 1.21.0
Pandas >= 1.3.0
Scikit-learn >= 1.0.0
Matplotlib >= 3.4.0
SciPy >= 1.7.0
Optuna >= 3.0.0 (optional, for hyperparameter tuning)
```
## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd beta-oscillation-model
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
```

## Project Structure

```
beta-oscillation-model/
├── main.py                  # Main entry point for training/evaluation
├── config.py               # Configuration management
├── data_processor.py       # Data loading and preprocessing
├── model_builder.py        # Neural network architecture
├── trainer.py              # Training pipelines and callbacks
├── evaluator.py            # Model evaluation and visualization
├── hyperparameter_tuner.py # Bayesian optimization (optional)
├── analyze_predictions.py  # Post-training analysis tools
├── merged_data_with_error.csv  # Input data file
└── README.md               # This file
```

### Output Structure (after training)
```
<output_dir>/
├── beta_oscillation_model.h5     # Trained model
├── training_config.json          # Configuration used
├── training_summary.json         # Training results
├── unified_training_history.png  # Training curves
├── evaluation/
│   ├── all_predictions.pkl       # Predictions for analysis
│   ├── metrics.json              # Evaluation metrics
│   ├── parameter_grid_unscaled.png  # Predictions by DBS parameters
│   └── mse_heatmap_comparison.png   # Performance heatmaps
└── logs/                         # TensorBoard logs
```


## Quick Start


### 1. Basic Training
```bash
# Train with optimized hyperparameters
python main.py --mode train --output_dir my_model

# Train with fewer epochs for testing
python main.py --mode train --output_dir quick_test --epochs 20
```

### 2. Evaluate Existing Model
```bash
python main.py --mode evaluate --model_path my_model/beta_oscillation_model.h5
```

### 3. Analyze Predictions
```bash
python analyze_predictions.py \
    --predictions_file my_model/evaluation/all_predictions.pkl \
    --output_dir my_model/analysis
```

## Detailed Usage

### Training Options

#### Standard Training (Recommended)
```bash
python main.py --mode train 
```
This uses optimized hyperparameters and the full training pipeline.

#### Legacy Training
```bash
python main.py --mode train --use_legacy --output_dir beta_model_legacy
```
Uses original hyperparameters for comparison.

#### Custom Configuration
Create a JSON configuration file:
```json
{
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.0005,
    "high_oscillation_weight": 8
}
```

Then train with:
```bash
python main.py --mode train --config_path custom_config.json
```

#### Hyperparameter Tuning
```bash
python main.py --mode tune --output_dir tuning_results
```
Runs Bayesian optimization to find optimal hyperparameters.

### Evaluation Options

#### Standard Evaluation
```bash
python main.py --mode evaluate --model_path path/to/model.h5
```

#### Model Comparison
```bash
python main.py --mode compare \
    --model_path model1.h5 \
    --model2_path model2.h5 \
    --output_dir comparison
```

### Advanced Analysis
```bash
# Detailed prediction analysis
python analyze_predictions.py \
    --predictions_file results/evaluation/all_predictions.pkl \
    --output_dir results/detailed_analysis
```

## Configuration

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 10 | Input sequence length |
| `prediction_horizon` | 10 | Number of steps to predict |
| `batch_size` | 32 | Training batch size |
| `epochs` | 100 | Maximum training epochs |
| `learning_rate` | 0.000926 | Initial learning rate |
| `high_oscillation_threshold` | 0.1 | Threshold for high oscillations |
| `high_oscillation_weight` | 5 | Weight multiplier for high oscillations |

### Model Architecture Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lstm_units_1` | 64 | Units in first LSTM layer |
| `lstm_units_2` | 256 | Units in second/third LSTM layers |
| `dense_units_1` | 384 | Units in first dense layer |
| `dense_units_2` | 96 | Units in second dense layer |
| `dropout_rate` | 0.15 | Dropout rate |
| `use_residual` | True | Enable residual connections |

## Model Architecture

The model consists of:
1. **Input Layer**: Accepts sequences of (amplitude, frequency, beta_value)
2. **LSTM Layers**: 3 stacked LSTM layers with layer normalization
3. **Dense Layers**: 2 fully connected layers with L2 regularization
4. **Output Layer**: Predicts future beta values

### Custom Loss Function
The oscillation-aware loss combines:
- Mean Squared Error (MSE)
- Peak-to-peak range accuracy
- Frequency domain similarity

## Training Pipeline

### Phase 1: Unified Training
- Data augmentation for high oscillation samples
- Sample weighting based on oscillation magnitude
- Progressive weight scheduling
- Custom oscillation-aware loss

### Phase 2: Closed-Loop Refinement
- Fine-tuning with recursive predictions
- Reduced learning rate
- Focus on multi-step accuracy

## Evaluation Metrics

### Standard Metrics
- **MSE**: Mean Squared Error (scaled and unscaled)
- **MAE**: Mean Absolute Error
- **Improvement**: Percentage improvement over baseline

### Oscillation-Specific Metrics
- **Peak-to-Peak Ratio**: Accuracy of oscillation amplitude
- **Within 20%**: Percentage of predictions within 20% of true amplitude
- **Frequency Error**: Accuracy of oscillation frequency

### Visualization Outputs
1. **Parameter Grid**: Shows predictions for each amplitude-frequency pair
2. **MSE Heatmaps**: Performance across parameter space
3. **Time Series Plots**: Predicted vs actual beta values
4. **Error Distributions**: Statistical analysis of prediction errors


### Performance Tips
1. **Use GPU**: Ensure TensorFlow GPU is installed
2. **Optimize batch size**: Larger batches train faster but use more memory
3. **Monitor with TensorBoard**: `tensorboard --logdir output_dir/logs`
4. **Early stopping**: Prevents overfitting automatically


