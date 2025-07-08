# Beta Oscillation Model - Workflow and File Interactions

## File Interaction Flowchart

```
main.py (Entry Point)
    ├── config.py (Configuration Management)
    ├── data_processor.py (Data Pipeline)
    ├── model_builder.py (Neural Network Architecture)
    ├── trainer.py (Training Pipeline)
    │   └── Uses model from model_builder.py
    ├── evaluator.py (Evaluation & Visualization)
    │   └── Uses trained model and data
    └── hyperparameter_tuner.py (Optional Optimization)
        └── Uses all above components
```

## Detailed File Descriptions and Interactions

### 1. **main.py** - Orchestrator
**Purpose**: Central control point that coordinates all other modules.

**Key Functions**:
- `train_model()`: Orchestrates the complete training pipeline
- `evaluate_model()`: Runs evaluation on existing models
- `compare_models()`: Compares performance of two models

**Workflow**:
```python
1. Load configuration (from config.py)
2. Process data (using data_processor.py)
3. Build model (using model_builder.py)
4. Train model (using trainer.py)
5. Evaluate model (using evaluator.py)
6. Save results and configuration
```

### 2. **config.py** - Configuration Hub
**Purpose**: Centralized configuration management with multiple preset configurations.

**Key Components**:
- `BASE_CONFIG`: Optimized hyperparameters from Bayesian tuning
- `LEGACY_CONFIG`: Original hyperparameters for comparison
- `EXPERIMENTAL_CONFIG`: Testing new approaches

**How it's used**:
```python
# In main.py
config = get_config(config_path, use_legacy=False)
# Config flows to all other modules
data_processor = DataProcessor(config)
model = build_oscillation_model(input_shape, output_steps, config)
```

### 3. **data_processor.py** - Data Pipeline
**Purpose**: Handles all data loading, preprocessing, and sequence creation.

**Key Methods**:
- `load_data()`: Reads raw CSV data
- `split_data_by_time()`: Temporal splitting to prevent leakage
- `scale_data()`: Normalizes features using MinMaxScaler
- `create_sequences_from_df()`: Creates sliding window sequences

**Data Flow**:
```
Raw CSV → Load → Temporal Split → Scale → Create Sequences → Return Dictionary
                   (by amp/freq)    (fit on train)   (with oversampling)
```

**Output Structure**:
```python
{
    'X_train': array(n_samples, window_size, 3),  # [amplitude, frequency, beta]
    'y_train': array(n_samples, horizon),         # future beta values
    'X_val': ...,
    'y_val': ...,
    'X_test': ...,
    'y_test': ...,
    'scaler': MinMaxScaler object,
    'groups_train': array of (amp, freq) pairs
}
```

### 4. **model_builder.py** - Neural Network Factory
**Purpose**: Constructs the LSTM-based neural network architecture.

**Architecture Flow**:
```
Input (window_size, 3)
    ↓
LSTM Layer 1 (64 units) → LayerNorm → Dropout
    ↓
LSTM Layer 2 (256 units) → LayerNorm → (Optional Residual)
    ↓
LSTM Layer 3 (256 units) → LayerNorm → Dropout
    ↓
Dense Layer 1 (384 units, L2 reg) → Dense Layer 2 (96 units, L2 reg)
    ↓
Output Layer (prediction_horizon units)
```

**Configuration Integration**:
- Reads architecture parameters from config
- Supports different activations, dropout rates, and layer sizes
- Compiles with specified optimizer and learning rate

### 5. **trainer.py** - Training Engine
**Purpose**: Implements sophisticated training strategies with oscillation awareness.

**Key Components**:

1. **OscillationAwareLoss**: Custom loss function
   ```python
   loss = mse + range_weight * (true_range - pred_range)² + freq_weight * freq_loss
   ```

2. **Training Pipeline** (two phases):
   - **Phase 1**: Unified training
     - Data augmentation for high oscillations
     - Sample weighting based on oscillation magnitude
     - Progressive weight scheduling
   - **Phase 2**: Closed-loop refinement
     - Fine-tuning with recursive predictions
     - Reduced learning rate

3. **Key Methods**:
   - `augment_oscillation_data()`: Creates synthetic high-oscillation samples
   - `compute_sample_weights()`: Weights samples by oscillation magnitude
   - `train_unified_model()`: Main training with all enhancements
   - `train_closed_loop_refinement()`: Improves multi-step predictions

### 6. **evaluator.py** - Analysis and Visualization
**Purpose**: Comprehensive model evaluation and visualization generation.

**Evaluation Flow**:
```
1. Generate predictions (open-loop and closed-loop)
2. Calculate metrics (MSE, MAE, oscillation-specific)
3. Create visualizations
4. Save results
```

**Key Outputs**:
1. **Predictions**: 
   - Open-loop: Direct model predictions
   - Closed-loop: Recursive predictions using previous outputs

2. **Visualizations**:
   - `parameter_grid_unscaled.png`: Grid showing predictions for each (amp, freq) pair
   - `mse_heatmap_comparison.png`: Performance heatmaps
   - `evaluation_comprehensive.png`: Overall performance plots
   - `oscillation_analysis.png`: Detailed oscillation characteristics

3. **Metrics**:
   - Standard: MSE, MAE for both scaled/unscaled data
   - Oscillation-specific: Peak-to-peak ratios, frequency accuracy

### 7. **hyperparameter_tuner.py** - Optimization Engine
**Purpose**: Bayesian optimization for finding optimal hyperparameters.

**Process**:
```
1. Define search space (architecture, training params)
2. Run trials with different configurations
3. Evaluate using validation set performance
4. Use Optuna's Bayesian optimization
5. Save best parameters
```

**Integration**:
- Uses all other components in each trial
- Updates config.py with best parameters

## Data Flow Through the System

```
1. Raw Data (CSV)
    ↓ data_processor.py
2. Processed Sequences
    ↓ model_builder.py
3. Neural Network Model
    ↓ trainer.py
4. Trained Model
    ↓ evaluator.py
5. Predictions & Visualizations
```

## Key Design Patterns

### 1. **Configuration-Driven Design**
All components accept a configuration dictionary, making the system highly modular and experiment-friendly.

### 2. **Separation of Concerns**
Each file has a single, well-defined responsibility:
- Data processing is isolated from model building
- Training logic is separate from evaluation
- Configuration management is centralized

### 3. **Pipeline Architecture**
The system follows a clear pipeline pattern:
```
Config → Data → Model → Train → Evaluate → Visualize
```

### 4. **Extensibility**
New features can be added by:
- Adding new loss functions in `trainer.py`
- Modifying architecture in `model_builder.py`
- Adding new metrics in `evaluator.py`
- Creating new data preprocessing in `data_processor.py`

## Typical Execution Flow

### Training a New Model:
```python
# main.py orchestrates:
config = get_config()  # From config.py
data = DataProcessor(config).prepare_data()  # From data_processor.py
model = build_oscillation_model(...)  # From model_builder.py
trainer = ModelTrainer(model, data, config)  # From trainer.py
trained_model = trainer.run_training_pipeline()
evaluator = ModelEvaluator(trained_model, data, config)  # From evaluator.py
metrics = evaluator.run_evaluation()
```

### Evaluating Existing Model:
```python
# main.py orchestrates:
model = load_model(path)
data = DataProcessor(config).prepare_data()
evaluator = ModelEvaluator(model, data, config)
metrics = evaluator.run_evaluation()
```

## Inter-Module Dependencies

- **config.py** → Used by all modules
- **data_processor.py** → Independent, provides data to all
- **model_builder.py** → Uses config only
- **trainer.py** → Uses model_builder and data
- **evaluator.py** → Uses trained model and data
- **main.py** → Orchestrates all modules
- **hyperparameter_tuner.py** → Uses all modules for optimization

This modular design allows for easy testing, modification, and extension of individual components while maintaining a clean, understandable workflow.
