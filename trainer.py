import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, Callback
)
import matplotlib.pyplot as plt
from config import CONFIG
import time

class OscillationAwareLoss(tf.keras.losses.Loss):
    """Custom loss that emphasizes oscillation accuracy"""
    def __init__(self, mse_weight=1.0, range_weight=0.3, freq_weight=0.1, name='oscillation_aware_loss'):
        super().__init__(name=name)
        self.mse_weight = mse_weight
        self.range_weight = range_weight
        self.freq_weight = freq_weight
    
    def call(self, y_true, y_pred):
        # Base MSE loss
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        
        # Oscillation range loss
        true_range = tf.reduce_max(y_true, axis=-1) - tf.reduce_min(y_true, axis=-1)
        pred_range = tf.reduce_max(y_pred, axis=-1) - tf.reduce_min(y_pred, axis=-1)
        range_loss = tf.square(true_range - pred_range)
        
        # Frequency domain loss (simplified to avoid symbolic tensor issues)
        if self.freq_weight > 0:
            # Use a simpler frequency-like metric that works with symbolic tensors
            # Calculate the variance as a proxy for oscillation content
            true_var = tf.math.reduce_variance(y_true, axis=-1)
            pred_var = tf.math.reduce_variance(y_pred, axis=-1)
            freq_loss = tf.abs(true_var - pred_var)
        else:
            freq_loss = 0.0
        
        # Combined loss
        total_loss = (self.mse_weight * mse + 
                     self.range_weight * range_loss + 
                     self.freq_weight * freq_loss)
        
        return tf.reduce_mean(total_loss)

class ProgressiveOscillationCallback(Callback):
    """Callback to progressively increase focus on oscillations during training"""
    def __init__(self, X_train, y_train, config):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.config = config
        self.base_weights = self._compute_base_weights()
        
    def _compute_base_weights(self):
        """Compute base weights based on oscillation characteristics"""
        weights = np.ones(len(self.X_train))
        
        for i in range(len(self.X_train)):
            # Current oscillation range in input
            input_range = self.X_train[i, :, 2].max() - self.X_train[i, :, 2].min()
            
            # Future oscillation range in target
            target_range = self.y_train[i].max() - self.y_train[i].min()
            
            # Combined weight based on both input and output oscillations
            osc_score = (input_range + target_range) / 2
            
            # Weight calculation
            if osc_score > self.config['high_oscillation_threshold']:
                # Smooth weight increase based on oscillation magnitude
                weight_factor = np.clip(osc_score / self.config['high_oscillation_threshold'], 1.0, 3.0)
                weights[i] = weight_factor
        
        # Normalize weights to prevent numerical issues
        weights = weights / weights.mean()
        return weights
    
    def on_epoch_begin(self, epoch, logs=None):
        """Update sample weights progressively"""
        # Calculate progress through training
        if hasattr(self, 'params') and self.params and 'epochs' in self.params:
            total_epochs = self.params['epochs']
        else:
            total_epochs = self.config.get('epochs', 100)
        
        progress = epoch / total_epochs
        
        # Progressive weight scaling: start at 1, increase to high_oscillation_weight
        scale_factor = 1.0 + (self.config['high_oscillation_weight'] - 1.0) * progress
        
        # Apply progressive scaling to base weights
        current_weights = 1.0 + (self.base_weights - 1.0) * scale_factor
        
        # Store for use in training 
        #  this is for reference, actual sample_weight is passed in fit
        self.current_weights = current_weights

class ModelTrainer:
    def __init__(self, model, data, config=CONFIG):
        self.model = model
        self.data = data
        self.config = config
        self.checkpoint_path = os.path.join(config['output_dir'], "best_model.h5")
        self.log_dir = os.path.join(config['output_dir'], "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
    def compute_sample_weights(self, X, y):
        """Compute sample weights based on oscillation characteristics"""
        weights = np.ones(len(X))
        
        for i in range(len(X)):
            # Current oscillation range in input
            input_range = X[i, :, 2].max() - X[i, :, 2].min()
            
            # Future oscillation range in target
            target_range = y[i].max() - y[i].min()
            
            # Combined weight based on both input and output oscillations
            osc_score = (input_range + target_range) / 2
            
            # Exponential weighting for high oscillations
            if osc_score > self.config['high_oscillation_threshold']:
                # Smooth exponential increase
                relative_score = osc_score / self.config['high_oscillation_threshold']
                weights[i] = 1 + (self.config['high_oscillation_weight'] - 1) * \
                            (1 - np.exp(-2 * (relative_score - 1)))
        
        # Normalize weights to prevent numerical issues
        weights = weights / weights.mean()
        print(f"Weight statistics: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
        
        return weights
    
    def augment_oscillation_data(self, X, y, augment_ratio=0.2):
        """Augment training data with synthetic oscillations"""
        print(f"Augmenting data with {augment_ratio*100:.0f}% synthetic oscillations...")
        
        # Find high oscillation samples
        high_osc_indices = []
        for i in range(len(X)):
            beta_range = X[i, :, 2].max() - X[i, :, 2].min()
            if beta_range > self.config['high_oscillation_threshold']:
                high_osc_indices.append(i)
        
        if len(high_osc_indices) == 0:
            print("No high oscillation samples found for augmentation")
            return X, y
        
        num_augment = min(int(len(X) * augment_ratio), len(high_osc_indices) * 3)
        X_aug = []
        y_aug = []
        
        for _ in range(num_augment):
            # Select a high oscillation sample
            idx = np.random.choice(high_osc_indices)
            x_sample = X[idx].copy()
            y_sample = y[idx].copy()
            
            # Add controlled perturbations
            # 1. Phase shift
            phase_shift = np.random.uniform(-0.2, 0.2)
            # 2. Amplitude modulation
            amp_factor = np.random.uniform(0.8, 1.2)
            # 3. Frequency perturbation
            freq_factor = np.random.uniform(0.9, 1.1)
            
            # Apply perturbations to beta values
            t = np.linspace(0, 1, len(y_sample))
            base_signal = y_sample - y_sample.mean()
            
            # Simple frequency modulation through interpolation
            t_new = t * freq_factor
            t_new = np.clip(t_new, 0, 1)
            perturbed_signal = np.interp(t, t_new, base_signal)
            
            # Apply amplitude and phase
            perturbed_signal = amp_factor * perturbed_signal
            y_sample = perturbed_signal + y_sample.mean() + phase_shift
            
            # Ensure continuity between input and output
            if len(x_sample) > 0:
                # Adjust last input to match first output
                x_sample[-1, 2] = y_sample[0]
            
            X_aug.append(x_sample)
            y_aug.append(y_sample)
        
        print(f"Created {len(X_aug)} augmented samples")
        
        if len(X_aug) == 0:
            return X, y
        
        # Combine original and augmented data
        X_combined = np.vstack([X, np.array(X_aug)])
        y_combined = np.vstack([y, np.array(y_aug)])
        
        # Shuffle the combined data
        indices = np.random.permutation(len(X_combined))
        
        return X_combined[indices], y_combined[indices]
    
    def train_unified_model(self):
        """Train model with unified approach combining all strategies"""
        print("Starting unified model training with oscillation awareness...")
        
        # Augment training data
        X_train_aug, y_train_aug = self.augment_oscillation_data(
            self.data['X_train'], 
            self.data['y_train'],
            augment_ratio=0.2
        )
        
        # Compute sample weights
        sample_weights = self.compute_sample_weights(X_train_aug, y_train_aug)
        
        # Update model with custom loss
        # First, get current optimizer settings
        current_optimizer = self.model.optimizer
        if hasattr(current_optimizer, 'learning_rate'):
            if hasattr(current_optimizer.learning_rate, 'numpy'):
                learning_rate = float(current_optimizer.learning_rate.numpy())
            else:
                learning_rate = float(current_optimizer.learning_rate)
        else:
            learning_rate = float(self.config['learning_rate'])
        
        # Re-compile with oscillation-aware loss
        custom_loss = OscillationAwareLoss(
            mse_weight=self.config.get('oscillation_loss_weights', {}).get('mse_weight', 1.0),
            range_weight=self.config.get('oscillation_loss_weights', {}).get('range_weight', 0.3),
            freq_weight=self.config.get('oscillation_loss_weights', {}).get('freq_weight', 0.05)
        )
        
        # Re-compile model
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        
        self.model.compile(optimizer=optimizer, loss=custom_loss)
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                self.checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            ProgressiveOscillationCallback(X_train_aug, y_train_aug, self.config),
            TensorBoard(log_dir=self.log_dir)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train_aug, y_train_aug,
            validation_data=(self.data['X_val'], self.data['y_val']),
            sample_weight=sample_weights,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def train_closed_loop_refinement(self):
        """Final refinement with closed-loop training"""
        print("Starting closed-loop refinement...")
        
        # Switch back to standard MSE for stability
        optimizer = self.model.optimizer
        self.model.compile(optimizer=optimizer, loss='mse')
        
        # Select diverse samples for closed-loop training
        num_samples = min(2000, len(self.data['X_train']))
        
        # Get samples with different oscillation characteristics
        beta_ranges = []
        for i in range(len(self.data['X_train'])):
            beta_range = self.data['X_train'][i, :, 2].max() - self.data['X_train'][i, :, 2].min()
            beta_ranges.append(beta_range)
        
        beta_ranges = np.array(beta_ranges)
        
        # Select samples from different oscillation ranges
        low_osc = np.where(beta_ranges < np.percentile(beta_ranges, 33))[0]
        med_osc = np.where((beta_ranges >= np.percentile(beta_ranges, 33)) & 
                          (beta_ranges < np.percentile(beta_ranges, 67)))[0]
        high_osc = np.where(beta_ranges >= np.percentile(beta_ranges, 67))[0]
        
        # Sample equally from each range
        samples_per_range = num_samples // 3
        selected_indices = np.concatenate([
            np.random.choice(low_osc, min(samples_per_range, len(low_osc)), replace=False),
            np.random.choice(med_osc, min(samples_per_range, len(med_osc)), replace=False),
            np.random.choice(high_osc, min(samples_per_range, len(high_osc)), replace=False)
        ])
        
        X_subset = self.data['X_train'][selected_indices]
        y_subset = self.data['y_train'][selected_indices]
        
        # Generate closed-loop sequences
        X_closed = []
        y_closed = []
        
        for i in range(len(X_subset)):
            # Original sequence
            X_closed.append(X_subset[i])
            y_closed.append(y_subset[i])
            
            # Generate closed-loop variant
            current_seq = X_subset[i].copy()
            
            # Multi-step prediction
            predictions = []
            for step in range(2):  # 2-step lookahead
                pred = self.model.predict(np.expand_dims(current_seq, axis=0), verbose=0)[0]
                predictions.append(pred)
                
                # Update sequence
                new_seq = current_seq.copy()
                new_seq[:-1] = current_seq[1:]
                new_seq[-1, 2] = pred[0]  # Use first prediction as new input
                new_seq[-1, 0:2] = current_seq[-1, 0:2]  # Keep amplitude and frequency
                current_seq = new_seq
            
            # Create training sample mixing predictions and true values
            mixed_seq = X_subset[i].copy()
            # Use predictions to create realistic perturbations
            window_size = self.config['window_size']
            mix_ratio = 0.5
            for j in range(min(len(predictions[0]), window_size // 2)):
                mixed_seq[-(j+1), 2] = (mix_ratio * predictions[0][j] + 
                                       (1-mix_ratio) * mixed_seq[-(j+1), 2])
            
            X_closed.append(mixed_seq)
            y_closed.append(y_subset[i])
        
        X_closed = np.array(X_closed)
        y_closed = np.array(y_closed)
        
        print(f"Created {len(X_closed)} closed-loop training sequences")
        
        # Train with reduced learning rate for fine-tuning
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Use smaller learning rate for fine-tuning
        if hasattr(self.model.optimizer, 'learning_rate'):
            if hasattr(self.model.optimizer.learning_rate, 'numpy'):
                current_lr = float(self.model.optimizer.learning_rate.numpy())
            else:
                current_lr = float(self.model.optimizer.learning_rate)
        else:
            current_lr = float(self.config['learning_rate'])
        
        fine_tune_lr = current_lr * 0.1
        
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=fine_tune_lr)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=fine_tune_lr)
        
        self.model.compile(optimizer=optimizer, loss='mse')
        
        cl_history = self.model.fit(
            X_closed, y_closed,
            validation_data=(self.data['X_val'], self.data['y_val']),
            epochs=20,  # Fewer epochs for fine-tuning
            batch_size=max(16, self.config['batch_size'] // 2),
            callbacks=callbacks,
            verbose=1
        )
        
        return cl_history
    
    def run_training_pipeline(self):
        """Execute the improved training pipeline"""
        start_time = time.time()
        
        # Phase 1: Unified training with oscillation awareness
        print("\n=== Phase 1: Unified Training with Oscillation Awareness ===")
        unified_history = self.train_unified_model()
        self.plot_history(unified_history, "unified_training_history.png")
        
        # Save intermediate model
        self.model.save(os.path.join(self.config['output_dir'], "model_unified.h5"))
        
        # Phase 2: Closed-loop refinement
        print("\n=== Phase 2: Closed-loop Refinement ===")
        cl_history = self.train_closed_loop_refinement()
        if cl_history:
            self.plot_history(cl_history, "closed_loop_refinement_history.png")
        
        # Save final model
        model_path = os.path.join(self.config['output_dir'], self.config['model_name'])
        self.model.save(model_path)
        print(f"\nFinal model saved to {model_path}")
        
        # Training summary
        end_time = time.time()
        training_time = (end_time - start_time) / 60
        
        print(f"\n=== Training Summary ===")
        print(f"Total training time: {training_time:.2f} minutes")
        print(f"Final validation loss: {unified_history.history['val_loss'][-1]:.6f}")
        
        # Analyze oscillation performance
        self._analyze_oscillation_performance()
        
        return self.model
    
    def _analyze_oscillation_performance(self):
        """Analyze model performance on different oscillation ranges"""
        print("\n=== Oscillation Performance Analysis ===")
        
        # Get predictions
        y_pred = self.model.predict(self.data['X_val'], verbose=0)
        y_true = self.data['y_val']
        
        # Categorize by oscillation range
        results = {'low': [], 'medium': [], 'high': []}
        
        for i in range(len(y_true)):
            true_range = y_true[i].max() - y_true[i].min()
            pred_range = y_pred[i].max() - y_pred[i].min()
            
            mse = np.mean((y_true[i] - y_pred[i])**2)
            range_error = abs(true_range - pred_range)
            
            if true_range < self.config['high_oscillation_threshold'] * 0.5:
                category = 'low'
            elif true_range < self.config['high_oscillation_threshold'] * 1.5:
                category = 'medium'
            else:
                category = 'high'
            
            results[category].append({
                'mse': mse,
                'range_error': range_error,
                'true_range': true_range,
                'pred_range': pred_range
            })
        
        # Print statistics
        for category in ['low', 'medium', 'high']:
            if results[category]:
                data = results[category]
                avg_mse = np.mean([d['mse'] for d in data])
                avg_range_error = np.mean([d['range_error'] for d in data])
                avg_range_ratio = np.mean([d['pred_range']/d['true_range'] 
                                          for d in data if d['true_range'] > 0])
                
                print(f"\n{category.upper()} oscillations ({len(data)} samples):")
                print(f"  Average MSE: {avg_mse:.6f}")
                print(f"  Average range error: {avg_range_error:.6f}")
                print(f"  Average range ratio (pred/true): {avg_range_ratio:.3f}")
    
    def plot_history(self, history, filename):
        """Plot and save training history"""
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot learning rate if available
        if 'lr' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], filename), dpi=150)
        plt.close()

    def run_openloop_training_pipeline(self):
        """Execute open-loop only training pipeline with oscillation awareness"""
        start_time = time.time()
        
        print("\n=== Open-Loop Training with High Oscillation Weighting ===")
        
        # Phase 1: Augment training data
        X_train_aug, y_train_aug = self.augment_oscillation_data(
            self.data['X_train'], 
            self.data['y_train'],
            augment_ratio=self.config.get('augment_ratio', 0.2)
        )
        
        # Phase 2: Compute sample weights with higher emphasis on oscillations
        sample_weights = self.compute_sample_weights(X_train_aug, y_train_aug)
        
        # Phase 3: Configure model with oscillation-aware loss
        custom_loss = OscillationAwareLoss(
            mse_weight=self.config.get('oscillation_loss_weights', {}).get('mse_weight', 1.0),
            range_weight=self.config.get('oscillation_loss_weights', {}).get('range_weight', 0.3),
            freq_weight=self.config.get('oscillation_loss_weights', {}).get('freq_weight', 0.05)
        )
        
        # Get optimizer
        learning_rate = self.config.get('learning_rate', 0.001)
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        
        self.model.compile(optimizer=optimizer, loss=custom_loss)
        
        # Phase 4: Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                self.checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            ProgressiveOscillationCallback(X_train_aug, y_train_aug, self.config),
            TensorBoard(log_dir=self.log_dir)
        ]
        
        # Phase 5: Train the model
        print(f"\nTraining with {len(X_train_aug)} samples (including augmented data)")
        print(f"Sample weight range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
        
        history = self.model.fit(
            X_train_aug, y_train_aug,
            validation_data=(self.data['X_val'], self.data['y_val']),
            sample_weight=sample_weights,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the model
        model_path = os.path.join(self.config['output_dir'], self.config['model_name'])
        self.model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        # Plot training history
        self.plot_history(history, "openloop_training_history.png")
        
        # Training summary
        end_time = time.time()
        training_time = (end_time - start_time) / 60
        
        print(f"\n=== Training Summary ===")
        print(f"Total training time: {training_time:.2f} minutes")
        print(f"Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
        print(f"Total epochs trained: {len(history.history['loss'])}")
        
        # Analyze oscillation performance
        self._analyze_oscillation_performance()
        
        return self.model