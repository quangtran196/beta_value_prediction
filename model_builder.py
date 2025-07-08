
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, Concatenate, 
    LayerNormalization, Add
)
from tensorflow.keras.regularizers import l2


def build_oscillation_model(input_shape, output_steps, config):
    """Build an improved model with all tunable parameters"""
    
    # Get activation functions from config (with defaults)
    dense_activation = config.get('dense_activation', 'relu')
    lstm_activation = config.get('lstm_activation', 'tanh')
    output_activation = config.get('output_activation', 'linear')
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First layer with optional bidirectional
    if config.get('use_bidirectional', False):
        x = Bidirectional(LSTM(config.get('lstm_units_1', 64), 
                              activation=lstm_activation,
                              return_sequences=True))(inputs)
    else:
        x = LSTM(config.get('lstm_units_1', 64), 
                activation=lstm_activation,
                return_sequences=True)(inputs)
    
    x = LayerNormalization()(x)
    x = Dropout(config.get('dropout_rate', 0.2))(x)
    
    # Second layer with optional bidirectional
    if config.get('use_bidirectional', False):
        x2 = Bidirectional(LSTM(config.get('lstm_units_2', 96), 
                               activation=lstm_activation,
                               return_sequences=True))(x)
    else:
        x2 = LSTM(config.get('lstm_units_2', 96), 
                 activation=lstm_activation,
                 return_sequences=True)(x)
    
    x2 = LayerNormalization()(x2)
    
    # Add residual connection if specified
    if config.get('use_residual', False):
        # Check if shapes match for residual connection
        if config.get('use_bidirectional', False):
            # Bidirectional doubles the output units
            if config.get('lstm_units_1', 64)*2 == config.get('lstm_units_2', 96)*2:
                x = Add()([x, x2])
            else:
                x = x2
        else:
            if config.get('lstm_units_1', 64) == config.get('lstm_units_2', 96):
                x = Add()([x, x2])
            else:
                x = x2
    else:
        x = x2
        
    x = Dropout(config.get('dropout_rate', 0.2))(x)
    
    # Third layer
    x = LSTM(config.get('lstm_units_2', 96), 
            activation=lstm_activation)(x)
    x = LayerNormalization()(x)
    x = Dropout(config.get('dropout_rate', 0.2))(x)
    
    # Dense layers with L2 regularization
    x = Dense(
        config.get('dense_units_1', 256), 
        activation=dense_activation,
        kernel_regularizer=l2(config.get('l2_reg', 0.001))
    )(x)
    
    x = Dense(
        config.get('dense_units_2', 128), 
        activation=dense_activation,
        kernel_regularizer=l2(config.get('l2_reg', 0.001))
    )(x)
    
    # Output layer
    outputs = Dense(output_steps, activation=output_activation)(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Select optimizer based on config
    optimizer_name = config.get('optimizer', 'adam').lower()
    learning_rate = config.get('learning_rate', 0.001)
    
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:  # sgd
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    # Compile model
    model.compile(optimizer=optimizer, loss='mse')
    return model