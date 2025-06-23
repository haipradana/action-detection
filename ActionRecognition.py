#!/usr/bin/env python
# coding: utf-8

# ### Imports

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import gc

from data_gen import DataGenerator
from model import MyCL_Model, SimpleConvLSTM

# Multi-GPU Setup + Memory management
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f"🔥 Found {len(physical_devices)} GPU(s): {[d.name for d in physical_devices]}")

# Enable memory growth for all GPUs
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

# Multi-GPU Strategy
strategy = tf.distribute.MirroredStrategy()
print(f"🚀 Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")

# Enable mixed precision (saves ~40% memory)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

tf.keras.backend.clear_session()
gc.collect()

# Ensure TensorFlow 2.x behavior (eager execution enabled by default)

# ### Paths

videos_path = './Videos_MERL_Shopping_Dataset/'

x_train_path = videos_path+'train/'
y_train_path = 'train_y.pkl'

x_val_path = videos_path + 'val/'
y_val_path = 'val_y.pkl'

# ### Create Train and Validation Data Generator objects

# Increase batch size with 2x GPU power
train_data = DataGenerator(x_train_path, y_path=y_train_path, seq_len=15, batch_size=2)
val_data = DataGenerator(x_val_path, y_path=y_val_path, seq_len=15, batch_size=2)

# ### Define and Compile Model (Multi-GPU)

# Create model within strategy scope for multi-GPU
with strategy.scope():
    # Use original model with mixed precision optimization
    model = MyCL_Model(num_classes=6, dropout_rate=0.2)
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        metrics=['accuracy']
    )
    print("✅ Model created and compiled for multi-GPU training")

# ### Setup callbacks
callbacks = [
    ModelCheckpoint('mycl_best.h5', save_best_only=True, monitor='val_loss'),
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)
]

# ### Train Model (Modern approach)

epochs = 20

print("Starting training...")
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Training
    train_loss = 0.0
    train_acc = 0.0

    # Multi-GPU training approach
    for batch_idx in range(len(train_data)):
        try:
            batch_generator = train_data[batch_idx]
            
            # Use model.fit for proper multi-GPU handling
            history = model.fit(
                batch_generator,
                epochs=1,
                verbose=0,
                workers=1,
                use_multiprocessing=False
            )
            
            train_loss += history.history['loss'][0]
            train_acc += history.history['accuracy'][0]
            
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue
        
        # Memory cleanup every 5 batches (more frequent for multi-GPU)
        if batch_idx % 5 == 0:
            gc.collect()
            print(f"📊 Processed {batch_idx}/{len(train_data)} training batches")
    
    # Calculate average training metrics
    if len(train_data) > 0:
        train_loss /= len(train_data)
        train_acc /= len(train_data)
    
    # Validation
    val_loss = 0.0
    val_acc = 0.0
    
    for batch_idx in range(len(val_data)):
        try:
            batch_generator = val_data[batch_idx]
            # Use evaluate() instead of deprecated evaluate_generator()
            loss, acc = model.evaluate(batch_generator, verbose=0)
            val_loss += loss
            val_acc += acc
        except Exception as e:
            print(f"Error in validation batch {batch_idx}: {e}")
            continue
    
    # Calculate average validation metrics
    if len(val_data) > 0:
        val_loss /= len(val_data)
        val_acc /= len(val_data)
    
    # Save model checkpoint
    model.save(f'mycl_epoch_{epoch+1}.h5')
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

print("Training completed!")

# ### Testing Data Paths

x_test_path = videos_path + 'test/'
y_test_path = 'test_y.pkl'

# ### Create Test Data Generator Object

test_data = DataGenerator(x_test_path, y_path=y_test_path, seq_len=15, batch_size=2)

# ### Test the model

print("Starting testing...")
test_acc = 0.0
test_loss = 0.0
valid_batches = 0

for batch_idx in range(len(test_data)):
    try:
        batch_generator = test_data[batch_idx]
        # Fix: use batch_generator instead of undefined testing_generator
        loss, acc = model.evaluate(batch_generator, verbose=0)
        test_loss += loss
        test_acc += acc
        valid_batches += 1
    except Exception as e:
        print(f"Error in test batch {batch_idx}: {e}")
        continue

# Calculate average test metrics
if valid_batches > 0:
    test_loss /= valid_batches
    test_acc /= valid_batches
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Per Frame Accuracy for Test Data: {test_acc:.4f}")
else:
    print("No valid test batches found!")

