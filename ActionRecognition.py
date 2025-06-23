#!/usr/bin/env python
# coding: utf-8

# ### Imports

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from data_gen import DataGenerator
from model import MyCL_Model

# Ensure TensorFlow 2.x behavior (eager execution enabled by default)

# ### Paths

videos_path = './Videos_MERL_Shopping_Dataset/'

x_train_path = videos_path+'train/'
y_train_path = 'train_y.pkl'

x_val_path = videos_path + 'val/'
y_val_path = 'val_y.pkl'

# ### Create Train and Validation Data Generator objects

train_data = DataGenerator(x_train_path, y_path=y_train_path, seq_len=15, batch_size=1)
val_data = DataGenerator(x_val_path, y_path=y_val_path, seq_len=15, batch_size=1)

# ### Define and Compile Model

model = MyCL_Model()
model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    metrics=['accuracy']
)

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

    for batch_idx in range(len(train_data)):
        try:
            batch_generator = train_data[batch_idx]
            # Use fit() instead of deprecated fit_generator()
            history = model.fit(
                batch_generator,
                epochs=1,
                verbose=0
            )
            train_loss += history.history['loss'][0]
            train_acc += history.history['accuracy'][0]
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue
    
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

test_data = DataGenerator(x_test_path, y_path=y_test_path, seq_len=15, batch_size=1)

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

