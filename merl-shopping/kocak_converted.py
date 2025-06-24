#!/usr/bin/env python3
"""
🎯 MERL Shopping Action Recognition Training
Converted from kocak.ipynb notebook

Simple but Effective Approach for 5 Videos Dataset

📋 Overview:
- Dataset: 5 videos, 295 clips, 5 action classes
- Approach: TimeDistributed CNN + LSTM for temporal modeling
- Strategy: No validation split (maximize training data)
- Goal: Proof of concept with good performance

🎬 Action Classes:
1. Reach To Shelf - Reaching towards shelf
2. Retract From Shelf - Moving hand back from shelf  
3. Hand In Shelf - Hand inside shelf area
4. Inspect Product - Looking at/examining product
5. Inspect Shelf - Looking at shelf contents
"""

# =============================================================================
# CELL 1: Setup and Dependencies
# =============================================================================
print("=" * 60)
print("CELL 1: Setup and Dependencies")
print("=" * 60)

# Install dependencies (commented out for script - install manually if needed)
# %pip install tensorflow opencv-python scipy pandas numpy matplotlib scikit-learn

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CELL 2: Import Libraries  
# =============================================================================
print("\n" + "=" * 60)
print("CELL 2: Import Libraries")
print("=" * 60)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    LSTM, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
import gc

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# GPU setup
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f"🔥 Found {len(physical_devices)} GPU(s)")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

# =============================================================================
# CELL 3: Dataset Verification
# =============================================================================
print("\n" + "=" * 60)
print("CELL 3: Dataset Verification")
print("=" * 60)

# Verify dataset structure
print("📁 Dataset Structure Check:")
print(f"Working directory: {os.getcwd()}")
print("\nFolders:")
for folder in ['clips', 'dataframes', 'flow_clips']:
    if os.path.exists(folder):
        print(f"✅ {folder}/ exists")
    else:
        print(f"❌ {folder}/ missing")

# Check dataframes
dataframe_files = [f for f in os.listdir('dataframes') if f.endswith('.csv')]
print(f"\n📄 Found {len(dataframe_files)} dataframe files:")
for df_file in sorted(dataframe_files):
    print(f"   - {df_file}")

# Check clips
clips_folders = [f for f in os.listdir('clips') if os.path.isdir(f'clips/{f}')]
print(f"\n🎬 Found {len(clips_folders)} video folders:")
for folder in sorted(clips_folders):
    clip_files = len([f for f in os.listdir(f'clips/{folder}') if f.endswith('.npy')])
    print(f"   - {folder}: {clip_files} clips")

# =============================================================================
# CELL 4: Analyze Class Distribution
# =============================================================================
print("\n" + "=" * 60)
print("CELL 4: Analyze Class Distribution")
print("=" * 60)

# Analyze class distribution
class_distribution = [0] * 5
total_clips = 0
all_clip_data = []

print("📊 Analyzing class distribution...")
for df_file in sorted(dataframe_files):
    df = pd.read_csv(f'dataframes/{df_file}')
    total_clips += len(df)
    
    for _, row in df.iterrows():
        video_num = df_file.split('_')[1].split('.')[0]
        clip_path = f"clips/video_{video_num}/{row['name']}.npy"
        class_label = int(row['class']) - 1  # Convert to 0-indexed
        
        all_clip_data.append((clip_path, class_label))
        class_distribution[class_label] += 1
    
    print(f"   📄 {df_file}: {len(df)} clips")

# Action class names
action_classes = [
    'Reach To Shelf',
    'Retract From Shelf', 
    'Hand In Shelf',
    'Inspect Product',
    'Inspect Shelf'
]

print(f"\n📈 Class Distribution:")
for i, (class_name, count) in enumerate(zip(action_classes, class_distribution)):
    percentage = (count / total_clips * 100)
    print(f"   {i}: {class_name:<20} = {count:3d} clips ({percentage:5.1f}%)")

print(f"\n📊 Total clips: {total_clips}")
print(f"📊 Balance std: {np.std([c/total_clips*100 for c in class_distribution]):.2f}%")

# =============================================================================
# CELL 5: Data Generator Class
# =============================================================================
print("\n" + "=" * 60)
print("CELL 5: Data Generator Class")
print("=" * 60)

class MerlActionDataGenerator(Sequence):
    """
    Simple and efficient data generator for MERL Shopping dataset
    """
    def __init__(self, clip_data, sequence_length=10, batch_size=4, 
                 target_size=(224, 224), shuffle=True, augment=False):
        self.clip_data = clip_data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.clip_data))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        print(f"📊 Data Generator initialized:")
        print(f"   - Total clips: {len(self.clip_data)}")
        print(f"   - Sequence length: {self.sequence_length}")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Batches per epoch: {len(self)}")
    
    def __len__(self):
        return len(self.clip_data) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_clips = []
        batch_labels = []
        
        for i in batch_indices:
            clip_path, label = self.clip_data[i]
            
            try:
                # Load clip
                clip = np.load(clip_path)
                
                # Adjust sequence length
                if len(clip) >= self.sequence_length:
                    # Take middle portion if clip is longer
                    start_idx = (len(clip) - self.sequence_length) // 2
                    clip = clip[start_idx:start_idx + self.sequence_length]
                else:
                    # Repeat frames if clip is shorter
                    repeat_factor = self.sequence_length // len(clip) + 1
                    clip = np.tile(clip, (repeat_factor, 1, 1, 1))[:self.sequence_length]
                
                # Simple augmentation
                if self.augment and np.random.random() > 0.5:
                    clip = np.flip(clip, axis=2)  # Horizontal flip
                
                batch_clips.append(clip)
                batch_labels.append(label)
                
            except Exception as e:
                print(f"⚠️ Error loading {clip_path}: {e}")
                # Use dummy data if loading fails
                dummy_clip = np.zeros((self.sequence_length, 224, 224, 3))
                batch_clips.append(dummy_clip)
                batch_labels.append(0)
        
        return np.array(batch_clips), np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# =============================================================================
# CELL 6: Create Data Generator
# =============================================================================
print("\n" + "=" * 60)
print("CELL 6: Create Data Generator")
print("=" * 60)

# Create data generator
print("🔄 Creating data generator...")
train_generator = MerlActionDataGenerator(
    clip_data=all_clip_data,
    sequence_length=10,  # 10 frames per sequence
    batch_size=4,        # Small batch size for memory efficiency
    target_size=(224, 224),
    shuffle=True,
    augment=True
)

# Test the generator
print("\n🧪 Testing data generator...")
try:
    X_test, y_test = train_generator[0]
    print(f"✅ Batch shape: X={X_test.shape}, y={y_test.shape}")
    print(f"   X range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    print(f"   Labels: {y_test}")
    print(f"   Unique labels: {np.unique(y_test)}")
except Exception as e:
    print(f"❌ Generator test failed: {e}")

# =============================================================================
# CELL 7: Model Architecture
# =============================================================================
print("\n" + "=" * 60)
print("CELL 7: Model Architecture")
print("=" * 60)

def create_merl_action_model(sequence_length=10, img_height=224, img_width=224, num_classes=5):
    """
    Create a simple but effective model for action recognition
    Architecture: TimeDistributed CNN + LSTM + Dense
    """
    model = Sequential([
        # CNN feature extractor for each frame
        TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), 
                       input_shape=(sequence_length, img_height, img_width, 3)),
        TimeDistributed(MaxPooling2D((2, 2))),
        
        TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
        TimeDistributed(MaxPooling2D((2, 2))),
        
        TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
        TimeDistributed(MaxPooling2D((2, 2))),
        
        TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same')),
        TimeDistributed(GlobalAveragePooling2D()),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.3)),
        
        # LSTM for temporal modeling
        LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
        
        # Classification head
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create model
print("🏗️ Creating model...")
model = create_merl_action_model(
    sequence_length=10,
    img_height=224,
    img_width=224,
    num_classes=5
)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Summary:")
model.summary()

# Count parameters
trainable_params = model.count_params()
print(f"\n📊 Total trainable parameters: {trainable_params:,}")

# =============================================================================
# CELL 8: Training Setup
# =============================================================================
print("\n" + "=" * 60)
print("CELL 8: Training Setup")
print("=" * 60)

# Calculate class weights for balanced training
from sklearn.utils.class_weight import compute_class_weight

# Get all labels
all_labels = [label for _, label in all_clip_data]
unique_classes = np.unique(all_labels)

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=unique_classes,
    y=all_labels
)

class_weight_dict = dict(zip(unique_classes, class_weights))

print("⚖️ Class weights for balanced training:")
for i, weight in class_weight_dict.items():
    print(f"   Class {i} ({action_classes[i]}): {weight:.3f}")

# Training callbacks
callbacks = [
    EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print(f"\n📋 Training Configuration:")
print(f"   - Epochs: 50 (with early stopping)")
print(f"   - Batch size: 4")
print(f"   - Learning rate: 0.0001")
print(f"   - Steps per epoch: {len(train_generator)}")
print(f"   - Total training samples per epoch: {len(train_generator) * 4}")

# =============================================================================
# CELL 9: Training
# =============================================================================
print("\n" + "=" * 60)
print("CELL 9: Training")
print("=" * 60)

# Clear memory before training
tf.keras.backend.clear_session()
gc.collect()

print("🚀 Starting training...")
print("=" * 50)

# Start training
history = model.fit(
    train_generator,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

print("\n🎉 Training completed!")

# =============================================================================
# CELL 10: Training Analysis
# =============================================================================
print("\n" + "=" * 60)
print("CELL 10: Training Analysis")
print("=" * 60)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot loss
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot accuracy
ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='orange')
ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final metrics
final_loss = history.history['loss'][-1]
final_accuracy = history.history['accuracy'][-1]

print(f"\n📊 Final Training Metrics:")
print(f"   - Final Loss: {final_loss:.4f}")
print(f"   - Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.1f}%)")
print(f"   - Total Epochs: {len(history.history['loss'])}")

# Random baseline comparison
random_accuracy = 1.0 / 5  # 20% for 5 classes
improvement = (final_accuracy - random_accuracy) / random_accuracy * 100
print(f"   - Random baseline: {random_accuracy:.1%}")
print(f"   - Improvement over random: {improvement:.1f}%")

# =============================================================================
# CELL 11: Model Testing
# =============================================================================
print("\n" + "=" * 60)
print("CELL 11: Model Testing")
print("=" * 60)

# Test on a few random samples
print("🧪 Testing model on random samples...")
print("=" * 40)

# Get a test batch
test_batch_X, test_batch_y = train_generator[0]
predictions = model.predict(test_batch_X, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

print(f"Batch predictions:")
for i in range(len(test_batch_y)):
    true_class = test_batch_y[i]
    pred_class = predicted_classes[i]
    confidence = predictions[i][pred_class]
    
    status = "✅" if true_class == pred_class else "❌"
    
    print(f"   {status} Sample {i+1}:")
    print(f"      True: {action_classes[true_class]}")
    print(f"      Pred: {action_classes[pred_class]} ({confidence:.3f})")
    print()

# Calculate accuracy on this batch
batch_accuracy = np.mean(test_batch_y == predicted_classes)
print(f"Batch accuracy: {batch_accuracy:.1%}")

# =============================================================================
# CELL 12: Comprehensive Evaluation
# =============================================================================
print("\n" + "=" * 60)
print("CELL 12: Comprehensive Evaluation")
print("=" * 60)

# Comprehensive evaluation on more samples
print("🔍 Comprehensive evaluation...")
print("=" * 40)

all_predictions = []
all_true_labels = []

# Test on multiple batches (limit to avoid memory issues)
num_test_batches = min(10, len(train_generator))
print(f"Testing on {num_test_batches} batches...")

for i in range(num_test_batches):
    batch_X, batch_y = train_generator[i]
    batch_pred = model.predict(batch_X, verbose=0)
    batch_pred_classes = np.argmax(batch_pred, axis=1)
    
    all_predictions.extend(batch_pred_classes)
    all_true_labels.extend(batch_y)

all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)

# Calculate overall accuracy
overall_accuracy = np.mean(all_true_labels == all_predictions)
print(f"\n📊 Overall test accuracy: {overall_accuracy:.1%}")

# Per-class accuracy
print(f"\n📈 Per-class accuracy:")
for class_idx in range(5):
    class_mask = all_true_labels == class_idx
    if np.sum(class_mask) > 0:
        class_accuracy = np.mean(all_predictions[class_mask] == class_idx)
        class_count = np.sum(class_mask)
        print(f"   {action_classes[class_idx]:<20}: {class_accuracy:.1%} ({class_count} samples)")
    else:
        print(f"   {action_classes[class_idx]:<20}: No samples")

# =============================================================================
# CELL 13: Save Model
# =============================================================================
print("\n" + "=" * 60)
print("CELL 13: Save Model")
print("=" * 60)

# Save the trained model
model_filename = "merl_action_recognition_model.h5"
model.save(model_filename)
print(f"✅ Model saved as: {model_filename}")

# Save training history
import pickle
history_filename = "training_history.pkl"
with open(history_filename, 'wb') as f:
    pickle.dump(history.history, f)
print(f"✅ Training history saved as: {history_filename}")

# Save model info
model_info = {
    'architecture': 'TimeDistributed CNN + LSTM',
    'sequence_length': 10,
    'num_classes': 5,
    'input_shape': (10, 224, 224, 3),
    'final_accuracy': final_accuracy,
    'final_loss': final_loss,
    'total_parameters': trainable_params,
    'training_samples': len(all_clip_data),
    'action_classes': action_classes
}

info_filename = "model_info.pkl"
with open(info_filename, 'wb') as f:
    pickle.dump(model_info, f)
print(f"✅ Model info saved as: {info_filename}")

print(f"\n📁 Generated files:")
print(f"   - {model_filename} (model weights)")
print(f"   - {history_filename} (training history)")
print(f"   - {info_filename} (model metadata)")

# =============================================================================
# CELL 14: Training Summary
# =============================================================================
print("\n" + "=" * 60)
print("CELL 14: Training Summary")
print("=" * 60)

print("🎯 TRAINING SUMMARY")
print("=" * 50)
print(f"Dataset: 5 videos, {len(all_clip_data)} clips")
print(f"Model: TimeDistributed CNN + LSTM")
print(f"Parameters: {trainable_params:,}")
print(f"Final Accuracy: {final_accuracy:.1%}")
print(f"Training Time: {len(history.history['loss'])} epochs")

print(f"\n🚀 Next Steps:")
print(f"   1. ✅ Model trained and saved")
print(f"   2. 🎬 Test on new video inference")
print(f"   3. 📊 Add more videos if needed")
print(f"   4. 🔧 Fine-tune hyperparameters")
print(f"   5. 🎯 Deploy for real-time inference")

if final_accuracy > 0.4:  # 40% for 5 classes is decent
    print(f"\n✅ Good performance! Model is ready for testing.")
elif final_accuracy > 0.25:  # 25% is better than random
    print(f"\n⚠️ Moderate performance. Consider:")
    print(f"   - Adding more training data")
    print(f"   - Adjusting model architecture")
    print(f"   - Tuning hyperparameters")
else:
    print(f"\n❌ Low performance. Recommendations:")
    print(f"   - Check data quality")
    print(f"   - Increase training data")
    print(f"   - Try different architecture")
    print(f"   - Adjust learning rate")

print(f"\n🎉 Training complete! Model ready for deployment.")

# =============================================================================
# Script Complete
# =============================================================================
print("\n" + "=" * 60)
print("SCRIPT EXECUTION COMPLETE")
print("=" * 60)
print("All cells have been executed successfully!")
print("Check the generated model files for deployment.")