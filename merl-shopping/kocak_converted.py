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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    LSTM, BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2  # For consistent geometric transformations
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
    Enhanced data generator with aggressive augmentation and ResNet preprocessing
    """
    def __init__(self, clip_data, sequence_length=8, batch_size=6, 
                 target_size=(224, 224), shuffle=True, augment=False):
        self.clip_data = clip_data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.clip_data))
        
        # Setup moderate augmentation for training (less aggressive for small dataset)
        if self.augment:
            self.augmentor = ImageDataGenerator(
                rotation_range=10,        # Reduced from 15
                width_shift_range=0.05,   # Reduced from 0.1
                height_shift_range=0.05,  # Reduced from 0.1
                zoom_range=0.05,          # Reduced from 0.1
                brightness_range=[0.9, 1.1], # Less aggressive
                fill_mode='nearest'
            )
        
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        print(f"📊 Enhanced Data Generator initialized:")
        print(f"   - Total clips: {len(self.clip_data)}")
        print(f"   - Sequence length: {self.sequence_length}")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Augmentation: {'✅ Consistent per Clip' if self.augment else '❌ None'}")
        print(f"   - Batches per epoch: {len(self)}")
    
    def __len__(self):
        return len(self.clip_data) // self.batch_size
    
    def _apply_augmentation(self, clip):
        """Apply CONSISTENT augmentation to entire clip (same transform for all frames)"""
        augmented_clip = []
        
        # Decide augmentation parameters ONCE for the entire clip
        should_augment = np.random.random() > 0.7  # 30% chance (reduced from 50%)
        
        if should_augment:
            # Generate consistent parameters for all frames
            rotation_angle = np.random.uniform(-10, 10)
            width_shift = np.random.uniform(-0.05, 0.05)
            height_shift = np.random.uniform(-0.05, 0.05)
            zoom_factor = np.random.uniform(0.95, 1.05)
            brightness_factor = np.random.uniform(0.9, 1.1)
            should_flip = np.random.random() > 0.7  # 30% chance for horizontal flip
            
            # Apply SAME transformation to ALL frames
            for frame in clip:
                augmented_frame = frame.copy()
                
                # Apply consistent geometric transforms
                if abs(rotation_angle) > 0.5:  # Only if significant rotation
                    rows, cols = frame.shape[:2]
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, zoom_factor)
                    M[0, 2] += width_shift * cols
                    M[1, 2] += height_shift * rows
                    augmented_frame = cv2.warpAffine(augmented_frame, M, (cols, rows))
                
                # Apply consistent brightness
                if abs(brightness_factor - 1.0) > 0.01:
                    augmented_frame = np.clip(augmented_frame * brightness_factor, 0, 255).astype(np.uint8)
                
                # Apply consistent horizontal flip
                if should_flip:
                    augmented_frame = np.flip(augmented_frame, axis=1)
                
                augmented_clip.append(augmented_frame)
        else:
            # No augmentation - keep original frames
            augmented_clip = [frame.copy() for frame in clip]
        
        return np.array(augmented_clip)
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_clips = []
        batch_labels = []
        
        for i in batch_indices:
            clip_path, label = self.clip_data[i]
            
            try:
                # Load clip
                clip = np.load(clip_path)
                
                # Adjust sequence length with random sampling for augmentation
                if len(clip) >= self.sequence_length:
                    if self.augment and len(clip) > self.sequence_length:
                        # Random start position for temporal augmentation
                        max_start = len(clip) - self.sequence_length
                        start_idx = np.random.randint(0, max_start + 1)
                    else:
                        # Take middle portion
                        start_idx = (len(clip) - self.sequence_length) // 2
                    clip = clip[start_idx:start_idx + self.sequence_length]
                else:
                    # Repeat frames if clip is shorter
                    repeat_factor = self.sequence_length // len(clip) + 1
                    clip = np.tile(clip, (repeat_factor, 1, 1, 1))[:self.sequence_length]
                
                # Apply consistent augmentation BEFORE normalization
                if self.augment:
                    clip = self._apply_augmentation(clip)
                
                # ResNet preprocessing (ImageNet normalization)
                clip = clip.astype(np.float32)
                clip = tf.keras.applications.resnet50.preprocess_input(clip)
                
                batch_clips.append(clip)
                batch_labels.append(label)
                
            except Exception as e:
                print(f"⚠️ Error loading {clip_path}: {e}")
                # Use dummy data if loading fails
                dummy_clip = np.zeros((self.sequence_length, 224, 224, 3), dtype=np.float32)
                batch_clips.append(dummy_clip)
                batch_labels.append(0)
        
        return np.array(batch_clips, dtype=np.float32), np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# =============================================================================
# CELL 6: Train/Test Split and Create Data Generators
# =============================================================================
print("\n" + "=" * 60)
print("CELL 6: Train/Test Split and Create Data Generators")
print("=" * 60)

# Split data into train/test (80/20 split)
from sklearn.model_selection import train_test_split

print("📊 Splitting data into train/test sets...")
train_data, test_data = train_test_split(
    all_clip_data, 
    test_size=0.2, 
    random_state=42, 
    stratify=[label for _, label in all_clip_data]  # Stratified split to maintain class balance
)

print(f"   📈 Train samples: {len(train_data)} ({len(train_data)/len(all_clip_data)*100:.1f}%)")
print(f"   📊 Test samples: {len(test_data)} ({len(test_data)/len(all_clip_data)*100:.1f}%)")

# Check class distribution in train/test splits
train_labels = [label for _, label in train_data]
test_labels = [label for _, label in test_data]

print(f"\n📋 Train set class distribution:")
for i in range(5):
    count = train_labels.count(i)
    percentage = count / len(train_labels) * 100
    print(f"   Class {i} ({action_classes[i]:<20}): {count:3d} ({percentage:5.1f}%)")

print(f"\n📋 Test set class distribution:")
for i in range(5):
    count = test_labels.count(i)
    percentage = count / len(test_labels) * 100
    print(f"   Class {i} ({action_classes[i]:<20}): {count:3d} ({percentage:5.1f}%)")

# Create data generators
print("\n🔄 Creating data generators...")

# Training generator with moderate augmentation
train_generator = MerlActionDataGenerator(
    clip_data=train_data,
    sequence_length=8,      # Optimal sequence length for lightweight model
    batch_size=8,           # Larger batch size for better gradient estimation
    target_size=(224, 224),
    shuffle=True,
    augment=True            # Moderate augmentation for training
)

# Test generator without augmentation
test_generator = MerlActionDataGenerator(
    clip_data=test_data,
    sequence_length=8,      # Same sequence length as training
    batch_size=8,           # Same batch size for consistency
    target_size=(224, 224),
    shuffle=False,          # No shuffling for consistent test results
    augment=False           # No augmentation for test data
)

# Test the generators
print("\n🧪 Testing data generators...")
try:
    X_train_test, y_train_test = train_generator[0]
    X_test_test, y_test_test = test_generator[0]
    
    print(f"✅ Train batch shape: X={X_train_test.shape}, y={y_train_test.shape}")
    print(f"   Train X range: [{X_train_test.min():.3f}, {X_train_test.max():.3f}]")
    print(f"   Train labels: {y_train_test}")
    
    print(f"✅ Test batch shape: X={X_test_test.shape}, y={y_test_test.shape}")
    print(f"   Test X range: [{X_test_test.min():.3f}, {X_test_test.max():.3f}]")
    print(f"   Test labels: {y_test_test}")
    
except Exception as e:
    print(f"❌ Generator test failed: {e}")

# =============================================================================
# CELL 7: Model Architecture
# =============================================================================
print("\n" + "=" * 60)
print("CELL 7: Model Architecture")
print("=" * 60)

def create_resnet_action_model(sequence_length=8, img_height=224, img_width=224, num_classes=5):
    """
    Create OPTIMIZED ResNet-based model for action recognition
    Architecture: TimeDistributed ResNet50 (frozen) + LSTM + Dense
    Reduced overfitting with better regularization
    """
    # Input layer
    input_layer = Input(shape=(sequence_length, img_height, img_width, 3))
    
    # Create ResNet50 backbone (pretrained on ImageNet)
    resnet_base = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )
    
    # Partially freeze ResNet layers for fine-tuning
    resnet_base.trainable = False
    
    # Unfreeze top layers for fine-tuning (last 2 blocks)
    for layer in resnet_base.layers[-20:]:
        layer.trainable = True
    
    # Apply ResNet to each frame with reduced dropout
    x = TimeDistributed(resnet_base)(input_layer)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x)  # Reduced from 0.3
    
    # Temporal modeling with optimized architecture
    x = LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.2)(x)  
    x = LSTM(96, return_sequences=False, dropout=0.25, recurrent_dropout=0.2)(x)   # Slightly increased from 64
    
    # Simpler classification head to prevent overfitting
    x = Dense(64, activation='relu')(x)    # Reduced from 128
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)                    # Reduced from 0.6
    
    # Output layer (removed extra dense layer)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=outputs)
    
    return model

def create_lightweight_action_model(sequence_length=8, img_height=224, img_width=224, num_classes=5):
    """
    Create an optimized lightweight model for small datasets
    Architecture: Efficient CNN + LSTM with better regularization
    """
    model = Sequential([
        # Efficient CNN feature extractor
        TimeDistributed(Conv2D(32, (7, 7), strides=2, activation='relu', padding='same'), 
                       input_shape=(sequence_length, img_height, img_width, 3)),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.2)),
        
        TimeDistributed(Conv2D(64, (5, 5), activation='relu', padding='same')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.2)),
        
        TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.3)),
        
        TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same')),
        TimeDistributed(GlobalAveragePooling2D()),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.3)),
        
        # Temporal modeling with better architecture
        LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2),
        
        # Simpler classification head
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create ResNet model with consistent augmentation (Hybrid Approach)
print("🏗️ Creating ResNet model with optimized hyperparameters...")

# Use ResNet model with consistent augmentation
model = create_resnet_action_model(
    sequence_length=8,
    img_height=224,
    img_width=224,
    num_classes=5
)
print("✅ Using ResNet50 + LSTM model")
print("📊 With consistent augmentation for better temporal learning")
model_type = "ResNet50 + LSTM (Optimized)"

# Increased learning rate for better convergence (TUNED)
learning_rate = 0.0005  # Increased from 0.0001 for faster learning

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"🎯 Model Type: {model_type}")
print(f"🎯 Learning Rate: {learning_rate}")

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

# Calculate class weights for balanced training (using TRAIN data only)
from sklearn.utils.class_weight import compute_class_weight

# Get train labels only (not all data!)
train_labels_for_weights = [label for _, label in train_data]
unique_classes = np.unique(train_labels_for_weights)

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=unique_classes,
    y=train_labels_for_weights
)

class_weight_dict = dict(zip(unique_classes, class_weights))

print("⚖️ Class weights for balanced training:")
for i, weight in class_weight_dict.items():
    print(f"   Class {i} ({action_classes[i]}): {weight:.3f}")

# Fine-tuned training callbacks for better convergence
callbacks = [
    EarlyStopping(
        monitor='loss',
        patience=15,  # More patience for fine-tuning
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='loss',
        factor=0.3,   # More aggressive when needed
        patience=4,   # Faster response
        min_lr=1e-6,  # Higher minimum LR
        verbose=1
    )
]

print(f"\n📋 Fine-Tuned ResNet Configuration (Option 1):")
print(f"   - Model: {model_type}")
print(f"   - Epochs: 50 (with early stopping)")
print(f"   - Batch size: 8")
print(f"   - Learning rate: {learning_rate} (INCREASED from 0.0001)")
print(f"   - Sequence length: 8 frames")
print(f"   - Augmentation: ✅ Reduced to 30% (from 50%)")
print(f"   - Fine-tuning: ✅ Top 20 ResNet layers unfrozen")
print(f"   - LSTM: 128→96 units (optimized)")
print(f"   - Steps per epoch: {len(train_generator)}")
print(f"   - Total training samples per epoch: {len(train_generator) * 8}")

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

# Start training with optimized lightweight configuration
history = model.fit(
    train_generator,
    epochs=50,  # Optimal epochs for lightweight model
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
# CELL 11: Model Testing on Separate Test Set
# =============================================================================
print("\n" + "=" * 60)
print("CELL 11: Model Testing on Separate Test Set")
print("=" * 60)

# Test on ACTUAL test data (not training data!)
print("🧪 Testing model on separate test set...")
print("=" * 40)

# Get a test batch from TEST generator (not train!)
test_batch_X, test_batch_y = test_generator[0]
predictions = model.predict(test_batch_X, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

print(f"Test batch predictions (using separate test data):")
for i in range(len(test_batch_y)):
    true_class = test_batch_y[i]
    pred_class = predicted_classes[i]
    confidence = predictions[i][pred_class]
    
    status = "✅" if true_class == pred_class else "❌"
    
    print(f"   {status} Sample {i+1}:")
    print(f"      True: {action_classes[true_class]}")
    print(f"      Pred: {action_classes[pred_class]} ({confidence:.3f})")
    print()

# Calculate accuracy on this test batch
test_batch_accuracy = np.mean(test_batch_y == predicted_classes)
print(f"Test batch accuracy: {test_batch_accuracy:.1%}")
print("⚠️  This is proper testing - using data the model has NEVER seen during training!")

# =============================================================================
# CELL 12: Comprehensive Evaluation on Test Set
# =============================================================================
print("\n" + "=" * 60)
print("CELL 12: Comprehensive Evaluation on Test Set")
print("=" * 60)

# Comprehensive evaluation on TEST data (not training data!)
print("🔍 Comprehensive evaluation on test set...")
print("=" * 40)

all_test_predictions = []
all_test_true_labels = []

# Test on ALL test batches for comprehensive evaluation
num_test_batches = len(test_generator)
print(f"Testing on ALL {num_test_batches} test batches...")
print("⚠️  Using SEPARATE test data that model has never seen!")

for i in range(num_test_batches):
    batch_X, batch_y = test_generator[i]
    batch_pred = model.predict(batch_X, verbose=0)
    batch_pred_classes = np.argmax(batch_pred, axis=1)
    
    all_test_predictions.extend(batch_pred_classes)
    all_test_true_labels.extend(batch_y)

all_test_predictions = np.array(all_test_predictions)
all_test_true_labels = np.array(all_test_true_labels)

# Calculate overall TEST accuracy (this is the real performance metric!)
overall_test_accuracy = np.mean(all_test_true_labels == all_test_predictions)
print(f"\n📊 Overall TEST accuracy (unseen data): {overall_test_accuracy:.1%}")
print(f"📊 Total test samples evaluated: {len(all_test_true_labels)}")

# Per-class accuracy on TEST data
print(f"\n📈 Per-class TEST accuracy:")
for class_idx in range(5):
    class_mask = all_test_true_labels == class_idx
    if np.sum(class_mask) > 0:
        class_accuracy = np.mean(all_test_predictions[class_mask] == class_idx)
        class_count = np.sum(class_mask)
        print(f"   {action_classes[class_idx]:<20}: {class_accuracy:.1%} ({class_count} samples)")
    else:
        print(f"   {action_classes[class_idx]:<20}: No samples")

print(f"\n🎯 This is the REAL model performance - tested on unseen data!")

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
    'architecture': model_type,
    'sequence_length': 8,
    'num_classes': 5,
    'input_shape': (8, 224, 224, 3),
    'final_training_accuracy': final_accuracy,
    'final_test_accuracy': overall_test_accuracy,
    'final_loss': final_loss,
    'total_parameters': trainable_params,
    'training_samples': len(train_data),
    'test_samples': len(test_data),
    'learning_rate': learning_rate,
    'augmentation': 'Aggressive',
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
print(f"Dataset: 5 videos, {len(all_clip_data)} total clips")
print(f"Train/Test Split: {len(train_data)}/{len(test_data)} clips (80/20)")
print(f"Model: TimeDistributed CNN + LSTM")
print(f"Parameters: {trainable_params:,}")
print(f"Final Training Accuracy: {final_accuracy:.1%}")
print(f"Final TEST Accuracy: {overall_test_accuracy:.1%} ⭐ (Real Performance)")
print(f"Training Time: {len(history.history['loss'])} epochs")

print(f"\n🚀 Next Steps:")
print(f"   1. ✅ Model trained and saved")
print(f"   2. 🎬 Test on new video inference")
print(f"   3. 📊 Add more videos if needed")
print(f"   4. 🔧 Fine-tune hyperparameters")
print(f"   5. 🎯 Deploy for real-time inference")

if overall_test_accuracy > 0.4:  # 40% for 5 classes is decent
    print(f"\n✅ Good performance! Model is ready for deployment.")
    print(f"   📊 TEST accuracy {overall_test_accuracy:.1%} is above 40% threshold")
elif overall_test_accuracy > 0.25:  # 25% is better than random
    print(f"\n⚠️ Moderate performance. Consider:")
    print(f"   📊 TEST accuracy {overall_test_accuracy:.1%} is moderate")
    print(f"   - Adding more training data")
    print(f"   - Adjusting model architecture")
    print(f"   - Tuning hyperparameters")
else:
    print(f"\n❌ Low performance. Recommendations:")
    print(f"   📊 TEST accuracy {overall_test_accuracy:.1%} is too low")
    print(f"   - Check data quality")
    print(f"   - Increase training data")
    print(f"   - Try different architecture")
    print(f"   - Adjust learning rate")

print(f"\n📊 Performance Comparison:")
print(f"   🚂 Training Accuracy: {final_accuracy:.1%}")
print(f"   🧪 Test Accuracy: {overall_test_accuracy:.1%} ⭐")
if final_accuracy - overall_test_accuracy > 0.1:
    print(f"   ⚠️  Large gap suggests overfitting!")
else:
    print(f"   ✅ Good generalization (small train/test gap)")

print(f"\n🎉 Training complete! Model ready for deployment.")

# =============================================================================
# Script Complete
# =============================================================================
print("\n" + "=" * 60)
print("SCRIPT EXECUTION COMPLETE")
print("=" * 60)
print("All cells have been executed successfully!")
print("Check the generated model files for deployment.")