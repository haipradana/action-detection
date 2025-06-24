#!/usr/bin/env python3
"""
MERL Shopping Quick Start for Notebook
Copy these code blocks to your Jupyter notebook cells
"""

print("🎯 MERL SHOPPING NOTEBOOK QUICK START")
print("=" * 50)

# =====================================
# CELL 1: Install Dependencies
# =====================================
print("""
# CELL 1: Install Dependencies
%pip install tensorflow opencv-python scipy pandas numpy matplotlib scikit-learn huggingface-hub
""")

# =====================================
# CELL 2: Download Dataset  
# =====================================
print("""
# CELL 2: Download Dataset from HuggingFace
from huggingface_hub import snapshot_download

repo_id = "haipradana/merl-shopping-action-detection"
local_dir = "./MERL_Shopping_Dataset"

# Download dataset
snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)

print(f"Dataset downloaded to: {local_dir}")
!ls -la {local_dir}
""")

# =====================================
# CELL 3: Setup Folders
# =====================================
print("""
# CELL 3: Setup Folders & Check Dataset
import os
import glob

# Create output directories
os.makedirs('clips', exist_ok=True)
os.makedirs('dataframes', exist_ok=True) 
os.makedirs('flow_clips', exist_ok=True)

# Check dataset structure
videos_path = "./MERL_Shopping_Dataset/videos"
labels_path = "./MERL_Shopping_Dataset/labels"

print("📁 Dataset Structure:")
print(f"Videos path exists: {os.path.exists(videos_path)}")
print(f"Labels path exists: {os.path.exists(labels_path)}")

# Check for files in subfolders (train/val/test) and main folder
video_files = []
label_files = []

if os.path.exists(videos_path):
    for subfolder in ['train', 'val', 'test']:
        subfolder_path = os.path.join(videos_path, subfolder)
        if os.path.exists(subfolder_path):
            video_files.extend(glob.glob(os.path.join(subfolder_path, "*.mp4")))
    
    # Fallback: check main folder
    if not video_files:
        video_files = glob.glob(os.path.join(videos_path, "*.mp4"))
    
    print(f"✅ Found {len(video_files)} video files")

if os.path.exists(labels_path):
    for subfolder in ['train', 'val', 'test']:
        subfolder_path = os.path.join(labels_path, subfolder)
        if os.path.exists(subfolder_path):
            label_files.extend(glob.glob(os.path.join(subfolder_path, "*.mat")))
    
    # Fallback: check main folder
    if not label_files:
        label_files = glob.glob(os.path.join(labels_path, "*.mat"))
    
    print(f"✅ Found {len(label_files)} label files")

# Show subfolder structure
print("\\n📂 Subfolder structure:")
for path_name, path in [("Videos", videos_path), ("Labels", labels_path)]:
    if os.path.exists(path):
        subfolders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        print(f"   {path_name}: {subfolders}")
    else:
        print(f"   {path_name}: Not found")
""")

# =====================================
# CELL 4: Generate Sample Clips
# =====================================
print("""
# CELL 4: Generate Sample Clips (5 videos for testing)
!python utils/det2rec.py --start 1 --end 5
""")

# =====================================
# CELL 5: Verify Data
# =====================================
print("""
# CELL 5: Verify Generated Data
!python utils/test.py
""")

# =====================================
# CELL 6: Test Data Generator
# =====================================
print("""
# CELL 6: Test Data Generator
from merl_clips_data_gen import MerlClipsDataGenerator

print("🧪 Testing MerlClipsDataGenerator...")

# Create test generator
test_gen = MerlClipsDataGenerator(
    clips_base_path="clips",
    dataframes_path="dataframes", 
    seq_len=15,
    batch_size=2
)

print(f"\\n📊 Generator length: {len(test_gen)}")

# Test a batch
if len(test_gen) > 0:
    X, y = test_gen[0]
    print(f"✅ Batch shape: X={X.shape}, y={y.shape}")
    print(f"   Sample labels: {y}")
    print(f"   X range: [{X.min():.3f}, {X.max():.3f}]")
else:
    print("❌ No data available")
""")

# =====================================
# CELL 7: Quick Training Test
# =====================================
print("""
# CELL 7: Quick Training Test (3 epochs with small model)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import gc

print("🎯 QUICK TRAINING TEST")
print("=" * 40)

# GPU optimizations
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f"🔥 Found {len(physical_devices)} GPU(s)")

for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.backend.clear_session()
gc.collect()

# Simple model for testing
def create_simple_model(num_classes=5, seq_len=10):
    model = Sequential([
        ConvLSTM2D(32, (3, 3), return_sequences=False, 
                   input_shape=(seq_len, 224, 224, 3)),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Create and compile model
model = create_simple_model(5, 10)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print("✅ Simple model created")
model.summary()
""")

# =====================================
# CELL 8: Run Training
# =====================================
print("""
# CELL 8: Run Quick Training
# Create data generator
train_data = MerlClipsDataGenerator(
    clips_base_path="clips",
    dataframes_path="dataframes",
    seq_len=10,
    batch_size=2,
    target_size=(224, 224),
    shuffle=True,
    augment=True
)

print(f"✅ Data generator ready with {len(train_data.clip_data)} clips")

# Quick training (3 epochs, max 5 steps)
if len(train_data) > 0:
    print("\\n🚀 Starting quick training...")
    
    history = model.fit(
        train_data,
        epochs=3,
        steps_per_epoch=min(len(train_data), 5),
        verbose=1
    )
    
    print("✅ Quick training completed!")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
else:
    print("❌ No training data available")
""")

# =====================================
# CELL 9: Generate More Data & Full Training
# =====================================
print("""
# CELL 9: Generate More Data (Optional - for full training)
# Generate clips untuk 20 videos (will take longer)
!python utils/det2rec.py --start 1 --end 20

# Verify increased data
!python utils/test.py

# Run full training script
!python train_merl_clips.py
""")

# =====================================
# CELL 10: Check Results
# =====================================
print("""
# CELL 10: Check Training Results
import matplotlib.pyplot as plt
import os
from IPython.display import Image, display

# Check if training history plot exists
if os.path.exists('training_history_merl_clips.png'):
    print("📊 Training History:")
    display(Image('training_history_merl_clips.png'))
else:
    print("⚠️ Training history plot not found")

# Check if model exists
if os.path.exists('best_merl_clips_model.h5'):
    print("✅ Trained model saved: best_merl_clips_model.h5")
    
    # Load and check model
    model = tf.keras.models.load_model('best_merl_clips_model.h5')
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
else:
    print("⚠️ Trained model not found")
""")

print("\n🎉 ALL CODE BLOCKS READY!")
print("\n📋 Usage Instructions:")
print("1. Copy setiap code block ke cell terpisah di Jupyter notebook")
print("2. Run cells secara berurutan")
print("3. Start dengan CELL 1-5 untuk setup dan testing")
print("4. Run CELL 6-8 untuk quick training test")
print("5. Run CELL 9-10 untuk full training (optional)")
print("\n💡 Tips:")
print("- CELL 1-6 wajib dijalankan")
print("- CELL 7-8 untuk quick test (recommended)")  
print("- CELL 9-10 untuk full training (optional)")
print("- Monitor GPU memory dengan 'nvidia-smi'")
print("- Adjust batch_size jika out of memory") 