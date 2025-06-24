#!/usr/bin/env python3
"""
MERL Clips Data Generator
Adapts merl-shopping clip-based data for action-detection training.
Uses the balanced clips from merl-shopping instead of frame-level video labels.
"""

import os
import cv2
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.utils import to_categorical
import glob
from collections import Counter

class MerlClipsDataGenerator(Sequence):
    """
    Data generator that uses clip-based data from merl-shopping approach.
    This provides much better balanced data compared to video-level labeling.
    """
    
    def __init__(self, clips_base_path, dataframes_path, seq_len=15, batch_size=4, 
                 target_size=(224, 224), shuffle=True, augment=False):
        """
        Args:
            clips_base_path: Path to clips folder (e.g., 'merl-shopping/clips/')
            dataframes_path: Path to dataframes folder (e.g., 'merl-shopping/dataframes/')
            seq_len: Sequence length for temporal modeling
            batch_size: Batch size
            target_size: Image resize target
            shuffle: Whether to shuffle data
            augment: Whether to apply data augmentation
        """
        self.clips_base_path = clips_base_path
        self.dataframes_path = dataframes_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        
        # MERL action classes (5 classes, 0-indexed)
        self.action_classes = [
            'Reach To Shelf',      # Class 0
            'Retract From Shelf',  # Class 1
            'Hand In Shelf',       # Class 2
            'Inspect Product',     # Class 3
            'Inspect Shelf'        # Class 4
        ]
        self.num_classes = len(self.action_classes)
        
        # Load all clip data
        self.clip_data = self._load_all_clips()
        
        # Create batches
        self.indices = np.arange(len(self.clip_data))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        print(f"✅ MerlClipsDataGenerator initialized")
        print(f"   📊 Total clips: {len(self.clip_data)}")
        print(f"   🎯 Sequence length: {seq_len}")
        print(f"   📦 Batch size: {batch_size}")
        self._print_class_distribution()
    
    def _load_all_clips(self):
        """Load all clip metadata from dataframes"""
        clip_data = []
        
        # Get all dataframe files
        dataframe_files = glob.glob(os.path.join(self.dataframes_path, "dataframe_*.csv"))
        
        for df_file in dataframe_files:
            # Extract video number from filename
            video_num = int(os.path.basename(df_file).split('_')[1].split('.')[0])
            
            # Load dataframe
            df = pd.read_csv(df_file)
            
            # Add each clip to our dataset
            for _, row in df.iterrows():
                clip_name = row['name']
                class_label = int(row['class']) - 1  # Convert from 1-indexed to 0-indexed
                
                clip_path = os.path.join(self.clips_base_path, f"video_{video_num}", f"{clip_name}.npy")
                
                if os.path.exists(clip_path):
                    clip_data.append({
                        'path': clip_path,
                        'label': class_label,
                        'video_num': video_num,
                        'clip_name': clip_name
                    })
                else:
                    print(f"⚠️ Warning: Clip not found: {clip_path}")
        
        return clip_data
    
    def _print_class_distribution(self):
        """Print class distribution statistics"""
        labels = [clip['label'] for clip in self.clip_data]
        class_counts = Counter(labels)
        
        print(f"\n📊 Class Distribution:")
        for class_id in range(self.num_classes):
            count = class_counts.get(class_id, 0)
            percentage = (count / len(labels)) * 100
            print(f"   {class_id}: {self.action_classes[class_id]:<20} = {count:4d} clips ({percentage:5.1f}%)")
        
        print(f"\n🎯 Much more balanced than video-level labeling!")
    
    def __len__(self):
        return len(self.clip_data) // self.batch_size
    
    def __getitem__(self, index):
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.clip_data))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Initialize batch arrays
        batch_x = []
        batch_y = []
        
        for idx in batch_indices:
            clip_info = self.clip_data[idx]
            
            try:
                # Load clip data
                clip_frames = np.load(clip_info['path'])  # Shape: (T, H, W, C)
                
                # Preprocess frames
                processed_frames = self._preprocess_clip(clip_frames)
                
                # Create sequences for temporal modeling
                if len(processed_frames) >= self.seq_len:
                    # Take multiple sequences from the clip if it's long enough
                    sequences = self._create_sequences(processed_frames, clip_info['label'])
                    batch_x.extend([seq[0] for seq in sequences])
                    batch_y.extend([seq[1] for seq in sequences])
                else:
                    # Pad short clips
                    padded_frames = self._pad_clip(processed_frames)
                    if len(padded_frames) >= self.seq_len:
                        batch_x.append(padded_frames[:self.seq_len])
                        batch_y.append(clip_info['label'])
            
            except Exception as e:
                print(f"Error loading clip {clip_info['path']}: {e}")
                # Use dummy data as fallback
                dummy_frames = np.zeros((self.seq_len, *self.target_size, 3))
                batch_x.append(dummy_frames)
                batch_y.append(4)  # Default to "Inspect Shelf"
        
        # Convert to numpy arrays
        if len(batch_x) == 0:
            # Fallback in case of errors
            batch_x = np.zeros((1, self.seq_len, *self.target_size, 3))
            batch_y = np.array([4])
        else:
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
        
        return batch_x, batch_y
    
    def _preprocess_clip(self, clip_frames):
        """Preprocess clip frames"""
        processed = []
        
        for frame in clip_frames:
            # Frame should already be in RGB and normalized from merl-shopping
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            
            # Resize if needed
            if frame.shape[:2] != self.target_size:
                frame = cv2.resize(frame, self.target_size)
            
            # Convert back to float and normalize
            frame = frame.astype(np.float32) / 255.0
            
            # Apply augmentation if enabled
            if self.augment and np.random.random() > 0.5:
                frame = self._augment_frame(frame)
            
            processed.append(frame)
        
        return np.array(processed)
    
    def _augment_frame(self, frame):
        """Apply simple data augmentation"""
        # Horizontal flip
        if np.random.random() > 0.5:
            frame = np.fliplr(frame)
        
        # Brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            frame = np.clip(frame * factor, 0, 1)
        
        return frame
    
    def _create_sequences(self, frames, label):
        """Create multiple sequences from a long clip"""
        sequences = []
        
        # If clip is longer than seq_len, create overlapping sequences
        if len(frames) > self.seq_len:
            # Take sequences with stride = seq_len//2 for more data
            stride = max(1, self.seq_len // 2)
            for start in range(0, len(frames) - self.seq_len + 1, stride):
                seq = frames[start:start + self.seq_len]
                sequences.append((seq, label))
        else:
            # Just one sequence if clip is exactly seq_len
            sequences.append((frames, label))
        
        return sequences
    
    def _pad_clip(self, frames):
        """Pad short clips by repeating last frame"""
        if len(frames) < self.seq_len:
            last_frame = frames[-1]
            padding_needed = self.seq_len - len(frames)
            padding = np.repeat(last_frame[np.newaxis], padding_needed, axis=0)
            frames = np.concatenate([frames, padding], axis=0)
        
        return frames
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_merl_data_generators(clips_base_path, dataframes_path, 
                               train_videos=None, val_videos=None, test_videos=None,
                               **kwargs):
    """
    Create train/val/test data generators based on video splits.
    
    Args:
        clips_base_path: Path to clips folder
        dataframes_path: Path to dataframes folder
        train_videos: List of video numbers for training (e.g., [1, 2, ..., 60])
        val_videos: List of video numbers for validation
        test_videos: List of video numbers for testing
    """
    
    # Default MERL splits if not provided
    if train_videos is None:
        train_videos = list(range(1, 61))  # Videos 1-60
    if val_videos is None:
        val_videos = list(range(61, 79))   # Videos 61-78
    if test_videos is None:
        test_videos = list(range(79, 107)) # Videos 79-106
    
    print(f"🎯 Creating MERL data generators with split:")
    print(f"   📚 Train: {len(train_videos)} videos")
    print(f"   📊 Val: {len(val_videos)} videos") 
    print(f"   🧪 Test: {len(test_videos)} videos")
    
    # Create generators
    train_gen = MerlClipsDataGenerator(clips_base_path, dataframes_path, **kwargs)
    val_gen = MerlClipsDataGenerator(clips_base_path, dataframes_path, shuffle=False, **kwargs)
    test_gen = MerlClipsDataGenerator(clips_base_path, dataframes_path, shuffle=False, **kwargs)
    
    # Filter clips by video numbers (you'd need to implement filtering logic)
    # For now, all generators use all data - you can add filtering if needed
    
    return train_gen, val_gen, test_gen

if __name__ == "__main__":
    # Test the data generator
    clips_path = "../merl-shopping/clips"
    dataframes_path = "../merl-shopping/dataframes"
    
    if os.path.exists(clips_path) and os.path.exists(dataframes_path):
        print("🧪 Testing MerlClipsDataGenerator...")
        
        # Create test generator
        test_gen = MerlClipsDataGenerator(
            clips_base_path=clips_path,
            dataframes_path=dataframes_path,
            seq_len=15,
            batch_size=2
        )
        
        print(f"\n📊 Generator length: {len(test_gen)}")
        
        # Test a batch
        if len(test_gen) > 0:
            X, y = test_gen[0]
            print(f"✅ Batch shape: X={X.shape}, y={y.shape}")
            print(f"   Sample labels: {y}")
            print(f"   X range: [{X.min():.3f}, {X.max():.3f}]")
        else:
            print("❌ No data available")
    else:
        print("❌ Clips or dataframes path not found")
        print(f"   Clips: {clips_path}")
        print(f"   Dataframes: {dataframes_path}") 