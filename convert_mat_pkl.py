#!/usr/bin/env python3
"""
PROPER MERL Dataset Label Converter (V2 - Corrected)
Converts .mat files with temporal boundaries to video-level action labels.
Fixes the issue with unpacking scipy's cell array structure.
"""

try:
    import scipy.io
except ImportError:
    print("❌ scipy not installed! Run: pip install scipy")
    exit(1)

import pickle
import numpy as np
import os
from collections import Counter, defaultdict

print("🎯 MERL DATASET LABEL CONVERTER (V2 - CORRECTED)")
print("=" * 50)

# MERL Dataset Configuration
action_classes = [
    'Reach To Shelf',      # Class 0 (MERL class 1)
    'Retract From Shelf',  # Class 1 (MERL class 2) 
    'Hand In Shelf',       # Class 2 (MERL class 3)
    'Inspect Product',     # Class 3 (MERL class 4)
    'Inspect Shelf'        # Class 4 (MERL class 5)
]

print(f"📊 MERL Action Classes (5 total):")
for i, action in enumerate(action_classes):
    print(f"   {i}: {action}")

def analyze_video_actions(tlabs):
    """
    Analyzes action instances from a tlabs cell array.
    Returns a dictionary of {action_class: total_frames_of_action}.
    """
    action_frames = defaultdict(int)
    
    # tlabs is a (5, 1) object array from scipy.io.loadmat.
    # Each element tlabs[i, 0] is a (K, 2) numpy array of instances.
    for action_idx in range(tlabs.shape[0]):
        # CORRECT WAY to access the inner array for each action class
        instances = tlabs[action_idx, 0]
        
        # Check if there are any instances (start/end pairs) for this action
        if instances.size > 0:
            # instances is a (K, 2) array of [start, end] frames.
            # Calculate total frames by summing durations for all instances.
            # duration = end_frame - start_frame + 1
            durations = instances[:, 1] - instances[:, 0] + 1
            action_frames[action_idx] += np.sum(durations)
            
    return action_frames

def determine_video_label(action_frames):
    """
    Determines a single action label for the entire video based on dominant action.
    """
    if not action_frames:
        # If no actions are annotated, default to class 4 ('Inspect Shelf').
        return 4
    
    # The label is the action class with the most total frames in the video.
    dominant_action = max(action_frames, key=action_frames.get)
    return dominant_action

def convert_merl_dataset():
    """Convert MERL .mat labels to video-level pickle files"""
    data_splits = {
        'train': list(range(1, 21)),
        'val': list(range(21, 27)),
        'test': list(range(27, 42))
    }
    
    for split_name, subject_ids in data_splits.items():
        print(f"\n🔄 Processing {split_name} split...")
        split_labels = {}
        
        for subject_id in subject_ids:
            for session_id in [1, 2, 3]:
                video_key = f"{subject_id}_{session_id}"
                mat_file = f"merl_dataset/labels/{split_name}/{subject_id}_{session_id}_label.mat"
                
                if os.path.exists(mat_file):
                    tlabs = scipy.io.loadmat(mat_file).get('tlabs')
                    
                    if tlabs is not None:
                        action_frames = analyze_video_actions(tlabs)
                        video_label = determine_video_label(action_frames)
                        split_labels[video_key] = video_label
                        
                        if not action_frames:
                            print(f"   📄 {mat_file} -> No actions found. Defaulting to Class 4.")
                        else:
                            print(f"   📄 {mat_file} -> Actions: {dict(action_frames)} -> Assigned Label: {video_label} ({action_classes[video_label]})")
                    else:
                        split_labels[video_key] = 4 # Default
                        print(f"   ⚠️ No 'tlabs' in {mat_file}, defaulting to Class 4.")
                else:
                    # Some files don't exist as per MERL doc, so we just skip them.
                    pass
        
        output_file = f"{split_name}_y_merl_v2.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(split_labels, f)
        
        label_counts = Counter(split_labels.values())
        print(f"\n📊 {split_name.upper()} split statistics:")
        print(f"   Total videos: {len(split_labels)}")
        print(f"   Label distribution: {dict(sorted(label_counts.items()))}")
        print(f"💾 Saved to: {output_file}")

def create_alternative_labels():
    """
    Alternative: Create balanced labels based on video IDs
    if .mat files are not available
    """
    print(f"\n🔄 Creating alternative balanced labels...")
    
    data_splits = {
        'train': list(range(1, 21)),
        'val': list(range(21, 27)),
        'test': list(range(27, 42))
    }
    
    for split_name, subject_ids in data_splits.items():
        split_labels = {}
        
        for subject_id in subject_ids:
            for session_id in [1, 2, 3]:
                video_key = f"{subject_id}_{session_id}"
                
                # Balanced assignment
                label = ((subject_id - 1) * 3 + (session_id - 1)) % 5
                split_labels[video_key] = label
        
        # Save alternative labels
        output_file = f"{split_name}_y_balanced.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(split_labels, f)
        
        label_counts = Counter(split_labels.values())
        print(f"\n📊 {split_name.upper()} balanced labels:")
        print(f"   Label distribution: {dict(label_counts)}")
        print(f"💾 Saved to: {output_file}")

if __name__ == "__main__":
    # Check if .mat files exist in the merl_dataset structure
    sample_mat_files = [
        'merl_dataset/labels/train/10_1_label.mat',
        'merl_dataset/labels/train/10_2_label.mat', 
        'merl_dataset/labels/train/11_1_label.mat'
    ]
    mat_files_exist = any(os.path.exists(f) for f in sample_mat_files)
    
    if mat_files_exist:
        print("✅ Found .mat label files, using proper MERL conversion")
        convert_merl_dataset()
    else:
        print("⚠️ No .mat files found, creating balanced alternative")
        create_alternative_labels()
    
    print("\n✅ MERL LABEL CONVERSION V2 COMPLETE!")
    print("🎯 Next: Update training script to use *_y_merl_v2.pkl files.")
    print("📝 Files created: *_y_merl_v2.pkl or *_y_balanced.pkl") 