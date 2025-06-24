#!/usr/bin/env python3
"""
Test script to verify HuggingFace dataset structure is detected correctly
"""

import os
import glob

def test_dataset_structure():
    print("🧪 Testing HuggingFace Dataset Structure Detection")
    print("=" * 60)
    
    # Check basic paths
    base_path = "./MERL_Shopping_Dataset"
    videos_path = os.path.join(base_path, "videos")
    labels_path = os.path.join(base_path, "labels")
    
    print(f"📁 Base path: {base_path}")
    print(f"   Exists: {os.path.exists(base_path)}")
    
    if not os.path.exists(base_path):
        print("❌ Dataset not found! Please download first:")
        print("   Run the notebook CELL 2 to download from HuggingFace")
        return False
    
    print(f"\n📁 Videos path: {videos_path}")
    print(f"   Exists: {os.path.exists(videos_path)}")
    
    print(f"\n📁 Labels path: {labels_path}")
    print(f"   Exists: {os.path.exists(labels_path)}")
    
    # Check subfolders
    print(f"\n📂 Checking subfolders...")
    
    total_videos = 0
    total_labels = 0
    
    for path_name, path in [("Videos", videos_path), ("Labels", labels_path)]:
        if os.path.exists(path):
            subfolders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            print(f"   {path_name}: {subfolders}")
            
            # Count files in each subfolder
            for subfolder in subfolders:
                subfolder_path = os.path.join(path, subfolder)
                if path_name == "Videos":
                    files = glob.glob(os.path.join(subfolder_path, "*.mp4"))
                    total_videos += len(files)
                    print(f"      {subfolder}/: {len(files)} .mp4 files")
                else:
                    files = glob.glob(os.path.join(subfolder_path, "*.mat"))
                    total_labels += len(files)
                    print(f"      {subfolder}/: {len(files)} .mat files")
        else:
            print(f"   {path_name}: Not found")
    
    print(f"\n📊 Total files found:")
    print(f"   🎬 Videos: {total_videos} .mp4 files")
    print(f"   🏷️ Labels: {total_labels} .mat files")
    
    if total_videos > 0 and total_labels > 0:
        print(f"\n✅ SUCCESS! Dataset structure detected correctly")
        print(f"   Ready to run: python utils/det2rec.py --start 1 --end 5")
        return True
    else:
        print(f"\n❌ PROBLEM: No files found in subfolders")
        print(f"   This might be a different dataset structure")
        return False

def test_sample_files():
    """Test if we can access some sample files"""
    print(f"\n🔍 Testing sample file access...")
    
    videos_path = "./MERL_Shopping_Dataset/videos"
    labels_path = "./MERL_Shopping_Dataset/labels"
    
    # Try to find some sample files
    sample_video = None
    sample_label = None
    
    for subfolder in ['train', 'val', 'test']:
        if not sample_video:
            video_subfolder = os.path.join(videos_path, subfolder)
            if os.path.exists(video_subfolder):
                videos = glob.glob(os.path.join(video_subfolder, "*.mp4"))
                if videos:
                    sample_video = videos[0]
        
        if not sample_label:
            label_subfolder = os.path.join(labels_path, subfolder)
            if os.path.exists(label_subfolder):
                labels = glob.glob(os.path.join(label_subfolder, "*.mat"))
                if labels:
                    sample_label = labels[0]
    
    if sample_video:
        print(f"   📹 Sample video: {os.path.basename(sample_video)}")
        print(f"      Full path: {sample_video}")
        print(f"      Size: {os.path.getsize(sample_video) / (1024*1024):.1f} MB")
    else:
        print(f"   ❌ No sample video found")
    
    if sample_label:
        print(f"   🏷️ Sample label: {os.path.basename(sample_label)}")
        print(f"      Full path: {sample_label}")
        
        # Try to load the .mat file
        try:
            from scipy.io import loadmat
            data = loadmat(sample_label)
            print(f"      Contents: {list(data.keys())}")
            if 'tlabs' in data:
                print(f"      ✅ 'tlabs' found in .mat file")
            else:
                print(f"      ⚠️ 'tlabs' not found in .mat file")
        except Exception as e:
            print(f"      ❌ Error loading .mat file: {e}")
    else:
        print(f"   ❌ No sample label found")

if __name__ == "__main__":
    success = test_dataset_structure()
    test_sample_files()
    
    print(f"\n🎯 SUMMARY:")
    if success:
        print(f"   ✅ Dataset structure is correct")
        print(f"   ✅ Ready to generate clips")
        print(f"\n📋 Next steps:")
        print(f"   1. Run: python utils/det2rec.py --start 1 --end 5")
        print(f"   2. Verify: python utils/test.py")
        print(f"   3. Train: python train_merl_clips.py")
    else:
        print(f"   ❌ Dataset structure issues detected")
        print(f"   💡 Make sure you downloaded the dataset correctly")
        print(f"   📋 Try running notebook CELL 2 again") 