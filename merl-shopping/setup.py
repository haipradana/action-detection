#!/usr/bin/env python3
"""
MERL Shopping Setup Script
Quick setup and testing for the consolidated merl-shopping approach.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'tensorflow', 'opencv-python', 'scipy', 'pandas', 
        'numpy', 'matplotlib', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'tensorflow':
                import tensorflow as tf
            elif package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def check_dataset_paths():
    """Check if MERL dataset paths exist"""
    print("\n📁 Checking dataset paths...")
    
    labels_path = "./MERL_Shopping_Dataset/labels"
    videos_path = "./MERL_Shopping_Dataset/videos"
    
    paths_exist = True
    
    if os.path.exists(labels_path):
        label_files = [f for f in os.listdir(labels_path) if f.endswith('.mat')]
        print(f"   ✅ Labels: {labels_path} ({len(label_files)} .mat files)")
    else:
        print(f"   ❌ Labels: {labels_path} not found")
        paths_exist = False
    
    if os.path.exists(videos_path):
        video_files = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
        print(f"   ✅ Videos: {videos_path} ({len(video_files)} .mp4 files)")
    else:
        print(f"   ❌ Videos: {videos_path} not found")
        paths_exist = False
    
    if not paths_exist:
        print("\n💡 Dataset setup instructions:")
        print("   1. Download MERL Shopping Dataset from:")
        print("      https://www.merl.com/demos/merl-shopping-dataset")
        print("   2. Extract Labels_MERL_Shopping_Dataset.zip to ../../")
        print("   3. Extract Videos_MERL_Shopping_Dataset.zip to ../../")
        return False
    
    return True

def setup_folders():
    """Create necessary folders"""
    print("\n📂 Setting up folders...")
    
    folders = ['clips', 'dataframes', 'flow_clips']
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"   ✅ Created: {folder}/")
        else:
            print(f"   ✅ Exists: {folder}/")

def test_data_generator():
    """Test the data generator"""
    print("\n🧪 Testing data generator...")
    
    try:
        from merl_clips_data_gen import MerlClipsDataGenerator
        
        # Try to create a data generator
        gen = MerlClipsDataGenerator(
            clips_base_path="clips",
            dataframes_path="dataframes",
            seq_len=10,
            batch_size=1
        )
        
        if len(gen.clip_data) > 0:
            print(f"   ✅ Data generator works! Found {len(gen.clip_data)} clips")
            return True
        else:
            print("   ⚠️ No clips found. Need to generate clips first.")
            return False
            
    except Exception as e:
        print(f"   ❌ Data generator test failed: {e}")
        return False

def generate_sample_clips():
    """Generate sample clips for testing"""
    print("\n🎬 Generating sample clips...")
    
    try:
        # Run det2rec.py for first 3 videos
        result = subprocess.run([
            sys.executable, 'utils/det2rec.py', 
            '--start', '1', '--end', '3'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Sample clips generated successfully!")
            print("   📊 Run 'python utils/test.py' to verify")
            return True
        else:
            print(f"   ❌ Clip generation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error running det2rec.py: {e}")
        return False

def run_quick_test():
    """Run a quick training test"""
    print("\n🚀 Running quick training test...")
    
    try:
        # Test import of training script
        sys.path.append('.')
        from train_merl_clips import CONFIG
        
        print("   ✅ Training script imports successfully!")
        print(f"   📋 Config: {CONFIG['seq_len']} seq_len, {CONFIG['batch_size']} batch_size")
        
        # Test if we can create a model
        try:
            sys.path.append('..')
            from model import MyCL_Model
            model = MyCL_Model(num_classes=5)
            print("   ✅ Model creation successful!")
            return True
        except ImportError:
            print("   ⚠️ Parent model.py not found, will use fallback model")
            return True
            
    except Exception as e:
        print(f"   ❌ Training test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🎯 MERL SHOPPING SETUP WIZARD")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Setup failed: Missing dependencies")
        return
    
    # Step 2: Check dataset paths
    if not check_dataset_paths():
        print("\n❌ Setup failed: Dataset not found")
        return
    
    # Step 3: Setup folders
    setup_folders()
    
    # Step 4: Test data generator (might fail if no clips yet)
    data_gen_works = test_data_generator()
    
    # Step 5: Generate sample clips if needed
    if not data_gen_works:
        print("\n💡 No clips found. Generating sample clips...")
        if generate_sample_clips():
            # Test again after generating clips
            test_data_generator()
    
    # Step 6: Test training setup
    run_quick_test()
    
    print("\n🎉 SETUP COMPLETE!")
    print("\n📋 Next steps:")
    
    if os.path.exists("dataframes") and os.listdir("dataframes"):
        print("   ✅ Clips generated! Ready to train:")
        print("      python train_merl_clips.py")
    else:
        print("   📝 Generate clips first:")
        print("      python utils/det2rec.py --start 1 --end 10")
        print("   📊 Verify data:")
        print("      python utils/test.py")
        print("   🚀 Train model:")
        print("      python train_merl_clips.py")
    
    print("\n📚 For help, see README.md")

if __name__ == "__main__":
    main() 