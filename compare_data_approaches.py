#!/usr/bin/env python3
"""
Data Approach Comparison
Compares the class distribution between:
1. convert_mat_pkl.py (video-level, imbalanced)
2. merl-shopping approach (clip-level, balanced)
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import glob
from collections import Counter
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_convert_mat_pkl_distribution():
    """Analyze class distribution from convert_mat_pkl.py approach"""
    print("🔍 Analyzing convert_mat_pkl.py approach...")
    
    # Check if pickle files exist
    pickle_files = ['train_y_merl_v2.pkl', 'val_y_merl_v2.pkl', 'test_y_merl_v2.pkl']
    
    total_distribution = {}
    splits_data = {}
    
    for pkl_file in pickle_files:
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            split_name = pkl_file.replace('_y_merl_v2.pkl', '')
            labels = list(data.values())
            distribution = Counter(labels)
            
            splits_data[split_name] = {
                'total_videos': len(labels),
                'distribution': dict(distribution)
            }
            
            # Add to total
            for class_id, count in distribution.items():
                total_distribution[class_id] = total_distribution.get(class_id, 0) + count
        else:
            print(f"⚠️ File not found: {pkl_file}")
    
    return splits_data, total_distribution

def analyze_merl_shopping_distribution():
    """Analyze class distribution from merl-shopping approach"""
    print("🔍 Analyzing merl-shopping approach...")
    
    clips_path = '../merl-shopping/clips'
    dataframes_path = '../merl-shopping/dataframes'
    
    if not os.path.exists(dataframes_path):
        print(f"❌ Dataframes path not found: {dataframes_path}")
        return {}, {}
    
    # Get all dataframe files
    dataframe_files = glob.glob(os.path.join(dataframes_path, "dataframe_*.csv"))
    
    total_distribution = {}
    all_clips = []
    
    for df_file in dataframe_files:
        video_num = int(os.path.basename(df_file).split('_')[1].split('.')[0])
        df = pd.read_csv(df_file)
        
        for _, row in df.iterrows():
            class_label = int(row['class']) - 1  # Convert to 0-indexed
            all_clips.append(class_label)
            total_distribution[class_label] = total_distribution.get(class_label, 0) + 1
    
    clips_data = {
        'total_clips': len(all_clips),
        'distribution': total_distribution
    }
    
    return clips_data, total_distribution

def create_comparison_plots(mat_pkl_data, merl_clips_data):
    """Create comparison plots"""
    print("📊 Creating comparison plots...")
    
    # Action class names
    action_classes = [
        'Reach To Shelf',
        'Retract From Shelf', 
        'Hand In Shelf',
        'Inspect Product',
        'Inspect Shelf'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MERL Dataset: convert_mat_pkl vs merl-shopping Approach', fontsize=16, fontweight='bold')
    
    # 1. Video-level distribution (convert_mat_pkl)
    if mat_pkl_data[1]:  # If we have data
        classes = list(range(5))
        counts_pkl = [mat_pkl_data[1].get(i, 0) for i in classes]
        
        axes[0,0].bar(classes, counts_pkl, alpha=0.7, color='skyblue', edgecolor='navy')
        axes[0,0].set_title('convert_mat_pkl.py\n(Video-level, Dominant Action)', fontweight='bold')
        axes[0,0].set_xlabel('Action Class')
        axes[0,0].set_ylabel('Number of Videos')
        axes[0,0].set_xticks(classes)
        axes[0,0].set_xticklabels([f'{i}' for i in classes])
        
        # Add value labels on bars
        for i, v in enumerate(counts_pkl):
            axes[0,0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Calculate and show imbalance
        if sum(counts_pkl) > 0:
            percentages = [(c/sum(counts_pkl))*100 for c in counts_pkl]
            max_diff = max(percentages) - min([p for p in percentages if p > 0])
            axes[0,0].text(0.02, 0.98, f'Imbalance: {max_diff:.1f}% diff', 
                          transform=axes[0,0].transAxes, va='top', 
                          bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    else:
        axes[0,0].text(0.5, 0.5, 'No convert_mat_pkl data\navailable', 
                      ha='center', va='center', transform=axes[0,0].transAxes,
                      fontsize=12, style='italic')
        axes[0,0].set_title('convert_mat_pkl.py\n(No data)', fontweight='bold')
    
    # 2. Clip-level distribution (merl-shopping)
    if merl_clips_data[1]:  # If we have data
        classes = list(range(5))
        counts_clips = [merl_clips_data[1].get(i, 0) for i in classes]
        
        axes[0,1].bar(classes, counts_clips, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        axes[0,1].set_title('merl-shopping approach\n(Clip-level, All Actions)', fontweight='bold')
        axes[0,1].set_xlabel('Action Class')
        axes[0,1].set_ylabel('Number of Clips')
        axes[0,1].set_xticks(classes)
        axes[0,1].set_xticklabels([f'{i}' for i in classes])
        
        # Add value labels on bars
        for i, v in enumerate(counts_clips):
            axes[0,1].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Calculate and show balance
        if sum(counts_clips) > 0:
            percentages = [(c/sum(counts_clips))*100 for c in counts_clips]
            max_diff = max(percentages) - min([p for p in percentages if p > 0])
            axes[0,1].text(0.02, 0.98, f'Imbalance: {max_diff:.1f}% diff', 
                          transform=axes[0,1].transAxes, va='top',
                          bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    else:
        axes[0,1].text(0.5, 0.5, 'No merl-shopping data\navailable', 
                      ha='center', va='center', transform=axes[0,1].transAxes,
                      fontsize=12, style='italic')
        axes[0,1].set_title('merl-shopping approach\n(No data)', fontweight='bold')
    
    # 3. Percentage comparison
    if mat_pkl_data[1] and merl_clips_data[1]:
        pkl_total = sum(mat_pkl_data[1].values())
        clips_total = sum(merl_clips_data[1].values())
        
        pkl_percentages = [(mat_pkl_data[1].get(i, 0)/pkl_total)*100 for i in range(5)]
        clips_percentages = [(merl_clips_data[1].get(i, 0)/clips_total)*100 for i in range(5)]
        
        x = np.arange(5)
        width = 0.35
        
        axes[1,0].bar(x - width/2, pkl_percentages, width, label='convert_mat_pkl', 
                     alpha=0.7, color='skyblue', edgecolor='navy')
        axes[1,0].bar(x + width/2, clips_percentages, width, label='merl-shopping', 
                     alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        
        axes[1,0].set_title('Percentage Distribution Comparison', fontweight='bold')
        axes[1,0].set_xlabel('Action Class')
        axes[1,0].set_ylabel('Percentage (%)')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels([f'{i}' for i in range(5)])
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. Data volume comparison
    data_comparison = {
        'Approach': ['convert_mat_pkl\n(Videos)', 'merl-shopping\n(Clips)'],
        'Samples': [
            sum(mat_pkl_data[1].values()) if mat_pkl_data[1] else 0,
            sum(merl_clips_data[1].values()) if merl_clips_data[1] else 0
        ],
        'Colors': ['skyblue', 'lightgreen']
    }
    
    bars = axes[1,1].bar(data_comparison['Approach'], data_comparison['Samples'], 
                        color=data_comparison['Colors'], alpha=0.7,
                        edgecolor=['navy', 'darkgreen'])
    axes[1,1].set_title('Total Training Samples', fontweight='bold')
    axes[1,1].set_ylabel('Number of Samples')
    
    # Add value labels
    for bar, value in zip(bars, data_comparison['Samples']):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                      f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('merl_data_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Comparison plot saved as 'merl_data_comparison.png'")
    
    return fig

def print_detailed_analysis(mat_pkl_data, merl_clips_data):
    """Print detailed analysis"""
    print("\n" + "="*60)
    print("📊 DETAILED DATA ANALYSIS")
    print("="*60)
    
    action_classes = [
        'Reach To Shelf',
        'Retract From Shelf', 
        'Hand In Shelf',
        'Inspect Product',
        'Inspect Shelf'
    ]
    
    print("\n🎯 APPROACH 1: convert_mat_pkl.py (Video-level)")
    print("-" * 40)
    if mat_pkl_data[1]:
        total_videos = sum(mat_pkl_data[1].values())
        print(f"Total videos: {total_videos}")
        print("Class distribution:")
        for i in range(5):
            count = mat_pkl_data[1].get(i, 0)
            percentage = (count/total_videos)*100 if total_videos > 0 else 0
            print(f"  {i}: {action_classes[i]:<20} = {count:3d} videos ({percentage:5.1f}%)")
        
        # Calculate imbalance metrics
        counts = [mat_pkl_data[1].get(i, 0) for i in range(5)]
        if total_videos > 0:
            percentages = [(c/total_videos)*100 for c in counts]
            std_dev = np.std(percentages)
            cv = std_dev / np.mean(percentages) if np.mean(percentages) > 0 else 0
            print(f"\n📉 Imbalance metrics:")
            print(f"   Standard deviation: {std_dev:.2f}%")
            print(f"   Coefficient of variation: {cv:.3f}")
    else:
        print("❌ No data available")
    
    print("\n🎯 APPROACH 2: merl-shopping (Clip-level)")
    print("-" * 40)
    if merl_clips_data[1]:
        total_clips = sum(merl_clips_data[1].values())
        print(f"Total clips: {total_clips}")
        print("Class distribution:")
        for i in range(5):
            count = merl_clips_data[1].get(i, 0)
            percentage = (count/total_clips)*100 if total_clips > 0 else 0
            print(f"  {i}: {action_classes[i]:<20} = {count:4d} clips ({percentage:5.1f}%)")
        
        # Calculate balance metrics
        counts = [merl_clips_data[1].get(i, 0) for i in range(5)]
        if total_clips > 0:
            percentages = [(c/total_clips)*100 for c in counts]
            std_dev = np.std(percentages)
            cv = std_dev / np.mean(percentages) if np.mean(percentages) > 0 else 0
            print(f"\n📈 Balance metrics:")
            print(f"   Standard deviation: {std_dev:.2f}%")
            print(f"   Coefficient of variation: {cv:.3f}")
    else:
        print("❌ No data available")
    
    # Comparison summary
    print("\n🎯 COMPARISON SUMMARY")
    print("-" * 40)
    
    if mat_pkl_data[1] and merl_clips_data[1]:
        pkl_total = sum(mat_pkl_data[1].values())
        clips_total = sum(merl_clips_data[1].values())
        data_increase = (clips_total / pkl_total) if pkl_total > 0 else 0
        
        print(f"📊 Data volume increase: {data_increase:.1f}x more samples")
        print(f"📈 Training samples: {pkl_total} → {clips_total}")
        
        # Calculate balance improvement
        pkl_counts = [mat_pkl_data[1].get(i, 0) for i in range(5)]
        clips_counts = [merl_clips_data[1].get(i, 0) for i in range(5)]
        
        if pkl_total > 0 and clips_total > 0:
            pkl_percentages = [(c/pkl_total)*100 for c in pkl_counts]
            clips_percentages = [(c/clips_total)*100 for c in clips_counts]
            
            pkl_std = np.std(pkl_percentages)
            clips_std = np.std(clips_percentages)
            
            balance_improvement = ((pkl_std - clips_std) / pkl_std) * 100 if pkl_std > 0 else 0
            
            print(f"⚖️ Balance improvement: {balance_improvement:.1f}%")
            print(f"   convert_mat_pkl std: {pkl_std:.2f}%")
            print(f"   merl-shopping std: {clips_std:.2f}%")
    
    print(f"\n💡 RECOMMENDATION:")
    if merl_clips_data[1] and sum(merl_clips_data[1].values()) > 0:
        print(f"✅ Use merl-shopping approach for:")
        print(f"   - {data_increase:.1f}x more training data")
        print(f"   - Much better class balance")
        print(f"   - More precise temporal boundaries")
        print(f"   - Better generalization potential")
    else:
        print(f"⚠️ merl-shopping data not available")
        print(f"   Run det2rec.py in merl-shopping folder first")

def main():
    print("🎯 MERL DATASET APPROACH COMPARISON")
    print("=" * 60)
    
    # Analyze both approaches
    mat_pkl_splits, mat_pkl_total = analyze_convert_mat_pkl_distribution()
    merl_clips_data, merl_clips_total = analyze_merl_shopping_distribution()
    
    # Create visualization
    fig = create_comparison_plots((mat_pkl_splits, mat_pkl_total), 
                                 (merl_clips_data, merl_clips_total))
    
    # Print detailed analysis
    print_detailed_analysis((mat_pkl_splits, mat_pkl_total), 
                          (merl_clips_data, merl_clips_total))

if __name__ == "__main__":
    main() 