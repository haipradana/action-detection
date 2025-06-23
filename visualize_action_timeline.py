#!/usr/bin/env python3
"""
Action Timeline Visualizer
Creates visual plots and analysis of detected actions
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import pandas as pd

# ===================================================================
# CONFIGURATION
# ===================================================================

JSON_FILE = "./mauk_ngetes/action_timeline.json"  # Output from multi_action_detection.py
SAVE_PLOTS = True
PLOT_DIR = "./mauk_ngetes/timeline_plots"

ACTION_CLASSES = [
    'Reach To Shelf',
    'Retract From Shelf', 
    'Hand In Shelf',
    'Inspect Product',
    'Inspect Shelf',
    'None of the above'
]

ACTION_COLORS = {
    'Reach To Shelf': '#FF6B6B',        # Red
    'Retract From Shelf': '#4ECDC4',    # Teal
    'Hand In Shelf': '#45B7D1',         # Blue
    'Inspect Product': '#FFA07A',       # Orange
    'Inspect Shelf': '#98D8C8',         # Mint
    'None of the above': '#D3D3D3'      # Gray
}

# ===================================================================
# VISUALIZATION FUNCTIONS
# ===================================================================

def load_timeline_data(json_file):
    """Load timeline data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def create_timeline_plot(segments, save_path=None):
    """Create timeline visualization"""
    if not segments:
        print("⚠️ No segments to plot")
        return
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot segments as horizontal bars
    y_pos = 0
    for i, segment in enumerate(segments):
        start_time = segment['start_time']
        duration = segment['duration']
        action_name = segment['action_name']
        confidence = segment['avg_confidence']
        
        color = ACTION_COLORS.get(action_name, '#D3D3D3')
        
        # Create bar
        bar = ax.barh(y_pos, duration, left=start_time, height=0.8, 
                     color=color, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add text label
        label_x = start_time + duration / 2
        label_text = f"{action_name}\n{confidence:.1%}"
        ax.text(label_x, y_pos, label_text, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        
        y_pos += 1
    
    # Formatting
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Action Segments', fontsize=12)
    ax.set_title('Action Recognition Timeline', fontsize=16, fontweight='bold')
    ax.set_ylim(-0.5, len(segments) - 0.5)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Format x-axis to show minutes:seconds
    max_time = max(s['end_time'] for s in segments)
    x_ticks = np.arange(0, max_time + 30, 30)  # Every 30 seconds
    x_labels = [f"{int(t//60):02d}:{int(t%60):02d}" for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    
    plt.tight_layout()
    
    if save_path and SAVE_PLOTS:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Timeline plot saved: {save_path}")
    
    plt.show()

def create_action_distribution_plot(segments, save_path=None):
    """Create action distribution pie chart"""
    if not segments:
        return
    
    # Count actions
    action_counts = {}
    total_duration = {}
    
    for segment in segments:
        action = segment['action_name']
        duration = segment['duration']
        
        action_counts[action] = action_counts.get(action, 0) + 1
        total_duration[action] = total_duration.get(action, 0) + duration
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count pie chart
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    colors = [ACTION_COLORS.get(action, '#D3D3D3') for action in actions]
    
    wedges1, texts1, autotexts1 = ax1.pie(counts, labels=actions, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
    ax1.set_title('Action Count Distribution', fontsize=14, fontweight='bold')
    
    # Duration pie chart
    durations = list(total_duration.values())
    wedges2, texts2, autotexts2 = ax2.pie(durations, labels=actions, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
    ax2.set_title('Action Duration Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path and SAVE_PLOTS:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Distribution plot saved: {save_path}")
    
    plt.show()

def create_confidence_analysis(segments, save_path=None):
    """Create confidence analysis plots"""
    if not segments:
        return
    
    # Prepare data
    actions = [s['action_name'] for s in segments]
    confidences = [s['avg_confidence'] for s in segments]
    durations = [s['duration'] for s in segments]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confidence distribution by action
    df = pd.DataFrame({
        'Action': actions,
        'Confidence': confidences,
        'Duration': durations
    })
    
    sns.boxplot(data=df, x='Action', y='Confidence', ax=ax1)
    ax1.set_title('Confidence Distribution by Action')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Confidence vs Duration scatter
    for action in set(actions):
        action_data = df[df['Action'] == action]
        ax2.scatter(action_data['Duration'], action_data['Confidence'], 
                   label=action, color=ACTION_COLORS.get(action, '#D3D3D3'),
                   alpha=0.7, s=60)
    
    ax2.set_xlabel('Duration (seconds)')
    ax2.set_ylabel('Confidence')
    ax2.set_title('Confidence vs Duration')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Confidence histogram
    ax3.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Overall Confidence Distribution')
    ax3.axvline(np.mean(confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidences):.2f}')
    ax3.legend()
    
    # 4. Duration histogram
    ax4.hist(durations, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.set_xlabel('Duration (seconds)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Action Duration Distribution')
    ax4.axvline(np.mean(durations), color='red', linestyle='--',
                label=f'Mean: {np.mean(durations):.1f}s')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path and SAVE_PLOTS:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confidence analysis saved: {save_path}")
    
    plt.show()

def print_detailed_statistics(data):
    """Print detailed statistics"""
    segments = data['segments']
    
    print("\n📊 DETAILED STATISTICS")
    print("=" * 60)
    
    # Basic stats
    total_segments = len(segments)
    total_video_time = max(s['end_time'] for s in segments) if segments else 0
    total_action_time = sum(s['duration'] for s in segments)
    
    print(f"Total Segments: {total_segments}")
    print(f"Total Video Time: {total_video_time:.1f}s ({total_video_time/60:.1f} min)")
    print(f"Total Action Time: {total_action_time:.1f}s ({total_action_time/60:.1f} min)")
    print(f"Action Coverage: {(total_action_time/total_video_time)*100:.1f}%")
    
    # Action statistics
    print(f"\n🎭 ACTION BREAKDOWN:")
    action_stats = {}
    
    for segment in segments:
        action = segment['action_name']
        if action not in action_stats:
            action_stats[action] = {
                'count': 0,
                'total_duration': 0,
                'confidences': []
            }
        
        action_stats[action]['count'] += 1
        action_stats[action]['total_duration'] += segment['duration']
        action_stats[action]['confidences'].append(segment['avg_confidence'])
    
    for action, stats in action_stats.items():
        avg_duration = stats['total_duration'] / stats['count']
        avg_confidence = np.mean(stats['confidences'])
        
        print(f"  {action}:")
        print(f"    Count: {stats['count']} segments")
        print(f"    Total Duration: {stats['total_duration']:.1f}s")
        print(f"    Avg Duration: {avg_duration:.1f}s")
        print(f"    Avg Confidence: {avg_confidence:.1%}")
    
    # Transition analysis
    print(f"\n🔄 TRANSITION ANALYSIS:")
    transitions = []
    for i in range(len(segments) - 1):
        current = segments[i]['action_name']
        next_action = segments[i + 1]['action_name']
        if current != next_action:
            transitions.append((current, next_action))
    
    print(f"Total Transitions: {len(transitions)}")
    
    if transitions:
        transition_counts = {}
        for transition in transitions:
            key = f"{transition[0]} → {transition[1]}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        print("Most Common Transitions:")
        for transition, count in sorted(transition_counts.items(), 
                                      key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {transition}: {count} times")

def generate_summary_report(data, output_file="action_summary_report.txt"):
    """Generate text summary report"""
    segments = data['segments']
    
    with open(output_file, 'w') as f:
        f.write("ACTION RECOGNITION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Timeline
        f.write("ACTION TIMELINE:\n")
        f.write("-" * 30 + "\n")
        
        for i, segment in enumerate(segments):
            start_min = int(segment['start_time'] // 60)
            start_sec = int(segment['start_time'] % 60)
            end_min = int(segment['end_time'] // 60)
            end_sec = int(segment['end_time'] % 60)
            
            f.write(f"{i+1:2d}. {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d} "
                   f"({segment['duration']:.1f}s) | "
                   f"{segment['action_name']} "
                   f"(conf: {segment['avg_confidence']:.1%})\n")
        
        # Statistics
        if segments:
            total_time = max(s['end_time'] for s in segments)
            f.write(f"\nSUMMARY STATISTICS:\n")
            f.write(f"Total Segments: {len(segments)}\n")
            f.write(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min)\n")
            
            # Action counts
            action_counts = {}
            for segment in segments:
                action = segment['action_name']
                action_counts[action] = action_counts.get(action, 0) + 1
            
            f.write(f"\nACTION DISTRIBUTION:\n")
            for action, count in action_counts.items():
                f.write(f"  {action}: {count} segments\n")
    
    print(f"📝 Summary report saved: {output_file}")

# ===================================================================
# MAIN FUNCTION
# ===================================================================

def main():
    import os
    
    print("📊 Action Timeline Visualizer")
    print("=" * 40)
    
    # Check if JSON file exists
    if not os.path.exists(JSON_FILE):
        print(f"❌ JSON file not found: {JSON_FILE}")
        print("💡 Run multi_action_detection.py first to generate the timeline data")
        return
    
    # Load data
    print(f"📁 Loading data from: {JSON_FILE}")
    data = load_timeline_data(JSON_FILE)
    segments = data['segments']
    
    if not segments:
        print("❌ No action segments found in the data")
        return
    
    print(f"✅ Loaded {len(segments)} action segments")
    
    # Create output directory for plots
    if SAVE_PLOTS:
        os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    # 1. Timeline plot
    timeline_path = os.path.join(PLOT_DIR, "action_timeline.png") if SAVE_PLOTS else None
    create_timeline_plot(segments, timeline_path)
    
    # 2. Distribution plots
    dist_path = os.path.join(PLOT_DIR, "action_distribution.png") if SAVE_PLOTS else None
    create_action_distribution_plot(segments, dist_path)
    
    # 3. Confidence analysis
    conf_path = os.path.join(PLOT_DIR, "confidence_analysis.png") if SAVE_PLOTS else None
    create_confidence_analysis(segments, conf_path)
    
    # 4. Print statistics
    print_detailed_statistics(data)
    
    # 5. Generate report
    generate_summary_report(data)
    
    print(f"\n🎉 Visualization completed!")
    if SAVE_PLOTS:
        print(f"📁 Plots saved in: {PLOT_DIR}/")

if __name__ == "__main__":
    main() 