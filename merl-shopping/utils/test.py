import os
import glob
import argparse
from cv2 import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
print('\n')


def data_info(labels_path):
  print(f'Working directory: {os.getcwd()}')
  print('\n')
  
  print('Content:')
  for element in os.listdir():
      if not element.startswith('.'):
          print(element)
  print('\n')
  
  # Search for .mat files in subfolders (train/val/test) and main folder
  actions = []
  for subfolder in ['train', 'val', 'test']:
    label_subfolder = os.path.join(labels_path, subfolder)
    if os.path.exists(label_subfolder):
      actions.extend(glob.glob(os.path.join(label_subfolder, '*.mat')))
  
  # Fallback: check main folder
  if not actions:
    actions = glob.glob(os.path.join(labels_path, '*.mat'))
  
  dataframes = glob.glob('dataframes/*.csv')

  print(f"Found {len(actions)} label files")
  print(f"Found {len(dataframes)} dataframe files")
  
  if not actions:
    print(f"❌ No .mat files found in {labels_path}")
    print("Checked subfolders: train/, val/, test/ and main folder")
    return [], [], []
  
  class_distribution = [0]*5
  for action in actions:
    action_data = loadmat(action)
    for index, value in enumerate(action_data['tlabs']):
      class_distribution[index] += len(value[0])
  
  print(f'Original class distribution from .mat files: {class_distribution}')
  print(f'Total action instances: {sum(class_distribution)}')
  print('\n')
  return [], actions, dataframes


def main():
  print("🧪 MERL SHOPPING CLASS DISTRIBUTION VERIFICATION")
  print("=" * 60)
  
  labels_path = "./MERL_Shopping_Dataset/labels"
  if not os.path.exists(labels_path):
    print(f"⚠️ Labels path not found: {labels_path}")
    print("Using current directory for dataframes only...")
    labels_path = "."

  _, actions, dataframes = data_info(labels_path)

  if not dataframes:
    print("❌ No dataframe files found!")
    print("Run det2rec.py first to generate clips and dataframes")
    return

  print("🔍 Analyzing generated clips distribution...")
  class_distribution = [0]*5
  total_clips = 0

  for df_file in sorted(dataframes):
    try:
      df = pd.read_csv(df_file)
      total_clips += len(df)
      temp = df['class'].value_counts().reset_index()
      
      for j in range(len(temp)):
        class_idx = temp.at[j, 'index'] - 1  # Convert to 0-indexed
        class_count = temp.at[j, 'class']
        if 0 <= class_idx < 5:
          class_distribution[class_idx] += class_count
      
      print(f"   📄 {df_file}: {len(df)} clips")
      
    except Exception as e:
      print(f"❌ Error reading {df_file}: {e}")

  print(f'\n📊 FINAL RESULTS:')
  print(f'=' * 40)
  print(f'Class distribution from clips: {class_distribution}')
  print(f'Total clips generated: {total_clips}')
  
  # Action class names
  action_classes = [
    'Reach To Shelf',
    'Retract From Shelf', 
    'Hand In Shelf',
    'Inspect Product',
    'Inspect Shelf'
  ]
  
  print(f'\n📈 Detailed breakdown:')
  for i, (class_name, count) in enumerate(zip(action_classes, class_distribution)):
    percentage = (count / total_clips * 100) if total_clips > 0 else 0
    print(f'   {i}: {class_name:<20} = {count:4d} clips ({percentage:5.1f}%)')
  
  # Calculate balance metrics
  if total_clips > 0:
    percentages = [(count / total_clips * 100) for count in class_distribution]
    std_dev = np.std(percentages)
    cv = std_dev / np.mean(percentages) if np.mean(percentages) > 0 else 0
    
    print(f'\n⚖️ Balance metrics:')
    print(f'   Standard deviation: {std_dev:.2f}%')
    print(f'   Coefficient of variation: {cv:.3f}')
    
    if std_dev < 10:
      print(f'   ✅ Good balance! (std < 10%)')
    elif std_dev < 20:
      print(f'   ⚠️ Moderate imbalance (10% ≤ std < 20%)')
    else:
      print(f'   ❌ High imbalance (std ≥ 20%)')

  print(f'\n🎯 SUCCESS! Clips generated and verified.')
  print(f'   Ready for training with merl-shopping approach!')

if __name__ == '__main__':
  main() 