import os
import glob
import argparse
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
print('\n')


parser = argparse.ArgumentParser(description='Detection to recognition for MERL Shopping.')
parser.add_argument('--start', default=1, type=int, help='start video')
parser.add_argument('--end', default=1, type=int, help='end video')
parser.add_argument('--labels_path', default='./MERL_Shopping_Dataset/labels', type=str, help='path to labels folder')
parser.add_argument('--videos_path', default='./MERL_Shopping_Dataset/videos', type=str, help='path to videos folder')


def data_info(labels_path, videos_path):
  print(f'Working directory: {os.getcwd()}')
  print('\n')
  
  print('Content:')
  for element in os.listdir():
      if not element.startswith('.'):
          print(element)
  print('\n')
  
  # Updated paths to work with HuggingFace structure (with train/val/test subfolders)
  actions = []
  videos = []
  
  # Search in all subfolders (train, val, test)
  for subfolder in ['train', 'val', 'test']:
    label_subfolder = os.path.join(labels_path, subfolder)
    video_subfolder = os.path.join(videos_path, subfolder)
    
    if os.path.exists(label_subfolder):
      actions.extend(glob.glob(os.path.join(label_subfolder, '*.mat')))
    
    if os.path.exists(video_subfolder):
      videos.extend(glob.glob(os.path.join(video_subfolder, '*.mp4')))
  
  # Also check if files are directly in the main folders (fallback)
  if not actions:
    actions = glob.glob(os.path.join(labels_path, '*.mat'))
  if not videos:
    videos = glob.glob(os.path.join(videos_path, '*.mp4'))

  if not actions:
    print(f"⚠️ No .mat files found in {labels_path}")
    print("Checked subfolders: train/, val/, test/ and main folder")
    print("Make sure you have the MERL labels dataset extracted")
    return [], []
    
  if not videos:
    print(f"⚠️ No .mp4 files found in {videos_path}")
    print("Checked subfolders: train/, val/, test/ and main folder")
    print("Make sure you have the MERL videos dataset extracted")
    return [], []
  
  class_distribution = [0]*5
  for action in actions:
    action_data = loadmat(action)
    for index, value in enumerate(action_data['tlabs']):
      class_distribution[index] += len(value[0])
  
  print(f'Class distribution: {class_distribution}')
  print(f'Total action instances: {sum(class_distribution)}')
  print('\n')
  return videos, actions


def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]


def cv2npy(video):
  id_str = os.path.basename(video)
  print(f"Processing: {id_str}")
  
  cap = cv2.VideoCapture(video)
  if not cap.isOpened():
    print(f"❌ Error: Could not open video {video}")
    return None, None, None
    
  print('Stream start.')
  
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  t = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
  clip_mat = []
  flow_clip_mat = []
  
  old_ret, old_frame = cap.read()
  if not old_ret:
          print('Stream end.\n')
          cap.release()
          return None, None, None
          
  old_frame = cv2.resize(old_frame, (224,224))
  old_gray_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
  old_frame = old_frame[:, :, [2, 1, 0]]  # BGR to RGB
  clip_mat.append(old_frame)
  
  frame_count = 1
  while cap.isOpened():
      
      ret, frame = cap.read()
      # If frame is read correctly ret is True.
      if not ret:
          print('Stream end.\n')
          break    
      
      # Fill in video array.
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, (224,224))
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
      clip_mat.append(frame)
      
      # Calculate optical flow.
      retval = cv2.FarnebackOpticalFlow_create(5,0.5,False,15,3,5,1.2,0)       
      flow = retval.calc(old_gray_frame, gray_frame, None)
      
      # Fill in flow array.
      flow_clip_mat.append(flow)
      
      old_gray_frame = gray_frame
      frame_count += 1
      
      # Progress indicator
      if frame_count % 500 == 0:
        print(f"   Processed {frame_count} frames...")
  
  cap.release()
  
  clip_mat = np.array(clip_mat) / 255.0
  flow_clip_mat = np.array(flow_clip_mat) / 255.0
  
  print(f'Frame height: {h}')
  print(f'Frame width: {w}')
  print(f'Frame count: {t} (processed: {len(clip_mat)})')
  print('\n')
  return id_str, clip_mat[1:], flow_clip_mat


def main():
  global args
  args = parser.parse_args()

  print("🎯 MERL SHOPPING DETECTION TO RECOGNITION CONVERTER")
  print("=" * 60)
  
  # Check if paths exist
  if not os.path.exists(args.labels_path):
    print(f"❌ Labels path not found: {args.labels_path}")
    print("Please extract the MERL Labels dataset or specify correct path with --labels_path")
    return
    
  if not os.path.exists(args.videos_path):
    print(f"❌ Videos path not found: {args.videos_path}")
    print("Please extract the MERL Videos dataset or specify correct path with --videos_path")
    return

  videos, actions = data_info(args.labels_path, args.videos_path)
  
  if not videos or not actions:
    print("❌ No data found. Please check your dataset paths.")
    return

  start_video = args.start
  end_video = args.end
  
  print(f"🔄 Processing videos {start_video} to {end_video}")
  print(f"📊 Found {len(videos)} videos and {len(actions)} label files")

  for i in range(start_video, end_video+1):
    if i > len(videos):
      print(f"⚠️ Video {i} not found, skipping...")
      continue
      
    df = pd.DataFrame(columns=['name', 'class'])

    j = 1
    video = videos[i-1]
    
    # Create output directories
    clips_dir = f'clips/video_{i}'
    flow_clips_dir = f'flow_clips/video_{i}'
    
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(flow_clips_dir, exist_ok=True)

    print(f'🎬 Processing video_{i}')

    result = cv2npy(video)
    if result[0] is None:
      print(f"❌ Failed to process video {i}, skipping...")
      continue
      
    id_str, clip_mat, flow_clip_mat = result
    print(f"Video ID: {id_str}")
    id_str = id_str[:-9]  # Remove '_crop.mp4' extension for id.

    # Find corresponding action file
    action_files = [a for a in actions if id_str in os.path.basename(a)]
    if not action_files:
      print(f"⚠️ No label file found for {id_str}, skipping...")
      continue
      
    action_file = action_files[0]
    action_data = loadmat(action_file)

    total_clips_saved = 0
    for index, value in enumerate(action_data['tlabs']):
      #print(value[0])
      for start, stop in value[0]:
        if start < len(clip_mat) and stop <= len(clip_mat) and start < stop:
          clip_data = clip_mat[start:stop, ...]
          flow_data = flow_clip_mat[start:stop, ...]
          
          if len(clip_data) > 0:  # Only save non-empty clips
            np.save(f'{clips_dir}/clip_{j}', clip_data)
            print(f"   Saved clip_{j}: {clip_data.shape} (frames {start}-{stop}, class {index+1})")
            
            np.save(f'{flow_clips_dir}/flow_clip_{j}', flow_data)
            
            df = pd.concat([df, pd.DataFrame([{'name' : f'clip_{j}' , 'class' : index+1}])], ignore_index=True) 
            j += 1
            total_clips_saved += 1
        else:
          print(f"   ⚠️ Invalid clip bounds: {start}-{stop} (video length: {len(clip_mat)})")

    df.to_csv(f'dataframes/dataframe_{i}.csv', index=False)
    print(f"✅ Video {i} complete: {total_clips_saved} clips saved")
    print(f"📄 Dataframe saved: dataframes/dataframe_{i}.csv\n")

  print("🎉 CONVERSION COMPLETE!")
  print(f"\n📁 Generated files:")
  print(f"   - Clips: clips/video_*/clip_*.npy")
  print(f"   - Flow clips: flow_clips/video_*/flow_clip_*.npy") 
  print(f"   - Dataframes: dataframes/dataframe_*.csv")
  print(f"\n🎯 Next step: Run train script in parent directory")

if __name__ == '__main__':
  main() 