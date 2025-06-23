import os
import cv2
import pickle
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class DataGenerator(Sequence):
    
    def __init__(self, x_path, y_path=None, to_fit=True, seq_len=15, batch_size=2, **kwargs):
        super().__init__(**kwargs)
        self.x_path = x_path        
        self.batch_size = batch_size
        self.to_fit = to_fit
        self.seq_len = seq_len
        
        # Check if path exists
        if not os.path.exists(self.x_path):
            raise ValueError(f"Path {self.x_path} does not exist")
        
        self.list_X = [f for f in os.listdir(self.x_path) if os.path.isdir(os.path.join(self.x_path, f))]
        
        if len(self.list_X) == 0:
            raise ValueError(f"No directories found in {self.x_path}")
        
        if to_fit:
            if y_path is None:
                raise ValueError("y_path must be provided when to_fit=True")
            self.y_path = y_path
            self.dict_Y = self.get_y(y_path)
    
    def __len__(self):
        return len(self.list_X)
    
    def __getitem__(self, index):
        images_folder = self.list_X[index]
        images_folder_path = os.path.join(self.x_path, images_folder)
        
        if not os.path.exists(images_folder_path):
            raise ValueError(f"Folder {images_folder_path} does not exist")
        
        images_list = sorted([
            f for f in os.listdir(images_folder_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if len(images_list) == 0:
            raise ValueError(f"No image files found in {images_folder_path}")
        
        all_frames = []
        for img in images_list:
            img_path = os.path.join(images_folder_path, img)
            frame = cv2.imread(img_path)
            if frame is not None:
                # Center crop to square then resize (preserves important center content)
                h, w = frame.shape[:2]
                if h > w:
                    # Crop height to match width (make square)
                    start_h = (h - w) // 2
                    frame = frame[start_h:start_h + w, :]
                else:
                    # Crop width to match height (make square)
                    start_w = (w - h) // 2
                    frame = frame[:, start_w:start_w + h]
                
                # Now resize square image to 224x224
                frame = cv2.resize(frame, (224, 224))
                # Convert BGR to RGB and normalize to [0, 1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                all_frames.append(frame)
        
        if len(all_frames) == 0:
            raise ValueError(f"No valid images could be loaded from {images_folder_path}")
        
        all_frames = np.stack(all_frames)
        
        # Get targets if in training mode
        if self.to_fit:
            key = '_'.join(images_folder.split('_')[:2])
            if key not in self.dict_Y:
                # Try different key formats
                alt_key = images_folder
                if alt_key in self.dict_Y:
                    key = alt_key
                else:
                    raise KeyError(f"Key '{key}' not found in labels. Available keys: {list(self.dict_Y.keys())[:5]}...")
            
            Y = np.array(self.dict_Y[key])
            all_frames, targets = self.check_alignment(all_frames, Y)
            
            # Create time series generator
            series_data = TimeseriesGenerator(
                all_frames, targets, 
                length=self.seq_len, 
                batch_size=self.batch_size,
                sampling_rate=1,
                stride=1
            )
        else:
            # For inference mode
            series_data = TimeseriesGenerator(
                all_frames, all_frames,  # Use frames as both input and dummy target
                length=self.seq_len, 
                batch_size=self.batch_size,
                sampling_rate=1,
                stride=1
            )
        
        return series_data
    
    def get_y(self, path):
        """Load labels from pickle file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Label file {path} not found")
        
        try:
            with open(path, 'rb') as pickle_file:
                y_dict = pickle.load(pickle_file)
            return y_dict
        except Exception as e:
            raise ValueError(f"Error loading pickle file {path}: {e}")
    
    def check_alignment(self, all_frames, targets):
        """Align frames and targets to same length"""
        min_length = min(all_frames.shape[0], targets.shape[0])
        
        if min_length < self.seq_len:
            print(f"Warning: Sequence length {min_length} is shorter than required seq_len {self.seq_len}")
        
        # Trim to same length
        all_frames = all_frames[:min_length]
        targets = targets[:min_length]
        
        return all_frames, targets
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        # Shuffle indices if needed
        np.random.shuffle(self.list_X) 