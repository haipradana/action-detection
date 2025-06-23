# ===================================================================
# CELL 1: Imports and Setup
# ===================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import cv2
import numpy as np
import os
import pickle
import gc
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")
if torch.cuda.is_available():
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ===================================================================
# CELL 2: Data Paths and Configuration
# ===================================================================

# Paths configuration
videos_path = './Videos_MERL_Shopping_Dataset/'
x_train_path = videos_path + 'train/'
y_train_path = 'train_y.pkl'

x_val_path = videos_path + 'val/'
y_val_path = 'val_y.pkl'

x_test_path = videos_path + 'test/'
y_test_path = 'test_y.pkl'

# Model configuration
seq_len = 15
batch_size = 2  # Start with 2, can increase if memory allows
img_size = 224
num_classes = 6
num_epochs = 20
learning_rate = 0.001

# Action classes
action_classes = [
    'Reach To Shelf',
    'Retract From Shelf', 
    'Hand In Shelf',
    'Inspect Product',
    'Inspect Shelf',
    'None of the above'
]

print(f"📊 Configuration:")
print(f"   - Sequence length: {seq_len}")
print(f"   - Batch size: {batch_size}")
print(f"   - Image size: {img_size}x{img_size}")
print(f"   - Number of classes: {num_classes}")

# ===================================================================
# CELL 3: Dataset Class
# ===================================================================

class ActionDataset(Dataset):
    def __init__(self, x_path, y_path=None, seq_len=15, img_size=224, is_train=True):
        self.x_path = x_path
        self.seq_len = seq_len
        self.img_size = img_size
        self.is_train = is_train
        
        # Get list of video folders
        if not os.path.exists(self.x_path):
            raise ValueError(f"Path {self.x_path} does not exist")
        
        self.video_folders = [f for f in os.listdir(self.x_path) 
                             if os.path.isdir(os.path.join(self.x_path, f))]
        
        if len(self.video_folders) == 0:
            raise ValueError(f"No video folders found in {self.x_path}")
        
        # Load labels if provided
        if y_path and os.path.exists(y_path):
            with open(y_path, 'rb') as f:
                self.labels_dict = pickle.load(f)
        else:
            self.labels_dict = None
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        print(f"✅ Dataset initialized with {len(self.video_folders)} videos")
    
    def __len__(self):
        return len(self.video_folders)
    
    def load_video_sequence(self, folder_name):
        """Load and preprocess video sequence"""
        folder_path = os.path.join(self.x_path, folder_name)
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(folder_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {folder_path}")
        
        frames = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            
            # Load image
            frame = cv2.imread(img_path)
            if frame is not None:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply transforms
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
        
        if len(frames) == 0:
            raise ValueError(f"No valid frames loaded from {folder_path}")
        
        # Stack frames: (T, C, H, W)
        frames_tensor = torch.stack(frames)
        
        return frames_tensor, len(frames)
    
    def get_labels(self, folder_name):
        """Get labels for the video"""
        if self.labels_dict is None:
            return torch.zeros(self.seq_len, dtype=torch.long)
        
        # Try different key formats
        key = '_'.join(folder_name.split('_')[:2])
        if key not in self.labels_dict:
            key = folder_name
            if key not in self.labels_dict:
                print(f"⚠️ Warning: No labels found for {folder_name}")
                return torch.zeros(self.seq_len, dtype=torch.long)
        
        labels = torch.tensor(self.labels_dict[key], dtype=torch.long)
        return labels
    
    def __getitem__(self, idx):
        folder_name = self.video_folders[idx]
        
        try:
            # Load video sequence
            frames, total_frames = self.load_video_sequence(folder_name)
            
            # Load labels
            labels = self.get_labels(folder_name)
            
            # Ensure same length
            min_len = min(len(frames), len(labels))
            if min_len < self.seq_len:
                # Pad if too short
                if len(frames) < self.seq_len:
                    pad_size = self.seq_len - len(frames)
                    last_frame = frames[-1].unsqueeze(0)
                    padding = last_frame.repeat(pad_size, 1, 1, 1)
                    frames = torch.cat([frames, padding], dim=0)
                
                if len(labels) < self.seq_len:
                    pad_size = self.seq_len - len(labels)
                    padding = torch.full((pad_size,), labels[-1].item(), dtype=torch.long)
                    labels = torch.cat([labels, padding], dim=0)
            
            # Take first seq_len frames
            frames = frames[:self.seq_len]  # (seq_len, C, H, W)
            labels = labels[:self.seq_len]  # (seq_len,)
            
            # For action recognition, we usually predict one action per sequence
            # Take the most common label in the sequence
            action_label = torch.mode(labels)[0]
            
            return frames, action_label
            
        except Exception as e:
            print(f"Error loading {folder_name}: {e}")
            # Return dummy data
            dummy_frames = torch.zeros(self.seq_len, 3, self.img_size, self.img_size)
            dummy_label = torch.tensor(5, dtype=torch.long)  # "None of the above"
            return dummy_frames, dummy_label

# ===================================================================
# CELL 4: Model Architecture
# ===================================================================

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2))
    
    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes=6, seq_len=15, dropout_rate=0.3):
        super(ActionRecognitionModel, self).__init__()
        
        # 3D CNN Encoder
        self.conv3d_1 = Conv3DBlock(3, 16, kernel_size=(1, 7, 7))
        self.conv3d_2 = Conv3DBlock(16, 32, kernel_size=(1, 5, 5))
        self.conv3d_3 = Conv3DBlock(32, 64, kernel_size=(1, 3, 3))
        
        # ConvLSTM layer (using 2D ConvLSTM)
        self.convlstm = nn.LSTM(
            input_size=64 * 28 * 28,  # Flattened spatial features
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Input: (batch, seq_len, channels, height, width)
        batch_size, seq_len, C, H, W = x.shape
        
        # Reshape for 3D convolution: (batch, channels, seq_len, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        
        # 3D CNN feature extraction
        x = self.conv3d_1(x)  # (batch, 16, seq_len, H/2, W/2)
        x = self.conv3d_2(x)  # (batch, 32, seq_len, H/4, W/4)
        x = self.conv3d_3(x)  # (batch, 64, seq_len, H/8, W/8)
        
        # Reshape for LSTM: (batch, seq_len, features)
        _, C_new, T_new, H_new, W_new = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (batch, seq_len, channels, H, W)
        x = x.contiguous().view(batch_size, T_new, -1)  # (batch, seq_len, features)
        
        # LSTM for temporal modeling
        lstm_out, (hidden, _) = self.convlstm(x)
        
        # Use last hidden state for classification
        output = self.classifier(hidden[-1])  # (batch, num_classes)
        
        return output

# Test model creation
print("🏗️ Creating model...")
model = ActionRecognitionModel(num_classes=num_classes, seq_len=seq_len)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

# ===================================================================
# CELL 5: Data Loading
# ===================================================================

print("📁 Loading datasets...")

# Create datasets
train_dataset = ActionDataset(x_train_path, y_train_path, seq_len=seq_len, 
                             img_size=img_size, is_train=True)
val_dataset = ActionDataset(x_val_path, y_val_path, seq_len=seq_len, 
                           img_size=img_size, is_train=False)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=2, pin_memory=True)

print(f"✅ Data loaders ready:")
print(f"   - Training batches: {len(train_loader)}")
print(f"   - Validation batches: {len(val_loader)}")

# Test data loading
print("🔍 Testing data loading...")
try:
    for batch_frames, batch_labels in train_loader:
        print(f"   ✅ Batch shape: {batch_frames.shape}, Labels: {batch_labels.shape}")
        print(f"   ✅ Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        break
except Exception as e:
    print(f"   ❌ Error in data loading: {e}")

# ===================================================================
# CELL 6: Training Setup
# ===================================================================

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training history
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

def calculate_accuracy(outputs, labels):
    """Calculate accuracy"""
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    print(f"💾 Checkpoint saved: {filename}")

print("⚙️ Training setup complete!")
print(f"   - Optimizer: Adam (lr={learning_rate})")
print(f"   - Loss function: CrossEntropyLoss")
print(f"   - Scheduler: StepLR (step_size=7, gamma=0.1)")

# ===================================================================
# CELL 7: Training Loop
# ===================================================================

print("🚀 Starting training...")
print("=" * 60)

for epoch in range(num_epochs):
    start_time = time.time()
    
    # Training phase
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    
    for batch_idx, (frames, labels) in enumerate(train_pbar):
        # Move to device
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        try:
            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            accuracy = calculate_accuracy(outputs, labels)
            running_accuracy += accuracy
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%',
                'GPU': f'{torch.cuda.memory_allocated() / 1e9:.1f}GB'
            })
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n💥 OOM Error at batch {batch_idx}")
                print(f"   Frames shape: {frames.shape}")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e
    
    # Calculate epoch training metrics
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = running_accuracy / len(train_loader)
    
    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_running_accuracy = 0.0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        for frames, labels in val_pbar:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
            accuracy = calculate_accuracy(outputs, labels)
            val_running_accuracy += accuracy
            
            val_pbar.set_postfix({
                'Val Loss': f'{loss.item():.4f}',
                'Val Acc': f'{accuracy:.2f}%'
            })
    
    # Calculate epoch validation metrics
    epoch_val_loss = val_running_loss / len(val_loader)
    epoch_val_acc = val_running_accuracy / len(val_loader)
    
    # Update learning rate
    scheduler.step()
    
    # Store history
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)
    
    # Print epoch summary
    epoch_time = time.time() - start_time
    print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
    print(f"   Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
    print(f"   Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
    print(f"   Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
    print("-" * 60)
    
    # Save checkpoint
    if (epoch + 1) % 5 == 0:
        checkpoint_name = f'action_model_epoch_{epoch+1}.pth'
        save_checkpoint(model, optimizer, epoch, epoch_val_loss, checkpoint_name)
    
    # Memory cleanup
    torch.cuda.empty_cache()
    gc.collect()

print("🎉 Training completed!")

# ===================================================================
# CELL 8: Testing
# ===================================================================

print("🧪 Loading test dataset...")

test_dataset = ActionDataset(x_test_path, y_test_path, seq_len=seq_len, 
                            img_size=img_size, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=2, pin_memory=True)

print(f"✅ Test dataset loaded with {len(test_dataset)} videos")

# Test the model
print("🔍 Testing model...")
model.eval()
test_loss = 0.0
test_accuracy = 0.0
all_predictions = []
all_labels = []

with torch.no_grad():
    test_pbar = tqdm(test_loader, desc='Testing')
    
    for frames, labels in test_pbar:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        accuracy = calculate_accuracy(outputs, labels)
        test_accuracy += accuracy
        
        # Store predictions for analysis
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        test_pbar.set_postfix({
            'Test Loss': f'{loss.item():.4f}',
            'Test Acc': f'{accuracy:.2f}%'
        })

# Calculate final test metrics
final_test_loss = test_loss / len(test_loader)
final_test_acc = test_accuracy / len(test_loader)

print(f"\n🎯 Final Test Results:")
print(f"   Test Loss: {final_test_loss:.4f}")
print(f"   Test Accuracy: {final_test_acc:.2f}%")

# ===================================================================
# CELL 9: Results and Model Saving
# ===================================================================

# Save final model
final_model_path = 'action_recognition_pytorch_final.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'num_classes': num_classes,
        'seq_len': seq_len,
        'img_size': img_size
    },
    'action_classes': action_classes,
    'train_history': {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    },
    'test_results': {
        'test_loss': final_test_loss,
        'test_accuracy': final_test_acc
    }
}, final_model_path)

print(f"💾 Final model saved: {final_model_path}")
print("🎉 Training and evaluation completed!")
print(f"📊 Final Results Summary:")
print(f"   - Best Train Acc: {max(train_accuracies):.2f}%")
print(f"   - Best Val Acc: {max(val_accuracies):.2f}%") 
print(f"   - Test Acc: {final_test_acc:.2f}%") 