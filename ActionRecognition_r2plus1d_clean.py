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

# Model configuration - R(2+1)D optimized
seq_len = 16  # R(2+1)D works better with 16 frames
batch_size = 4  # Smaller batch size due to R(2+1)D memory requirements
img_size = 112  # R(2+1)D typically uses 112x112
num_classes = 6
num_epochs = 25
learning_rate = 0.0001  # Lower LR for transfer learning

# Action classes
action_classes = [
    'Reach To Shelf',
    'Retract From Shelf', 
    'Hand In Shelf',
    'Inspect Product',
    'Inspect Shelf',
    'None of the above'
]

print(f"📊 R(2+1)D Configuration:")
print(f"   - Sequence length: {seq_len}")
print(f"   - Batch size: {batch_size}")
print(f"   - Image size: {img_size}x{img_size}")
print(f"   - Number of classes: {num_classes}")

# ===================================================================
# CELL 3: Enhanced Dataset Class for R(2+1)D
# ===================================================================

class R2Plus1DDataset(Dataset):
    def __init__(self, x_path, y_path=None, seq_len=16, img_size=112, is_train=True):
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
        
        # Enhanced preprocessing for R(2+1)D with augmentation
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size + 16, img_size + 16)),  # Larger for cropping
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                                   std=[0.22803, 0.22145, 0.216989])  # Kinetics stats
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                                   std=[0.22803, 0.22145, 0.216989])  # Kinetics stats
            ])
        
        print(f"✅ R(2+1)D Dataset initialized with {len(self.video_folders)} videos")
    
    def __len__(self):
        return len(self.video_folders)
    
    def load_video_sequence(self, folder_name):
        """Load and preprocess video sequence for R(2+1)D"""
        folder_path = os.path.join(self.x_path, folder_name)
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(folder_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {folder_path}")
        
        # Smart frame sampling for exact sequence length
        if len(image_files) > self.seq_len:
            if self.is_train:
                # Random start for training
                start_idx = np.random.randint(0, len(image_files) - self.seq_len + 1)
            else:
                # Center sampling for validation/test
                start_idx = (len(image_files) - self.seq_len) // 2
            
            selected_files = image_files[start_idx:start_idx + self.seq_len]
        elif len(image_files) < self.seq_len:
            # Repeat frames to reach sequence length
            selected_files = []
            repeat_factor = self.seq_len // len(image_files) + 1
            extended_files = image_files * repeat_factor
            selected_files = extended_files[:self.seq_len]
        else:
            selected_files = image_files
        
        frames = []
        for img_file in selected_files:
            img_path = os.path.join(folder_path, img_file)
            
            # Load image
            frame = cv2.imread(img_path)
            if frame is not None:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply transforms
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
        
        if len(frames) != self.seq_len:
            raise ValueError(f"Expected {self.seq_len} frames, got {len(frames)} from {folder_path}")
        
        # Stack frames: (T, C, H, W)
        frames_tensor = torch.stack(frames)
        
        return frames_tensor
    
    def get_labels(self, folder_name):
        """Get labels for the video"""
        if self.labels_dict is None:
            return torch.tensor(5, dtype=torch.long)  # Default to "None of the above"
        
        # Try different key formats
        key = '_'.join(folder_name.split('_')[:2])
        if key not in self.labels_dict:
            key = folder_name
            if key not in self.labels_dict:
                print(f"⚠️ Warning: No labels found for {folder_name}, using default")
                return torch.tensor(5, dtype=torch.long)
        
        labels = self.labels_dict[key]
        # For action recognition, take the most common label
        if isinstance(labels, (list, np.ndarray)):
            # Get most frequent label
            unique, counts = np.unique(labels, return_counts=True)
            most_common_label = unique[np.argmax(counts)]
            return torch.tensor(most_common_label, dtype=torch.long)
        else:
            return torch.tensor(labels, dtype=torch.long)
    
    def __getitem__(self, idx):
        folder_name = self.video_folders[idx]
        
        try:
            # Load video sequence
            frames = self.load_video_sequence(folder_name)
            
            # Load labels
            action_label = self.get_labels(folder_name)
            
            return frames, action_label
            
        except Exception as e:
            print(f"Error loading {folder_name}: {e}")
            # Return dummy data
            dummy_frames = torch.zeros(self.seq_len, 3, self.img_size, self.img_size)
            dummy_label = torch.tensor(5, dtype=torch.long)  # "None of the above"
            return dummy_frames, dummy_label

# ===================================================================
# CELL 4: R(2+1)D Model Architecture
# ===================================================================

class R2Plus1DActionModel(nn.Module):
    """R(2+1)D model for action recognition with transfer learning"""
    
    def __init__(self, num_classes=6, dropout_rate=0.5, pretrained=True):
        super(R2Plus1DActionModel, self).__init__()
        
        # Import R(2+1)D from torchvision
        try:
            from torchvision.models.video import r2plus1d_18
            print("📦 Loading R(2+1)D-18 pretrained model...")
            
            # Load pretrained R(2+1)D
            self.backbone = r2plus1d_18(pretrained=pretrained)
            
            if pretrained:
                print("✅ Using pretrained weights from Kinetics-400")
            else:
                print("⚠️ Training from scratch (no pretrained weights)")
                
        except ImportError:
            raise ImportError("R(2+1)D not available. Please update PyTorch to version >= 1.7")
        
        # Get the number of features from the original classifier
        num_features = self.backbone.fc.in_features
        
        # Replace the final classifier with our custom head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Freeze early layers for transfer learning
        if pretrained:
            self._freeze_early_layers()
        
        # Initialize the new classifier layers
        self._initialize_classifier()
        
        print(f"🏗️ R(2+1)D model created with {num_classes} output classes")
    
    def _freeze_early_layers(self):
        """Freeze early layers for transfer learning"""
        # Freeze the first few blocks
        layers_to_freeze = [
            self.backbone.stem,
            self.backbone.layer1,
            self.backbone.layer2
        ]
        
        frozen_params = 0
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"❄️ Frozen {frozen_params:,} parameters for transfer learning")
    
    def _initialize_classifier(self):
        """Initialize the new classifier layers"""
        for module in self.backbone.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Input: (batch, seq_len, channels, height, width)
        # R(2+1)D expects: (batch, channels, seq_len, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Forward through R(2+1)D
        output = self.backbone(x)
        
        return output

# Test model creation
print("🏗️ Creating R(2+1)D model...")
model = R2Plus1DActionModel(num_classes=num_classes, dropout_rate=0.5, pretrained=True)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

# Test model with dummy input to verify dimensions
print("🧪 Testing model dimensions...")
with torch.no_grad():
    dummy_input = torch.randn(1, seq_len, 3, img_size, img_size).to(device)
    try:
        dummy_output = model(dummy_input)
        print(f"✅ Model test successful!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {dummy_output.shape}")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        raise e

# ===================================================================
# CELL 5: Data Loading
# ===================================================================

print("📁 Loading datasets...")

# Create datasets
train_dataset = R2Plus1DDataset(x_train_path, y_train_path, seq_len=seq_len, 
                               img_size=img_size, is_train=True)
val_dataset = R2Plus1DDataset(x_val_path, y_val_path, seq_len=seq_len, 
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

# Loss function with label smoothing for better generalization
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Enhanced training setup
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Learning rate scheduler with warmup
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)

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

print("⚙️ R(2+1)D Training setup complete!")
print(f"   - Optimizer: AdamW (lr={learning_rate})")
print(f"   - Loss function: LabelSmoothingCrossEntropy")
print(f"   - Scheduler: CosineAnnealingWarmRestarts")

# ===================================================================
# CELL 7: Training Loop
# ===================================================================

print("🚀 Starting R(2+1)D training...")
print("=" * 60)

best_val_acc = 0.0

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
            if batch_idx % 5 == 0:
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
    
    # Save best model
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        best_model_path = 'r2plus1d_best_model.pth'
        save_checkpoint(model, optimizer, epoch, epoch_val_loss, best_model_path)
        print(f"🎯 New best validation accuracy: {best_val_acc:.2f}%")
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_name = f'r2plus1d_epoch_{epoch+1}.pth'
        save_checkpoint(model, optimizer, epoch, epoch_val_loss, checkpoint_name)
    
    # Memory cleanup
    torch.cuda.empty_cache()
    gc.collect()

print("🎉 R(2+1)D training completed!")

# ===================================================================
# CELL 8: Testing
# ===================================================================

print("🧪 Loading test dataset...")

test_dataset = R2Plus1DDataset(x_test_path, y_test_path, seq_len=seq_len, 
                              img_size=img_size, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=2, pin_memory=True)

print(f"✅ Test dataset loaded with {len(test_dataset)} videos")

# Load best model for testing
print("📦 Loading best model for testing...")
checkpoint = torch.load('r2plus1d_best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Test the model
print("🔍 Testing R(2+1)D model...")
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

print(f"\n🎯 Final R(2+1)D Test Results:")
print(f"   Test Loss: {final_test_loss:.4f}")
print(f"   Test Accuracy: {final_test_acc:.2f}%")
print(f"   Best Val Accuracy: {best_val_acc:.2f}%")

# ===================================================================
# CELL 9: Results and Model Saving
# ===================================================================

# Save final model
final_model_path = 'action_recognition_r2plus1d_final.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'num_classes': num_classes,
        'seq_len': seq_len,
        'img_size': img_size,
        'model_type': 'R(2+1)D-18'
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
        'test_accuracy': final_test_acc,
        'best_val_accuracy': best_val_acc
    }
}, final_model_path)

print(f"💾 Final R(2+1)D model saved: {final_model_path}")
print("🎉 R(2+1)D training and evaluation completed!")
print(f"📊 Final Results Summary:")
print(f"   - Best Train Acc: {max(train_accuracies):.2f}%")
print(f"   - Best Val Acc: {best_val_acc:.2f}%") 
print(f"   - Test Acc: {final_test_acc:.2f}%")
print(f"   - Model: R(2+1)D-18 with transfer learning")
print(f"   - Trainable parameters: {trainable_params:,}")

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Val Loss', color='red')
plt.title('R(2+1)D Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Acc', color='blue')
plt.plot(val_accuracies, label='Val Acc', color='red')
plt.title('R(2+1)D Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('r2plus1d_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("📈 Training history plot saved as 'r2plus1d_training_history.png'")

# =================================================================== 