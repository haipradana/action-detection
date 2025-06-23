#!/usr/bin/env python3
"""
Screenshot-based Action Detection
Analyze static images for action recognition
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import glob
from pathlib import Path
import torchvision.transforms as transforms
from collections import Counter
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = "./model/action_recognition_pytorch_final.pth"
SCREENSHOTS_FOLDER = "./screenshots"  # Create this folder and put images
OUTPUT_JSON = "./screenshot_results.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

ACTION_CLASSES = [
    'Reach To Shelf', 'Retract From Shelf', 'Hand In Shelf',
    'Inspect Product', 'Inspect Shelf', 'None of the above'
]

# Simplified model for single image analysis
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2))
    
    def forward(self, x):
        return self.maxpool(self.relu(self.bn(self.conv3d(x))))

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes=6, seq_len=15, dropout_rate=0.3):
        super(ActionRecognitionModel, self).__init__()
        self.conv3d_1 = Conv3DBlock(3, 16, kernel_size=(1, 7, 7))
        self.conv3d_2 = Conv3DBlock(16, 32, kernel_size=(1, 5, 5))
        self.conv3d_3 = Conv3DBlock(32, 64, kernel_size=(1, 3, 3))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.convlstm = nn.LSTM(input_size=64 * 7 * 7, hidden_size=128, num_layers=1, batch_first=True, dropout=dropout_rate)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x) 
        x = self.conv3d_3(x)
        _, C_new, T_new, H_new, W_new = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(batch_size * T_new, C_new, H_new, W_new)
        x = self.adaptive_pool(x)
        x = x.view(batch_size, T_new, -1)
        lstm_out, (hidden, _) = self.convlstm(x)
        output = self.classifier(hidden[-1])
        return output

def load_model(model_path):
    print(f"📦 Loading model from: {model_path}")
    model = ActionRecognitionModel(num_classes=6, seq_len=15)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def analyze_single_image(model, image_path):
    """Analyze a single screenshot/image for action detection"""
    print(f"🖼️  Analyzing: {os.path.basename(image_path)}")
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return None
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb)
    
    # Create sequence by repeating the same image (model expects sequence)
    repeated_images = image_tensor.unsqueeze(0).repeat(15, 1, 1, 1)  # (15, C, H, W)
    input_tensor = repeated_images.unsqueeze(0).to(device)  # (1, 15, C, H, W)
    
    with torch.no_grad():
        # Forward pass
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
    
    result = {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'predicted_class': predicted_class,
        'action_name': ACTION_CLASSES[predicted_class],
        'confidence': confidence,
        'all_probabilities': {ACTION_CLASSES[i]: float(all_probs[i]) for i in range(len(ACTION_CLASSES))},
        'raw_probabilities': all_probs.tolist()
    }
    
    print(f"   Result: {ACTION_CLASSES[predicted_class]} ({confidence:.1%})")
    
    return result

def extract_video_screenshots(video_path, output_folder, interval_seconds=5):
    """Extract screenshots from video at regular intervals"""
    print(f"📹 Extracting screenshots from: {video_path}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📊 Video: {fps:.1f} FPS, {duration:.1f}s duration")
    
    frame_interval = int(fps * interval_seconds)  # Frames between screenshots
    screenshot_paths = []
    
    frame_count = 0
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save screenshot at interval
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            timestamp_str = f"{int(timestamp//60):02d}m{int(timestamp%60):02d}s"
            
            screenshot_name = f"screenshot_{screenshot_count:03d}_{timestamp_str}.jpg"
            screenshot_path = os.path.join(output_folder, screenshot_name)
            
            cv2.imwrite(screenshot_path, frame)
            screenshot_paths.append(screenshot_path)
            screenshot_count += 1
            
            print(f"   📸 Screenshot {screenshot_count}: {timestamp_str}")
        
        frame_count += 1
    
    cap.release()
    print(f"✅ Extracted {len(screenshot_paths)} screenshots")
    return screenshot_paths

def analyze_screenshots_folder(model, folder_path):
    """Analyze all images in a folder"""
    print(f"📁 Analyzing screenshots in: {folder_path}")
    
    # Supported image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_paths:
        print(f"❌ No images found in: {folder_path}")
        return []
    
    image_paths.sort()  # Sort for consistent order
    print(f"📊 Found {len(image_paths)} images")
    
    results = []
    for image_path in image_paths:
        result = analyze_single_image(model, image_path)
        if result:
            results.append(result)
    
    return results

def analyze_results(results):
    """Analyze the screenshot detection results"""
    print(f"\n📊 SCREENSHOT ANALYSIS RESULTS:")
    print("=" * 60)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Class distribution
    all_classes = [r['predicted_class'] for r in results]
    class_counter = Counter(all_classes)
    
    print(f"📈 Action Distribution ({len(results)} screenshots):")
    for class_id, count in class_counter.most_common():
        percentage = (count / len(results)) * 100
        print(f"   {ACTION_CLASSES[class_id]}: {count} screenshots ({percentage:.1f}%)")
    
    # Confidence statistics
    all_confidences = [r['confidence'] for r in results]
    print(f"\n📊 Confidence Statistics:")
    print(f"   Mean: {np.mean(all_confidences):.3f}")
    print(f"   Min: {np.min(all_confidences):.3f}")
    print(f"   Max: {np.max(all_confidences):.3f}")
    print(f"   Std: {np.std(all_confidences):.3f}")
    
    # Show individual results
    print(f"\n🖼️  Individual Screenshot Results:")
    for i, result in enumerate(results):
        print(f"{i+1:2d}. {result['image_name']:<25} | "
              f"{result['action_name']:<18} | "
              f"Conf: {result['confidence']:.1%}")
    
    # Check for diversity
    unique_classes = len(class_counter)
    print(f"\n🎯 Summary:")
    print(f"   Total screenshots: {len(results)}")
    print(f"   Unique actions detected: {unique_classes}")
    
    if unique_classes > 1:
        print("   ✅ Multiple actions detected!")
    else:
        print("   ❌ Single action only - same limitation as video")
    
    return {
        'total_screenshots': len(results),
        'unique_classes': unique_classes,
        'class_distribution': dict(class_counter),
        'confidence_stats': {
            'mean': np.mean(all_confidences),
            'min': np.min(all_confidences),
            'max': np.max(all_confidences),
            'std': np.std(all_confidences)
        }
    }

def create_visualization(results, output_path):
    """Create visualization of screenshot analysis"""
    if not results:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Action distribution pie chart
    all_classes = [r['predicted_class'] for r in results]
    class_counter = Counter(all_classes)
    
    labels = [ACTION_CLASSES[class_id] for class_id in class_counter.keys()]
    sizes = list(class_counter.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Action Distribution in Screenshots')
    
    # Confidence histogram
    confidences = [r['confidence'] for r in results]
    ax2.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Number of Screenshots')
    ax2.set_title('Confidence Score Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_path}")

def main():
    print("🚀 Screenshot-based Action Detection")
    print("=" * 60)
    
    # Check if screenshots folder exists
    if not os.path.exists(SCREENSHOTS_FOLDER):
        print(f"📁 Screenshots folder not found: {SCREENSHOTS_FOLDER}")
        
        # Option to extract from video
        video_choice = input("Extract screenshots from video? (y/n): ").strip().lower()
        if video_choice == 'y':
            video_path = input("Enter video path (default: ./mauk_ngetes/ngetes.mp4): ").strip()
            if not video_path:
                video_path = "./mauk_ngetes/ngetes.mp4"
            
            interval = input("Screenshot interval in seconds (default: 5): ").strip()
            interval = int(interval) if interval.isdigit() else 5
            
            screenshot_paths = extract_video_screenshots(video_path, SCREENSHOTS_FOLDER, interval)
            
            if not screenshot_paths:
                print("❌ No screenshots extracted. Exiting.")
                return
        else:
            print("❌ No screenshots available. Create folder and add images first.")
            return
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Analyze screenshots
    results = analyze_screenshots_folder(model, SCREENSHOTS_FOLDER)
    
    if not results:
        print("❌ No images could be analyzed")
        return
    
    # Analyze results
    stats = analyze_results(results)
    
    # Create visualization
    create_visualization(results, "./screenshot_analysis.png")
    
    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'method': 'screenshot_analysis',
        'screenshots_folder': SCREENSHOTS_FOLDER,
        'stats': stats,
        'results': results
    }
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n🎉 Screenshot analysis completed!")
    print(f"📁 Results saved to: {OUTPUT_JSON}")
    print(f"📊 Visualization: ./screenshot_analysis.png")

if __name__ == "__main__":
    main() 