#!/usr/bin/env python
"""
YOLO + PyTorch Action Recognition Pipeline
Universal integration supporting multiple YOLO formats
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from collections import deque, defaultdict
import time
from pathlib import Path

# Try importing different YOLO versions
try:
    from ultralytics import YOLO  # YOLOv8
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not found. Install: pip install ultralytics")

import torchvision.transforms as transforms

# ===================================================================
# ACTION MODEL CLASSES (same as training script)
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
        
        # LSTM
        self.convlstm = nn.LSTM(
            input_size=64 * 28 * 28,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        
        _, C_new, T_new, H_new, W_new = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(batch_size, T_new, -1)
        
        lstm_out, (hidden, _) = self.convlstm(x)
        output = self.classifier(hidden[-1])
        
        return output

# ===================================================================
# UNIVERSAL YOLO + ACTION PIPELINE
# ===================================================================

class UniversalYOLOActionPipeline:
    def __init__(self, yolo_model_path, action_model_path, 
                 seq_len=15, img_size=224, confidence_threshold=0.5):
        """
        Universal pipeline supporting multiple YOLO formats
        
        Args:
            yolo_model_path: Path to YOLO model (.pt, .onnx, etc.)
            action_model_path: Path to PyTorch action model (.pth)
            seq_len: Sequence length for action recognition
            img_size: Image size for action model
            confidence_threshold: YOLO confidence threshold
        """
        self.seq_len = seq_len
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Action classes
        self.action_classes = [
            'Reach To Shelf',
            'Retract From Shelf', 
            'Hand In Shelf',
            'Inspect Product',
            'Inspect Shelf',
            'None of the above'
        ]
        
        # Load YOLO model
        self.load_yolo_model(yolo_model_path)
        
        # Load action recognition model
        self.load_action_model(action_model_path)
        
        # Object tracking storage
        self.object_sequences = defaultdict(lambda: deque(maxlen=seq_len))
        self.object_counter = 0
        
        # Preprocessing for action model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Universal YOLO + Action Pipeline initialized!")
    
    def load_yolo_model(self, model_path):
        """Load YOLO model (supports multiple formats)"""
        model_path = Path(model_path)
        
        if YOLO_AVAILABLE and model_path.suffix in ['.pt', '.onnx']:
            # Use ultralytics YOLO (recommended)
            self.yolo_model = YOLO(model_path)
            self.yolo_type = 'ultralytics'
            print(f"✅ Loaded YOLOv8: {model_path}")
            
        else:
            # Fallback to OpenCV DNN
            if model_path.suffix == '.weights':
                # Darknet format
                config_path = model_path.with_suffix('.cfg')
                self.yolo_net = cv2.dnn.readNetFromDarknet(str(config_path), str(model_path))
                self.yolo_type = 'opencv_darknet'
                print(f"✅ Loaded Darknet YOLO: {model_path}")
                
            elif model_path.suffix == '.onnx':
                # ONNX format
                self.yolo_net = cv2.dnn.readNetFromONNX(str(model_path))
                self.yolo_type = 'opencv_onnx'
                print(f"✅ Loaded ONNX YOLO: {model_path}")
                
            else:
                raise ValueError(f"Unsupported YOLO format: {model_path.suffix}")
    
    def load_action_model(self, model_path):
        """Load PyTorch action recognition model"""
        # Load model state
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            num_classes = config.get('num_classes', 6)
            seq_len = config.get('seq_len', 15)
        else:
            num_classes = 6
            seq_len = 15
        
        # Create model
        self.action_model = ActionRecognitionModel(
            num_classes=num_classes, 
            seq_len=seq_len
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.action_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.action_model.load_state_dict(checkpoint)
        
        self.action_model.to(self.device)
        self.action_model.eval()
        
        print(f"✅ Loaded PyTorch action model: {model_path}")
    
    def detect_objects_ultralytics(self, frame):
        """YOLO detection using ultralytics"""
        results = self.yolo_model(frame, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        'confidence': float(conf),
                        'class_id': cls
                    })
        
        return detections
    
    def detect_objects_opencv(self, frame):
        """YOLO detection using OpenCV DNN"""
        height, width = frame.shape[:2]
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.yolo_net.setInput(blob)
        
        # Get output layer names
        layer_names = self.yolo_net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
        
        # Forward pass
        outputs = self.yolo_net.forward(output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'bbox': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i]
                })
        
        return detections
    
    def detect_objects(self, frame):
        """Universal object detection"""
        if self.yolo_type == 'ultralytics':
            return self.detect_objects_ultralytics(frame)
        else:
            return self.detect_objects_opencv(frame)
    
    def preprocess_roi(self, roi):
        """Preprocess ROI for action recognition"""
        if roi.size == 0:
            return None
        
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        roi_tensor = self.transform(roi_rgb)
        
        return roi_tensor
    
    def predict_action(self, sequence):
        """Predict action from sequence"""
        if len(sequence) < self.seq_len:
            return None, 0.0
        
        # Stack sequence
        sequence_tensor = torch.stack(list(sequence))  # (seq_len, C, H, W)
        sequence_tensor = sequence_tensor.unsqueeze(0)  # (1, seq_len, C, H, W)
        sequence_tensor = sequence_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.action_model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            action_id = predicted.item()
            action_confidence = confidence.item()
            action_name = self.action_classes[action_id]
        
        return action_name, action_confidence
    
    def process_frame(self, frame):
        """Process single frame"""
        # YOLO detection
        detections = self.detect_objects(frame)
        
        results = []
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            
            # Extract ROI
            roi = frame[y:y+h, x:x+w]
            
            # Preprocess ROI
            processed_roi = self.preprocess_roi(roi)
            if processed_roi is None:
                continue
            
            # Simple object tracking (you can improve this)
            object_id = self.object_counter % 10
            
            # Add to sequence
            self.object_sequences[object_id].append(processed_roi)
            
            # Predict action if sequence is ready
            if len(self.object_sequences[object_id]) == self.seq_len:
                action_name, action_confidence = self.predict_action(
                    self.object_sequences[object_id]
                )
                
                if action_name:
                    results.append({
                        'bbox': detection['bbox'],
                        'yolo_confidence': detection['confidence'],
                        'action_name': action_name,
                        'action_confidence': action_confidence
                    })
            
            self.object_counter += 1
        
        return results
    
    def draw_results(self, frame, results):
        """Draw detection and action results"""
        for result in results:
            x, y, w, h = result['bbox']
            action_name = result['action_name']
            action_conf = result['action_confidence']
            yolo_conf = result['yolo_confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw labels
            label = f"{action_name}: {action_conf:.2f}"
            yolo_label = f"YOLO: {yolo_conf:.2f}"
            
            # Background for text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - 40), (x + label_size[0], y), (0, 255, 0), -1)
            
            # Text
            cv2.putText(frame, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, yolo_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def process_video(self, video_path, output_path=None):
        """Process entire video"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.process_frame(frame)
            
            # Draw results
            frame = self.draw_results(frame, results)
            
            # Save or display
            if output_path:
                out.write(frame)
            else:
                cv2.imshow('YOLO + Action Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

# ===================================================================
# USAGE EXAMPLES
# ===================================================================

def main():
    """Example usage"""
    
    # Example: YOLOv8 + PyTorch Action Model
    print("🚀 YOLOv8 + PyTorch Action Recognition")
    pipeline = UniversalYOLOActionPipeline(
        yolo_model_path='yolov8n.pt',  # YOLOv8 nano
        action_model_path='action_recognition_pytorch_final.pth',  # Your trained model
        seq_len=15
    )
    
    # Process video
    # pipeline.process_video('input_video.mp4', 'output_with_actions.mp4')

if __name__ == "__main__":
    main()

# ===================================================================
# MODEL CONVERSION UTILITIES
# ===================================================================

def convert_pytorch_to_onnx(pytorch_model_path, onnx_output_path, seq_len=15, img_size=224):
    """Convert PyTorch action model to ONNX"""
    print(f"🔄 Converting {pytorch_model_path} to ONNX...")
    
    device = torch.device('cpu')  # Use CPU for ONNX export
    
    # Load model
    checkpoint = torch.load(pytorch_model_path, map_location=device)
    model = ActionRecognitionModel(num_classes=6, seq_len=seq_len)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, seq_len, 3, img_size, img_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX model saved: {onnx_output_path}")

def load_and_test_onnx(onnx_path):
    """Test ONNX model loading"""
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(onnx_path)
        print(f"✅ ONNX model loaded successfully: {onnx_path}")
        
        # Test inference
        dummy_input = np.random.randn(1, 15, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {'input': dummy_input})
        print(f"✅ ONNX inference test passed. Output shape: {outputs[0].shape}")
        
    except ImportError:
        print("⚠️ onnxruntime not installed. Install: pip install onnxruntime")
    except Exception as e:
        print(f"❌ ONNX test failed: {e}")

# Example conversion
if __name__ == "__main__":
    # Convert PyTorch to ONNX for universal compatibility
    # convert_pytorch_to_onnx(
    #     'action_recognition_pytorch_final.pth',
    #     'action_recognition_model.onnx'
    # )
    pass 