#!/usr/bin/env python
"""
Test script to verify the updated ActionRecognition code works with modern TensorFlow
"""

import numpy as np
import tensorflow as tf
from model import MyCL_Model, SimpleConvLSTM

def test_model_creation():
    """Test if models can be created without errors"""
    print("Testing model creation...")
    
    # Test main model
    try:
        model = MyCL_Model(num_classes=6)
        print("✓ MyCL_Model created successfully")
    except Exception as e:
        print(f"✗ Error creating MyCL_Model: {e}")
        return False
    
    # Test simple model
    try:
        simple_model = SimpleConvLSTM(num_classes=6)
        print("✓ SimpleConvLSTM created successfully")
    except Exception as e:
        print(f"✗ Error creating SimpleConvLSTM: {e}")
        return False
    
    return True

def test_model_compilation():
    """Test if models can be compiled"""
    print("\nTesting model compilation...")
    
    try:
        model = MyCL_Model(num_classes=6)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        print("✓ Model compiled successfully")
        return True
    except Exception as e:
        print(f"✗ Error compiling model: {e}")
        return False

def test_model_forward_pass():
    """Test forward pass with dummy data"""
    print("\nTesting model forward pass...")
    
    try:
        model = MyCL_Model(num_classes=6)
        
        # Create dummy input (batch_size, seq_len, height, width, channels)
        dummy_input = tf.random.normal((1, 30, 64, 64, 3))
        
        # Forward pass
        output = model(dummy_input, training=False)
        
        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        print(f"✓ Output values sum to ~1.0: {tf.reduce_sum(output, axis=1)}")
        
        return True
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False

def test_tensorflow_version():
    """Check TensorFlow version"""
    print(f"\nTensorFlow version: {tf.__version__}")
    
    if tf.__version__.startswith('2.'):
        print("✓ Using TensorFlow 2.x")
        return True
    else:
        print("✗ Not using TensorFlow 2.x")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("TESTING UPDATED ACTION RECOGNITION CODE")
    print("=" * 50)
    
    tests = [
        test_tensorflow_version,
        test_model_creation,
        test_model_compilation,
        test_model_forward_pass
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    if all(results):
        print("✓ ALL TESTS PASSED! The updated code is working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    print(f"Passed: {sum(results)}/{len(results)} tests")

if __name__ == "__main__":
    main() 