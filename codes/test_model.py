# Test script for evaluating trained models
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


def load_test_data():
    """Load MNIST test dataset."""
    test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
    test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'
    
    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28).copy()
    
    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8).copy()
    
    # normalize
    test_imgs = test_imgs / 255.0
    
    return test_imgs, test_labs


def batch_predict(model, X, batch_size=512):
    """Predict in batches to avoid memory issues."""
    all_logits = []
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_logits = model(batch_X)
        all_logits.append(batch_logits)
    return np.concatenate(all_logits, axis=0)


def test_mlp_model(model_path, test_imgs, test_labs):
    """Test MLP model."""
    print("=" * 50)
    print("Testing MLP Model")
    print("=" * 50)
    
    model = nn.models.Model_MLP()
    model.load_model(model_path)
    
    logits = batch_predict(model, test_imgs)
    accuracy = nn.metric.accuracy(logits, test_labs)
    
    print(f"MLP Test Accuracy: {accuracy:.4f}")
    
    # Compute confusion matrix
    cm = nn.metric.confusion_matrix(logits, test_labs, num_classes=10)
    
    return accuracy, cm, logits


def test_cnn_model(model_path, test_imgs, test_labs):
    """Test CNN model."""
    print("=" * 50)
    print("Testing CNN Model")
    print("=" * 50)
    
    model = nn.models.Model_CNN()
    model.load_model(model_path)
    
    logits = batch_predict(model, test_imgs)
    accuracy = nn.metric.accuracy(logits, test_labs)
    
    print(f"CNN Test Accuracy: {accuracy:.4f}")
    
    # Compute confusion matrix
    cm = nn.metric.confusion_matrix(logits, test_labs, num_classes=10)
    
    return accuracy, cm, logits


if __name__ == "__main__":
    # Create results directory
    os.makedirs('./results/figures', exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    test_imgs, test_labs = load_test_data()
    print(f"Test set: {test_imgs.shape[0]} samples")
    
    # Test MLP
    mlp_accuracy, mlp_cm, mlp_logits = test_mlp_model(
        r'./saved_models/mlp/best_model.pickle', test_imgs, test_labs)
    
    # Test CNN
    cnn_accuracy, cnn_cm, cnn_logits = test_cnn_model(
        r'./saved_models/cnn/best_model.pickle', test_imgs, test_labs)
    
    print("\n" + "=" * 50)
    print("Final Test Results")
    print("=" * 50)
    print(f"MLP Test Accuracy: {mlp_accuracy:.4f}")
    print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")
    print(f"Improvement: {(cnn_accuracy - mlp_accuracy)*100:.2f}%")
    print("=" * 50)
    
    # Save results
    results = {
        'mlp_accuracy': mlp_accuracy,
        'cnn_accuracy': cnn_accuracy,
        'mlp_confusion_matrix': mlp_cm,
        'cnn_confusion_matrix': cnn_cm
    }
    
    with open('./results/test_results.pickle', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nResults saved to ./results/test_results.pickle")
