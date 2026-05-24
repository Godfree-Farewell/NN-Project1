# Training script for MLP and CNN models
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

def load_mnist_data():
    """Load MNIST dataset."""
    train_images_path = r'./dataset/MNIST/train-images-idx3-ubyte.gz'
    train_labels_path = r'./dataset/MNIST/train-labels-idx1-ubyte.gz'
    
    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28).copy()
    
    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8).copy()
    
    return train_imgs, train_labs


def prepare_data(train_imgs, train_labs):
    """Prepare train/validation split."""
    num = len(train_imgs)
    idx = np.random.permutation(np.arange(num))
    
    with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
    
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]
    
    train_imgs = train_imgs / 255.0
    valid_imgs = valid_imgs / 255.0
    
    return train_imgs, train_labs, valid_imgs, valid_labs


def train_mlp_baseline(train_imgs, train_labs, valid_imgs, valid_labs, save_dir='./saved_models/mlp'):
    """Train MLP baseline model."""
    print("=" * 50)
    print("Training MLP Baseline")
    print("=" * 50)
    
    mlp_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
    optimizer = nn.optimizer.SGD(init_lr=0.01, model=mlp_model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=mlp_model, max_classes=train_labs.max()+1)
    
    runner = nn.runner.RunnerM(mlp_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
    
    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], 
                 num_epochs=5, log_iters=100, save_dir=save_dir)
    
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes = axes.reshape(-1)
    plot(runner, axes)
    plt.tight_layout()
    plt.savefig('./results/figures/mlp_learning_curve.png', dpi=150)
    plt.close()
    
    print(f"MLP Best Validation Accuracy: {runner.best_score:.4f}")
    return runner


def train_cnn_model(train_imgs, train_labs, valid_imgs, valid_labs, save_dir='./saved_models/cnn'):
    """Train CNN model."""
    print("=" * 50)
    print("Training CNN Model")
    print("=" * 50)
    
    cnn_model = nn.models.Model_CNN()
    optimizer = nn.optimizer.SGD(init_lr=0.05, model=cnn_model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[2000, 5000, 7000], gamma=0.5)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max()+1)
    
    runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
    
    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], 
                 num_epochs=10, log_iters=200, save_dir=save_dir)
    
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes = axes.reshape(-1)
    plot(runner, axes)
    plt.tight_layout()
    plt.savefig('./results/figures/cnn_learning_curve.png', dpi=150)
    plt.close()
    
    print(f"CNN Best Validation Accuracy: {runner.best_score:.4f}")
    return runner


if __name__ == "__main__":
    os.makedirs('./results/figures', exist_ok=True)
    os.makedirs('./saved_models/mlp', exist_ok=True)
    os.makedirs('./saved_models/cnn', exist_ok=True)
    
    print("Loading MNIST dataset...")
    train_imgs, train_labs = load_mnist_data()
    train_imgs, train_labs, valid_imgs, valid_labs = prepare_data(train_imgs, train_labs)
    
    print(f"Training set: {train_imgs.shape[0]} samples")
    print(f"Validation set: {valid_imgs.shape[0]} samples")
    
    mlp_runner = train_mlp_baseline(train_imgs, train_labs, valid_imgs, valid_labs)
    
    cnn_runner = train_cnn_model(train_imgs, train_labs, valid_imgs, valid_labs)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"MLP Best Accuracy: {mlp_runner.best_score:.4f}")
    print(f"CNN Best Accuracy: {cnn_runner.best_score:.4f}")
    print("=" * 50)
