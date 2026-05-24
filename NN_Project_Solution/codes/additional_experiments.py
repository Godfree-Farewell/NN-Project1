# Additional experiments for Part C
# Direction 1: Optimization (Momentum)
# Direction 5: Error Analysis and Visualization

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

np.random.seed(309)


def load_data():
    """Load MNIST dataset."""
    train_images_path = r'./dataset/MNIST/train-images-idx3-ubyte.gz'
    train_labels_path = r'./dataset/MNIST/train-labels-idx1-ubyte.gz'
    
    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28).copy()
    
    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8).copy()
    
    # Split train/valid
    idx = np.random.permutation(np.arange(num))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]
    
    # Normalize
    train_imgs = train_imgs / 255.0
    valid_imgs = valid_imgs / 255.0
    
    return train_imgs, train_labs, valid_imgs, valid_labs


def batch_predict(model, X, batch_size=512):
    """Predict in batches to avoid memory issues."""
    all_logits = []
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_logits = model(batch_X)
        all_logits.append(batch_logits)
    return np.concatenate(all_logits, axis=0)


def experiment_momentum(train_imgs, train_labs, valid_imgs, valid_labs):
    """
    Direction 1: Optimization - Compare SGD vs Momentum
    """
    print("=" * 60)
    print("Direction 1: Optimization - Momentum vs SGD")
    print("=" * 60)
    
    results = {}
    
    # Baseline SGD
    print("\n--- Training with SGD ---")
    mlp_sgd = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
    optimizer_sgd = nn.optimizer.SGD(init_lr=0.01, model=mlp_sgd)
    scheduler_sgd = nn.lr_scheduler.MultiStepLR(optimizer=optimizer_sgd, milestones=[800, 2400, 4000], gamma=0.5)
    loss_fn_sgd = nn.op.MultiCrossEntropyLoss(model=mlp_sgd, max_classes=10)
    runner_sgd = nn.runner.RunnerM(mlp_sgd, optimizer_sgd, nn.metric.accuracy, loss_fn_sgd, scheduler=scheduler_sgd)
    runner_sgd.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=200, save_dir='./saved_models/sgd')
    results['sgd'] = runner_sgd
    
    # Momentum
    print("\n--- Training with Momentum (mu=0.9) ---")
    mlp_momentum = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
    optimizer_momentum = nn.optimizer.MomentGD(init_lr=0.01, model=mlp_momentum, mu=0.9)
    scheduler_momentum = nn.lr_scheduler.MultiStepLR(optimizer=optimizer_momentum, milestones=[800, 2400, 4000], gamma=0.5)
    loss_fn_momentum = nn.op.MultiCrossEntropyLoss(model=mlp_momentum, max_classes=10)
    runner_momentum = nn.runner.RunnerM(mlp_momentum, optimizer_momentum, nn.metric.accuracy, loss_fn_momentum, scheduler=scheduler_momentum)
    runner_momentum.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=200, save_dir='./saved_models/momentum')
    results['momentum'] = runner_momentum
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(runner_sgd.train_loss, label='SGD (train)', alpha=0.7)
    axes[0].plot(runner_sgd.dev_loss, '--', label='SGD (dev)', alpha=0.7)
    axes[0].plot(runner_momentum.train_loss, label='Momentum (train)', alpha=0.7)
    axes[0].plot(runner_momentum.dev_loss, '--', label='Momentum (dev)', alpha=0.7)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Comparison: SGD vs Momentum')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(runner_sgd.train_scores, label='SGD (train)', alpha=0.7)
    axes[1].plot(runner_sgd.dev_scores, '--', label='SGD (dev)', alpha=0.7)
    axes[1].plot(runner_momentum.train_scores, label='Momentum (train)', alpha=0.7)
    axes[1].plot(runner_momentum.dev_scores, '--', label='Momentum (dev)', alpha=0.7)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Comparison: SGD vs Momentum')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/figures/momentum_comparison.png', dpi=150)
    plt.close()
    
    print("\n--- Results Summary ---")
    print(f"SGD Best Validation Accuracy: {runner_sgd.best_score:.4f}")
    print(f"Momentum Best Validation Accuracy: {runner_momentum.best_score:.4f}")
    print(f"Improvement: {(runner_momentum.best_score - runner_sgd.best_score)*100:.2f}%")
    
    return results


def experiment_visualization():
    """
    Direction 5: Error Analysis and Visualization
    """
    print("=" * 60)
    print("Direction 5: Error Analysis and Visualization")
    print("=" * 60)
    
    # Load test data
    test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
    test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'
    
    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28).copy()
    
    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8).copy()
    
    test_imgs = test_imgs / 255.0
    
    # Load CNN model
    cnn_model = nn.models.Model_CNN()
    cnn_model.load_model('./saved_models/cnn/best_model.pickle')
    
    # Get predictions (batched)
    logits = batch_predict(cnn_model, test_imgs)
    predictions = np.argmax(logits, axis=-1)
    
    # 1. Confusion Matrix
    print("\n--- Generating Confusion Matrix ---")
    cm = nn.metric.confusion_matrix(logits, test_labs, num_classes=10)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = [str(i) for i in range(10)]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix (CNN on Test Set)',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('./results/figures/confusion_matrix.png', dpi=150)
    plt.close()
    
    # 2. Misclassified Examples
    print("\n--- Analyzing Misclassified Examples ---")
    misclassified_idx = np.where(predictions != test_labs)[0]
    print(f"Total misclassified: {len(misclassified_idx)} out of {len(test_labs)}")
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()
    
    for i in range(min(15, len(misclassified_idx))):
        idx = misclassified_idx[i]
        img = test_imgs[idx].reshape(28, 28)
        true_label = test_labs[idx]
        pred_label = predictions[idx]
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}')
        axes[i].axis('off')
    
    for i in range(min(15, len(misclassified_idx)), 15):
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Examples (CNN)', fontsize=14)
    plt.tight_layout()
    plt.savefig('./results/figures/misclassified_examples.png', dpi=150)
    plt.close()
    
    # 3. Visualize Convolution Kernels
    print("\n--- Visualizing Convolution Kernels ---")
    
    # First conv layer kernels
    conv1_weights = cnn_model.conv1.W  # [16, 1, 3, 3]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()
    
    for i in range(16):
        axes[i].imshow(conv1_weights[i, 0], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i}')
    
    plt.suptitle('First Conv Layer Kernels (16 filters, 3x3)', fontsize=14)
    plt.tight_layout()
    plt.savefig('./results/figures/conv1_kernels.png', dpi=150)
    plt.close()
    
    # Second conv layer kernels (show first 16 filters, first channel)
    conv2_weights = cnn_model.conv2.W  # [32, 16, 3, 3]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()
    
    for i in range(16):
        axes[i].imshow(conv2_weights[i, 0], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i}')
    
    plt.suptitle('Second Conv Layer Kernels (First 16 filters, channel 0)', fontsize=14)
    plt.tight_layout()
    plt.savefig('./results/figures/conv2_kernels.png', dpi=150)
    plt.close()
    
    # 4. Per-class accuracy analysis
    print("\n--- Per-class Accuracy ---")
    per_class_correct = np.zeros(10)
    per_class_total = np.zeros(10)
    
    for true_label, pred_label in zip(test_labs, predictions):
        per_class_total[true_label] += 1
        if true_label == pred_label:
            per_class_correct[true_label] += 1
    
    per_class_accuracy = per_class_correct / per_class_total
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(10), per_class_accuracy, color='steelblue', edgecolor='black')
    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-class Accuracy on Test Set (CNN)')
    ax.set_xticks(range(10))
    ax.set_ylim([0.9, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('./results/figures/per_class_accuracy.png', dpi=150)
    plt.close()
    
    # Print per-class accuracy
    for i in range(10):
        print(f"Digit {i}: {per_class_accuracy[i]:.4f} ({int(per_class_correct[i])}/{int(per_class_total[i])})")
    
    return cm, misclassified_idx


if __name__ == "__main__":
    # Create directories
    os.makedirs('./results/figures', exist_ok=True)
    os.makedirs('./saved_models/sgd', exist_ok=True)
    os.makedirs('./saved_models/momentum', exist_ok=True)
    
    # Load data
    print("Loading MNIST dataset...")
    train_imgs, train_labs, valid_imgs, valid_labs = load_data()
    print(f"Training set: {train_imgs.shape[0]} samples")
    print(f"Validation set: {valid_imgs.shape[0]} samples")
    
    # Run experiments
    print("\n" + "=" * 60)
    print("Running Additional Experiments for Part C")
    print("=" * 60)
    
    # Direction 1: Optimization
    momentum_results = experiment_momentum(train_imgs, train_labs, valid_imgs, valid_labs)
    
    # Direction 5: Visualization
    cm, misclassified = experiment_visualization()
    
    print("\n" + "=" * 60)
    print("Additional Experiments Complete!")
    print("=" * 60)
