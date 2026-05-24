# plot the score and loss
import matplotlib.pyplot as plt
import numpy as np

colors_set = {'Kraftime' : ('#E3E37D', '#968A62')}

def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]
    
    # Train curves: use all iterations
    train_epochs = [i for i in range(len(runner.train_scores))]
    axes[0].plot(train_epochs, runner.train_loss, color=train_color, label="Train loss")
    axes[1].plot(train_epochs, runner.train_scores, color=train_color, label="Train accuracy")
    
    # Dev curves: only logged at log_iters intervals, map to iteration indices
    if len(runner.dev_loss) > 0:
        # dev was logged every log_iters iterations
        log_iters = len(runner.train_scores) // len(runner.dev_loss) if len(runner.dev_scores) > 0 else 100
        dev_epochs = [i * log_iters for i in range(len(runner.dev_scores))]
        axes[0].plot(dev_epochs, runner.dev_loss, color=dev_color, linestyle="--", label="Dev loss")
        axes[1].plot(dev_epochs, runner.dev_scores, color=dev_color, linestyle="--", label="Dev accuracy")
    
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_title("")
    axes[0].legend(loc='upper right')
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("iteration")
    axes[1].legend(loc='lower right')


def plot_comparison(runners, labels, save_path=None):
    """
    Plot comparison of multiple training runs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for runner, label in zip(runners, labels):
        epochs = [i for i in range(len(runner.train_scores))]
        axes[0].plot(epochs, runner.train_loss, label=f"{label} (train)")
        axes[0].plot(epochs, runner.dev_loss, linestyle="--", label=f"{label} (dev)")
        
        axes[1].plot(epochs, runner.train_scores, label=f"{label} (train)")
        axes[1].plot(epochs, runner.dev_scores, linestyle="--", label=f"{label} (dev)")
    
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_title("Loss")
    axes[0].legend()
    
    axes[1].set_ylabel("accuracy")
    axes[1].set_xlabel("iteration")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(cm, classes, save_path=None):
    """
    Plot confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def visualize_weights(weights, save_path=None, title="Weight Visualization"):
    """
    Visualize weights as images.
    weights: [in_dim, out_dim] or [out_channels, in_channels, h, w]
    """
    if len(weights.shape) == 2:
        # FC layer weights
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(weights.T, cmap='viridis', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        plt.colorbar(im, ax=ax)
    elif len(weights.shape) == 4:
        # Conv layer weights
        out_c, in_c, h, w = weights.shape
        n = min(out_c, 16)  # Show at most 16 filters
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.flatten()
        for i in range(n):
            if in_c == 1:
                axes[i].imshow(weights[i, 0], cmap='gray')
            else:
                # Show first channel
                axes[i].imshow(weights[i, 0], cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Filter {i}')
        # Hide unused subplots
        for i in range(n, 16):
            axes[i].axis('off')
        plt.suptitle(title)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
