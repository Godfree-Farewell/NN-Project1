# Neural Network and Deep Learning - Project 1

This is the implementation for Project 1 of the "Neural Network and Deep Learning" course.

## Project Structure

```
codes/
├── mynn/                   # Custom neural network library
│   ├── op.py              # Core operators (Linear, Conv2D, ReLU, etc.)
│   ├── models.py          # Model definitions (MLP, CNN)
│   ├── optimizer.py       # Optimizers (SGD, Momentum)
│   ├── lr_scheduler.py    # Learning rate schedulers
│   ├── runner.py          # Training runner
│   ├── metric.py          # Evaluation metrics
│   └── __init__.py
├── draw_tools/            # Visualization tools
│   └── plot.py
├── dataset/MNIST/         # MNIST dataset (not included in repo)
├── test_train.py          # Training script
├── test_model.py          # Testing script
└── additional_experiments.py  # Part C experiments
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

## Part A: MLP Baseline

The MLP baseline is implemented in `mynn/models.py` as `Model_MLP`.

**Architecture:**
- Input: 784 (28x28 flattened)
- Hidden: 600 units with ReLU activation
- Output: 10 units (digits 0-9)

**To train the MLP baseline:**
```bash
cd codes
python test_train.py
```

## Part B: CNN Model

The CNN model is implemented in `mynn/models.py` as `Model_CNN`.

**Architecture:**
- Conv1: 1 -> 16 channels, 3x3 kernel, ReLU, MaxPool(2x2)
- Conv2: 16 -> 32 channels, 3x3 kernel, ReLU, MaxPool(2x2)
- Flatten
- FC1: 1568 -> 128, ReLU
- FC2: 128 -> 10

**Key implementations:**
- `conv2D` in `mynn/op.py`: Manual convolution implementation
- `MaxPool2D` in `mynn/op.py`: Max pooling with backward pass
- `Flatten` in `mynn/op.py`: Reshape layer

## Part C: Additional Directions

### Direction 1: Optimization (Momentum)

Implemented `MomentGD` optimizer in `mynn/optimizer.py`.

**To run the momentum experiment:**
```bash
cd codes
python additional_experiments.py
```

### Direction 5: Error Analysis and Visualization

Includes:
- Confusion matrix visualization
- Misclassified examples analysis
- Convolution kernel visualization
- Per-class accuracy analysis

## Running the Complete Pipeline

1. **Train models:**
   ```bash
   cd codes
   python test_train.py
   ```

2. **Test models:**
   ```bash
   python test_model.py
   ```

3. **Run additional experiments:**
   ```bash
   python additional_experiments.py
   ```

## Results

Results and figures are saved in `../results/figures/`.

## Implementation Notes

### Core Operators (`mynn/op.py`)

1. **Linear Layer:**
   - Forward: `Y = X @ W + b`
   - Backward: Computes gradients for W, b, and input

2. **Conv2D Layer:**
   - Manual convolution implementation with padding support
   - Forward: Sliding window convolution
   - Backward: Gradient computation for kernels and input

3. **MultiCrossEntropyLoss:**
   - Combines Softmax and Cross-Entropy
   - Forward: Computes loss
   - Backward: Initiates gradient propagation

### Optimizers (`mynn/optimizer.py`)

1. **SGD:** Standard stochastic gradient descent with weight decay
2. **MomentGD:** SGD with momentum (mu=0.9)

### Learning Rate Schedulers (`mynn/lr_scheduler.py`)

1. **StepLR:** Decay LR at fixed intervals
2. **MultiStepLR:** Decay LR at specified milestones
3. **ExponentialLR:** Exponential decay

## Authors

- Name: [Your Name]
- Student ID: [Your Student ID]
