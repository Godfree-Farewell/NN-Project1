# Basic functionality test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mynn as nn

print('Testing basic imports...')
print('mynn modules loaded successfully!')

# Test Linear layer
print('\nTesting Linear layer...')
linear = nn.op.Linear(10, 5)
x = np.random.randn(3, 10)
out = linear(x)
print(f'Linear forward: input {x.shape} -> output {out.shape}')

# Test backward
grad = np.random.randn(3, 5)
grad_in = linear.backward(grad)
print(f'Linear backward: output grad {grad.shape} -> input grad {grad_in.shape}')
print(f'W grad shape: {linear.grads["W"].shape}')
print(f'b grad shape: {linear.grads["b"].shape}')

# Test ReLU
print('\nTesting ReLU...')
relu = nn.op.ReLU()
x = np.array([[-1, 0, 1], [2, -3, 4]])
out = relu(x)
print(f'ReLU input: {x}')
print(f'ReLU output: {out}')

# Test Softmax + CrossEntropy
print('\nTesting MultiCrossEntropyLoss...')
model = nn.models.Model_MLP([10, 5, 3], 'ReLU')
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=3)
predicts = np.random.randn(4, 3)
labels = np.array([0, 1, 2, 1])
loss = loss_fn(predicts, labels)
print(f'Loss: {loss:.4f}')

# Test CNN
print('\nTesting CNN model...')
cnn = nn.models.Model_CNN()
x = np.random.randn(2, 784)  # 2 samples, flattened 28x28
out = cnn(x)
print(f'CNN forward: input {x.shape} -> output {out.shape}')

# Test Conv2D
print('\nTesting Conv2D...')
conv = nn.op.conv2D(in_channels=1, out_channels=4, kernel_size=3, padding=1)
x = np.random.randn(2, 1, 28, 28)
out = conv(x)
print(f'Conv2D forward: input {x.shape} -> output {out.shape}')

# Test MaxPool2D
print('\nTesting MaxPool2D...')
pool = nn.op.MaxPool2D(pool_size=2, stride=2)
x = np.random.randn(2, 4, 28, 28)
out = pool(x)
print(f'MaxPool2D forward: input {x.shape} -> output {out.shape}')

# Test Flatten
print('\nTesting Flatten...')
flatten = nn.op.Flatten()
x = np.random.randn(2, 4, 7, 7)
out = flatten(x)
print(f'Flatten forward: input {x.shape} -> output {out.shape}')

# Test optimizers
print('\nTesting SGD optimizer...')
model = nn.models.Model_MLP([10, 5, 3], 'ReLU')
optimizer = nn.optimizer.SGD(init_lr=0.01, model=model)
print('SGD optimizer created successfully')

print('\nTesting MomentGD optimizer...')
optimizer = nn.optimizer.MomentGD(init_lr=0.01, model=model, mu=0.9)
print('MomentGD optimizer created successfully')

# Test schedulers
print('\nTesting schedulers...')
optimizer = nn.optimizer.SGD(init_lr=0.01, model=model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[100, 200], gamma=0.5)
print('MultiStepLR scheduler created successfully')

print('\n' + '='*50)
print('All basic tests passed!')
print('='*50)
