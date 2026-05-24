import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mynn as nn
import numpy as np

np.random.seed(42)

cnn = nn.models.Model_CNN()
X = np.random.randn(2, 784)
y = np.array([0, 1])

# Save all weights
def save_weights(model):
    return {
        'conv1_W': model.conv1.W.copy(), 'conv1_b': model.conv1.b.copy(),
        'conv2_W': model.conv2.W.copy(), 'conv2_b': model.conv2.b.copy(),
        'fc1_W': model.fc1.W.copy(), 'fc1_b': model.fc1.b.copy(),
        'fc2_W': model.fc2.W.copy(), 'fc2_b': model.fc2.b.copy(),
    }

def restore_weights(model, w):
    model.conv1.W = w['conv1_W'].copy(); model.conv1.b = w['conv1_b'].copy()
    model.conv1.params['W'] = model.conv1.W; model.conv1.params['b'] = model.conv1.b
    model.conv2.W = w['conv2_W'].copy(); model.conv2.b = w['conv2_b'].copy()
    model.conv2.params['W'] = model.conv2.W; model.conv2.params['b'] = model.conv2.b
    model.fc1.W = w['fc1_W'].copy(); model.fc1.b = w['fc1_b'].copy()
    model.fc1.params['W'] = model.fc1.W; model.fc1.params['b'] = model.fc1.b
    model.fc2.W = w['fc2_W'].copy(); model.fc2.b = w['fc2_b'].copy()
    model.fc2.params['W'] = model.fc2.W; model.fc2.params['b'] = model.fc2.b

# Get analytical gradients
logits = cnn(X)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn, max_classes=10)
loss = loss_fn(logits, y)
loss_fn.backward()
analytical_c1 = cnn.conv1.grads['W'].copy()
analytical_c2 = cnn.conv2.grads['W'].copy()

# Numerical gradient check for conv1
eps = 1e-4
orig = save_weights(cnn)
np.random.seed(99)
test_indices = [tuple(idx) for idx in np.random.randint(0, cnn.conv1.W.shape, size=(20, 4))]

max_diff = 0
for idx in test_indices:
    # Plus
    restore_weights(cnn, orig)
    cnn.conv1.W[idx] += eps
    cnn.conv1.params['W'] = cnn.conv1.W
    logits_p = cnn(X)
    loss_p = loss_fn(logits_p, y)
    
    # Minus
    restore_weights(cnn, orig)
    cnn.conv1.W[idx] -= eps
    cnn.conv1.params['W'] = cnn.conv1.W
    logits_m = cnn(X)
    loss_m = loss_fn(logits_m, y)
    
    num_g = (loss_p - loss_m) / (2 * eps)
    ana_g = analytical_c1[idx]
    d = abs(ana_g - num_g)
    max_diff = max(max_diff, d)
    if d > 1e-4:
        print(f"  {idx}: analytical={ana_g:.8f}, numerical={num_g:.8f}, diff={d:.8f}")

print(f"\nconv1.W max diff: {max_diff:.8f}")
if max_diff < 1e-3:
    print("✅ conv1.W gradient check PASSED!")
else:
    print("❌ conv1.W gradient check FAILED!")

# Same for conv2
restore_weights(cnn, orig)
logits = cnn(X)
loss_fn2 = nn.op.MultiCrossEntropyLoss(model=cnn, max_classes=10)
loss_fn2(logits, y)
loss_fn2.backward()
analytical_c2 = cnn.conv2.grads['W'].copy()

orig2 = save_weights(cnn)
np.random.seed(77)
test_indices2 = [tuple(idx) for idx in np.random.randint(0, cnn.conv2.W.shape, size=(20, 4))]

max_diff2 = 0
for idx in test_indices2:
    restore_weights(cnn, orig2)
    cnn.conv2.W[idx] += eps
    cnn.conv2.params['W'] = cnn.conv2.W
    logits_p = cnn(X)
    loss_p = loss_fn(logits_p, y)
    
    restore_weights(cnn, orig2)
    cnn.conv2.W[idx] -= eps
    cnn.conv2.params['W'] = cnn.conv2.W
    logits_m = cnn(X)
    loss_m = loss_fn(logits_m, y)
    
    num_g = (loss_p - loss_m) / (2 * eps)
    ana_g = analytical_c2[idx]
    d = abs(ana_g - num_g)
    max_diff2 = max(max_diff2, d)
    if d > 1e-4:
        print(f"  {idx}: analytical={ana_g:.8f}, numerical={num_g:.8f}, diff={d:.8f}")

print(f"\nconv2.W max diff: {max_diff2:.8f}")
if max_diff2 < 1e-3:
    print("✅ conv2.W gradient check PASSED!")
else:
    print("❌ conv2.W gradient check FAILED!")
