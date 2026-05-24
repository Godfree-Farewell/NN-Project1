"""Quick test: verify CNN can learn with He initialization"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mynn as nn

np.random.seed(42)

# Load small subset
from struct import unpack
import gzip

with gzip.open('./dataset/MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28).copy()[:256] / 255.0

with gzip.open('./dataset/MNIST/train-labels-idx1-ubyte.gz', 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    labs = np.frombuffer(f.read(), dtype=np.uint8).copy()[:256]

print(f"Data: {imgs.shape}, Labels: {labs.shape}")
print(f"Weight init scale - Conv1 W std: {nn.models.Model_CNN().layers[0].W.std():.4f}")

# Train CNN
model = nn.models.Model_CNN()
optimizer = nn.optimizer.SGD(init_lr=0.05, model=model)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

batch_size = 32
for step in range(200):
    idx = np.random.choice(len(imgs), batch_size, replace=False)
    X, y = imgs[idx], labs[idx]
    
    logits = model(X)
    loss = loss_fn(logits, y)
    loss_fn.backward()
    optimizer.step()
    
    if step % 20 == 0:
        preds = logits.argmax(axis=1)
        acc = (preds == y).mean()
        print(f"Step {step:3d}: loss={loss:.4f}, acc={acc:.4f}")

print("\nCNN training test PASSED!" if loss < 1.0 else "\nCNN still not learning!")
