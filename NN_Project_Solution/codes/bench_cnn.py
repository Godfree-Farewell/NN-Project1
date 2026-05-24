import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mynn as nn
import numpy as np

np.random.seed(42)

cnn = nn.models.Model_CNN()
X = np.random.randn(32, 784)
y = np.random.randint(0, 10, size=32)

loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn, max_classes=10)
opt = nn.optimizer.SGD(init_lr=0.05, model=cnn)

# Warmup
logits = cnn(X)
loss = loss_fn(logits, y)
loss_fn.backward()
opt.step()

# Benchmark: 10 iterations
t0 = time.time()
for i in range(10):
    logits = cnn(X)
    loss = loss_fn(logits, y)
    loss_fn.backward()
    opt.step()
t1 = time.time()

print(f"10 iterations took {t1-t0:.2f}s")
print(f"Average per iteration: {(t1-t0)/10:.2f}s")
print(f"Estimated time for 5 epochs (1562 iters): {(t1-t0)/10 * 1562 / 60:.1f} minutes")
print(f"Loss: {loss:.4f}")
