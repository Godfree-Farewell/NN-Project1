from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # He 初始化: std = sqrt(2 / fan_in)
        self.W = initialize_method(size=(in_dim, out_dim)) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None
        self.params = {'W' : self.W, 'b' : self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.dot(X, self.W) + self.b
        return output

    def backward(self, grad : np.ndarray):
        batch_size = self.input.shape[0]
        self.grads['W'] = np.dot(self.input.T, grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        grad_to_prev = np.dot(grad, self.W.T)
        return grad_to_prev
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class conv2D(Layer):
    """
    2D convolutional layer — 使用 im2col + matmul 向量化实现。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He 初始化: std = sqrt(2 / fan_in), fan_in = in_channels * kernel_size^2
        fan_in = in_channels * kernel_size * kernel_size
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((out_channels, 1))
        
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def _im2col(self, X_padded):
        """将 padded 输入展开为列矩阵 [batch, in_c*kH*kW, out_H*out_W]"""
        batch, in_c, H, W = X_padded.shape
        kH = kW = self.kernel_size
        s = self.stride
        out_H = (H - kH) // s + 1
        out_W = (W - kW) // s + 1

        # 用 stride_tricks 提取所有 patch
        shape = (batch, in_c, kH, kW, out_H, out_W)
        st = X_padded.strides
        strides = (st[0], st[1], st[2], st[3], st[2]*s, st[3]*s)
        cols = np.lib.stride_tricks.as_strided(X_padded, shape=shape, strides=strides)
        # 确保连续后 reshape
        cols = np.ascontiguousarray(cols).reshape(batch, in_c * kH * kW, out_H * out_W)
        return cols, out_H, out_W

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        output: [batch, out_channels, new_H, new_W]
        """
        self.input = X
        batch, in_c, H, W = X.shape

        if self.padding > 0:
            X_padded = np.pad(np.ascontiguousarray(X),
                              ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),
                              mode='constant')
        else:
            X_padded = np.ascontiguousarray(X)

        cols, out_H, out_W = self._im2col(X_padded)

        # W_col: [out_c, in_c*kH*kW]
        W_col = self.W.reshape(self.out_channels, -1)

        # out = W_col @ cols for each batch: [batch, out_c, out_H*out_W]
        out = np.matmul(W_col, cols)  # broadcasting: [out_c, K] @ [batch, K, L] -> [batch, out_c, L]
        out = out.reshape(batch, self.out_channels, out_H, out_W)
        out += self.b.reshape(1, -1, 1, 1)

        self._X_padded = X_padded
        self._cols = cols
        self._out_H = out_H
        self._out_W = out_W
        return out

    def backward(self, grads):
        """
        grads: [batch, out_c, out_H, out_W]
        """
        batch = grads.shape[0]
        kH = kW = self.kernel_size
        in_c = self.in_channels

        grads_col = grads.reshape(batch, self.out_channels, -1)  # [batch, out_c, N]
        W_col = self.W.reshape(self.out_channels, -1)              # [out_c, K]

        # dW = sum_b grads_col[b] @ cols[b]^T  => [out_c, K]
        dW_col = np.matmul(grads_col, self._cols.transpose(0, 2, 1)).sum(axis=0)
        self.grads['W'] = dW_col.reshape(self.W.shape)

        # db
        self.grads['b'] = grads.sum(axis=(0, 2, 3)).reshape(-1, 1)

        # dX_col = W_col^T @ grads_col  => [batch, K, N]
        dX_col = np.matmul(W_col.T, grads_col)  # [K, out_c] @ [batch, out_c, N] -> [batch, K, N]

        # col2im: 散射回 padded input
        _, _, H_pad, W_pad = self._X_padded.shape
        out_H, out_W = self._out_H, self._out_W
        grad_padded = np.zeros((batch, in_c, H_pad, W_pad), dtype=self._X_padded.dtype)

        dX_col_reshaped = dX_col.reshape(batch, in_c, kH, kW, out_H, out_W)

        for oh in range(out_H):
            for ow in range(out_W):
                h_start = oh * self.stride
                w_start = ow * self.stride
                grad_padded[:, :, h_start:h_start+kH, w_start:w_start+kW] += dX_col_reshaped[:, :, :, :, oh, ow]

        if self.padding > 0:
            p = self.padding
            grad_input = grad_padded[:, :, p:-p, p:-p]
        else:
            grad_input = grad_padded

        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X < 0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class MaxPool2D(Layer):
    """
    Max pooling layer — 向量化实现。
    """
    def __init__(self, pool_size=2, stride=2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.optimizable = False
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        """
        X: [batch, channels, H, W]
        output: [batch, channels, new_H, new_W]
        """
        self.input = X.copy()  # Save a copy to avoid view issues
        X = np.ascontiguousarray(X)
        batch, channels, H, W = X.shape
        pH = pW = self.pool_size
        sH = sW = self.stride
        new_H = (H - pH) // sH + 1
        new_W = (W - pW) // sW + 1

        shape = (batch, channels, new_H, new_W, pH, pW)
        st = X.strides
        strides = (st[0], st[1], st[2]*sH, st[3]*sW, st[2], st[3])
        windows = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

        output = windows.max(axis=(4, 5))
        self._output = output
        return output
    
    def backward(self, grads):
        """
        grads: [batch, channels, new_H, new_W]
        """
        X = self.input
        batch, channels, H, W = X.shape
        pH = pW = self.pool_size
        sH = sW = self.stride
        new_H = (H - pH) // sH + 1
        new_W = (W - pW) // sW + 1

        grad_input = np.zeros_like(X)

        for oh in range(new_H):
            for ow in range(new_W):
                h_start = oh * sH
                h_end = h_start + pH
                w_start = ow * sW
                w_end = w_start + pW
                window = X[:, :, h_start:h_end, w_start:w_end]
                out_val = self._output[:, :, oh, ow][:, :, np.newaxis, np.newaxis]
                mask = (window == out_val)
                grad_w = grads[:, :, oh, ow][:, :, np.newaxis, np.newaxis] * mask
                count = mask.sum(axis=(2, 3), keepdims=True).clip(min=1)
                grad_w = grad_w / count
                grad_input[:, :, h_start:h_end, w_start:w_end] += grad_w

        return grad_input
    
    def clear_grad(self):
        pass


class Flatten(Layer):
    """
    Flatten layer to convert 4D tensor to 2D.
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        self.input_shape = X.shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)
    
    def backward(self, grads):
        return grads.reshape(self.input_shape)
    
    def clear_grad(self):
        pass


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer with Softmax.
    """
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        batch_size = predicts.shape[0]
        
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts
        
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        true_class_probs = probs[np.arange(batch_size), labels]
        loss = -np.mean(np.log(true_class_probs))
        self.probs = probs
        return loss
    
    def backward(self):
        batch_size = self.predicts.shape[0]
        self.grads = self.probs.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    

class L2Regularization(Layer):
    pass
       

def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
