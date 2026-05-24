from .op import *
import pickle
import numpy as np

class Model_MLP(Layer):
    """
    A model with linear layers.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func
        self.layers = []

        if size_list is not None and act_func is not None:
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 
                                 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN(Layer):
    """
    A CNN model for MNIST classification.
    Architecture: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten -> Linear -> ReLU -> Linear
    """
    def __init__(self):
        super().__init__()
        self.layers = []
        
        # First conv block: 1 -> 16 channels, 3x3 kernel
        self.conv1 = conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        
        # Second conv block: 16 -> 32 channels, 3x3 kernel
        self.conv2 = conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        
        # Flatten and FC layers
        # After two 2x2 pooling: 28 -> 14 -> 7, so 32 * 7 * 7 = 1568
        self.flatten = Flatten()
        self.fc1 = Linear(in_dim=32 * 7 * 7, out_dim=128)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_dim=128, out_dim=10)
        
        # Build layer list
        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten, self.fc1, self.relu3, self.fc2
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        X: [batch, 1, 28, 28] for MNIST
        """
        # Ensure input is 4D
        if len(X.shape) == 2:
            batch_size = X.shape[0]
            X = np.ascontiguousarray(X.reshape(batch_size, 1, 28, 28))
        
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, save_path):
        with open(save_path, 'rb') as f:
            params = pickle.load(f)
        
        # Load conv1
        self.conv1.W = params['conv1']['W']
        self.conv1.b = params['conv1']['b']
        self.conv1.params['W'] = self.conv1.W
        self.conv1.params['b'] = self.conv1.b
        
        # Load conv2
        self.conv2.W = params['conv2']['W']
        self.conv2.b = params['conv2']['b']
        self.conv2.params['W'] = self.conv2.W
        self.conv2.params['b'] = self.conv2.b
        
        # Load fc1
        self.fc1.W = params['fc1']['W']
        self.fc1.b = params['fc1']['b']
        self.fc1.params['W'] = self.fc1.W
        self.fc1.params['b'] = self.fc1.b
        
        # Load fc2
        self.fc2.W = params['fc2']['W']
        self.fc2.b = params['fc2']['b']
        self.fc2.params['W'] = self.fc2.W
        self.fc2.params['b'] = self.fc2.b
        
    def save_model(self, save_path):
        params = {
            'conv1': {'W': self.conv1.W, 'b': self.conv1.b},
            'conv2': {'W': self.conv2.W, 'b': self.conv2.b},
            'fc1': {'W': self.fc1.W, 'b': self.fc1.b},
            'fc2': {'W': self.fc2.W, 'b': self.fc2.b}
        }
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)
