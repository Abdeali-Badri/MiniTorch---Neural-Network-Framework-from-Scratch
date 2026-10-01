import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import urllib.request                 # urllib -> to download MNIST data
import gzip                           # gzip -> to unzip the MNIST data
import struct                         # struct ->help me to read binary data


# Unbroadcast function is basically to convert the shape of the gradient to the shape of the original tensor after broadcasting.
def unbroadcast(gradient, original_shape):
    while(len(gradient.shape)>len(original_shape)):
        gradient = gradient.sum(axis=0)
    for axis,size in enumerate(original_shape):
        if size==1 and gradient.shape[axis]!=1:
            gradient = gradient.sum(axis = axis, keepdims=True)
    return gradient

class Tensor:
    def __init__(self,data,required_grad = False,_children = (),_op = ''):
         # convert input data to numpy array
         self.data = np.asarray(data, dtype=np.float32)
         self.required_grad = required_grad
         # to store gradient of the tensor
         self.grad = np.zeros_like(self.data) if self.required_grad else None
         self._prev = set(_children)
         self._op = _op
         # store backward function function to this tensor
         self._backward = lambda: None
    
    # Return a readable representation of the Tensor.
    def __repr__(self):
        return f"Tensor(Data = {self.data}, required_grad = {self.grad})"
    
    # Addition operation between two tensors.
    def __add__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data+other.data,
                        required_grad = self.required_grad or other.required_grad,
                        _children = (self,other),
                        _op = "+")
        
        def _backward():
            if self.required_grad:
                self.grad += unbroadcast(output.grad,self.data.shape)
            if other.required_grad:
                other.grad += unbroadcast(output.grad,other.data.shape)
        
        output._backward = _backward
        return output
    __radd__ = __add__
    
    # Define negative tensor
    def __neg__(self):
        return self * -1
    
    # Subtraction between two vectors:
    
    def __sub__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)
    
    def __rsub__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    def __mul__(self,other):
        other = other if isinstance(self, Tensor) else Tensor(other)
        output = Tensor(self.data * other.data,
                        required_grad = self.required_grad or other.required_grad,
                        _children = (self,other),
                        _op = "*")
        
        def _backward():
            if self.required_grad:
                self.grad+= unbroadcast(output.grad * other.data, self.data.shape)
            if other.required_grad:
                other.grad += unbroadcast(output.grad * self.data, other.data.shape)
        
        output._backward = _backward
        return output
    __rmul__ = __mul__
    
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Division is multiplication by inverse.
        return self * other.pow(-1)

    def __rtruediv__(self, other):

        
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Return other divided by self.
        return other / self

  
    def pow(self, exponent):
        output = Tensor(
            self.data ** exponent,
            required_grad=self.required_grad,
            _children=(self,),
            _op=f"pow({exponent})"
        )
        def _backward():

            if self.required_grad:
                # Derivative of x^n is n*x^(n-1).
                self.grad += unbroadcast(
                    output.grad
                    * exponent
                    * self.data ** (exponent - 1), self.data.shape)
        output._backward = _backward
        return output
    
    
    # Define matrix multiplication
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(
            self.data @ other.data,
            required_grad=self.required_grad or other.required_grad,
            _children=(self, other),
            _op="@"
        )
        def _backward():
            if self.required_grad:

                # d(A@B)/dA = output_gradient @ B.T
                self.grad += output.grad @ other.data.T
                
            if other.required_grad:

                # d(A@B)/dB = A.T @ output_gradient
                other.grad += self.data.T @ output.grad
                
        output._backward = _backward
        return output
    __matmul__ = matmul


    # Define sum operation.
    def sum(self, axis=None, keepdims=False):
        output = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            required_grad=self.required_grad,
            _children=(self,),
            _op="sum"
        )
        def _backward():
            if self.required_grad:

                # Start with the output gradient.
                gradient = output.grad

                # If an axis was reduced, restore that dimension.
                if axis is not None and not keepdims:

                    # Convert axis into a tuple if necessary.
                    axes = axis if isinstance(axis, tuple) else (axis,)

                    # Insert dimensions in sorted order.
                    for ax in sorted([a if a >= 0 else a + self.data.ndim
                                      for a in axes]):
                        gradient = np.expand_dims(gradient, ax)
                self.grad += np.ones_like(self.data) * gradient
        output._backward = _backward
        return output


    # Define mean operation.
    def mean(self, axis=None, keepdims=False):
        if axis is None:
            divisor = self.data.size

        else:
            # Calculate the number of elements being reduced.
            axes = axis if isinstance(axis, tuple) else (axis,)

            # Calculate the divisor.
            divisor = np.prod([self.data.shape[a] for a in axes])

        return self.sum(axis=axis, keepdims=keepdims) / divisor



    # Define exponential function.
    def exp(self):
        output = Tensor(
            np.exp(self.data),
            required_grad=self.required_grad,
            _children=(self,),
            _op="exp"
        )
        def _backward():
            # The derivative of exp(x) is exp(x).
            if self.required_grad:
                self.grad += unbroadcast(output.grad * output.data, self.data.shape)
        output._backward = _backward
        return output


    # Define natural logarithm.
    def log(self):
        output = Tensor(
            np.log(self.data),
            required_grad=self.required_grad,
            _children=(self,),
            _op="log"
        )

        # Define backward operation.
        def _backward():
            # Derivative of log(x) is 1/x.
            if self.required_grad:
                self.grad += unbroadcast(output.grad / self.data, self.data.shape)

        output._backward = _backward
        return output

    # Define reshape.
    def reshape(self, *shape):
        output = Tensor(
            self.data.reshape(*shape),
            required_grad=self.required_grad,
            _children=(self,),
            _op="reshape"
        )
        def _backward():
            if self.required_grad:
                self.grad += unbroadcast(output.grad.reshape(self.data.shape), self.data.shape)
        output._backward = _backward
        return output


    def backward(self):

        # Make a list for the topological ordering.
        topology = []

        # Keep track of visited tensors.
        visited = set()
        def build_topology(tensor):
            if tensor not in visited:
                visited.add(tensor)

                # Visit all parent tensors first.
                for child in tensor._prev:
                    build_topology(child)

                # Add the tensor after its parents.
                topology.append(tensor)
        build_topology(self)
        # Set the derivative of the final output to 1.
        self.grad = np.ones_like(self.data)

        # Traverse the graph in reverse order.
        for tensor in reversed(topology):
            tensor._backward()


def relu(x):
    # Create mask showing where values are positive.
    mask = (x.data > 0).astype(np.float64)
    # Multiply by mask to implement max(0,x).
    output = Tensor(
        np.maximum(0, x.data),
        required_grad=x.required_grad,
        _children=(x,),
        _op="relu"
    )
    # Define ReLU backward pass.
    def _backward():
        if x.required_grad:
            # Gradient is 1 for positive values and 0 otherwise.
            x.grad += output.grad * mask
    output._backward = _backward
    return output
    
def sigmoid(x):
    
    # Clip values to prevent exponential overflow.
    clipped = np.clip(x.data, -50, 50)
    values = 1.0 / (1.0 + np.exp(-clipped))
    output = Tensor(
        values,
        required_grad=x.required_grad,
        _children=(x,),
        _op="sigmoid"
    )
    def _backward():
        if x.required_grad:
            # Derivative of sigmoid is sigmoid*(1-sigmoid).
            x.grad += output.grad * output.data * (1 - output.data)
    output._backward = _backward
    return output


def softmax(x, axis=1):

    # Subtract maximum for numerical stability.
    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)

    # Calculate exponentials.
    exp_values = np.exp(shifted)

    # Calculate denominator.
    denominator = np.sum(exp_values, axis=axis, keepdims=True)

    # Calculate probabilities.
    probabilities = exp_values / denominator
    output = Tensor(
        probabilities,
        required_grad=x.required_grad,
        _children=(x,),
        _op="softmax"
    )
    def _backward():
        if x.required_grad:
            # Softmax Jacobian-vector product can be simplified.
            dot = np.sum(
                output.grad * output.data,
                axis=axis,
                keepdims=True
            )
            x.grad += output.data * (output.grad - dot)
    output._backward = _backward
    return output

# Loss Functions:
def mse_loss(prediction, target):
    target = target if isinstance(target, Tensor) else Tensor(target)
    difference = prediction - target
    squared = difference * difference
    return squared.mean()


def cross_entropy(logits, labels):

    # Get logits as a NumPy array.
    logits_data = logits.data
    # Subtract maximum for numerical stability.
    shifted = logits_data - np.max(
        logits_data,
        axis=1,
        keepdims=True
    )

    # Calculate exponentials.
    exp_values = np.exp(shifted)
    exp_sum = np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )
    log_sum_exp = np.log(exp_sum)

    log_probabilities = shifted - log_sum_exp

    # Get the number of training examples.
    batch_size = logits_data.shape[0]

    # Select the correct class probability for each sample.
    correct_log_probabilities = log_probabilities[
        np.arange(batch_size),
        labels
    ]
    # Calculate average negative log likelihood.
    loss_value = -np.mean(correct_log_probabilities)
    output = Tensor(
        loss_value,
        required_grad=logits.required_grad,
        _children=(logits,),
        _op="cross_entropy"
    )
    def _backward():
        if logits.required_grad:
            gradient = exp_values / exp_sum

            # Subtract 1 from the correct class.
            gradient[
                np.arange(batch_size),
                labels
            ] -= 1
            # Divide by batch size because loss uses mean.
            gradient /= batch_size

            # Multiply by output gradient.
            logits.grad += output.grad * gradient
    output._backward = _backward
    return output

# Define a fully connected neural network layer.
class Linear:
    def __init__(self, input_features, output_features):
        # Use He initialization for weights.
        weight_scale = np.sqrt(2.0 / input_features)
        # Create random weights.
        self.weight = Tensor(
            np.random.randn(
                input_features,
                output_features
            ) * weight_scale,
            requires_grad=True
        )

        # Create bias initialized to zero.
        self.bias = Tensor(
            np.zeros(output_features),
            requires_grad=True
        )
    def __call__(self, x):
        output = x @ self.weight
        output += self.bias
        return output
    def parameters(self):
        return [self.weight, self.bias]

# Define a small multilayer perceptron.
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = Linear(input_size, hidden_size)
        self.layer2 = Linear(hidden_size, output_size)
    def __call__(self, x):
        x = self.layer1(x)
        x = relu(x)
        x = self.layer2(x)
        return x
    def parameters(self):
        return (
            self.layer1.parameters()
            + self.layer2.parameters()
        )


class SGD:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate
    def step(self):
        for parameter in self.parameters:
            parameter.data -= (
                self.learning_rate * parameter.grad
            )
    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad.fill(0)


# Create XOR input data.
XOR_X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float64)

# Create XOR labels.
XOR_Y = np.array([
    0,
    1,
    1,
    0
])

def train_xor():
    np.random.seed(42)
    model = MLP(
        input_size=2,
        hidden_size=8,
        output_size=2
    )
    optimizer = SGD(
        model.parameters(),
        learning_rate=0.1
    )
    for epoch in range(3000):
        # Create Tensor containing XOR input.
        x = Tensor(XOR_X)
        logits = model(x)
        loss = cross_entropy(logits, XOR_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(
                "Epoch:",
                epoch,
                "Loss:",
                round(loss.data.item(), 6)
            )
    predictions = model(Tensor(XOR_X))
    predicted_classes = np.argmax(
        predictions.data,
        axis=1
    )
    print("XOR predictions:")

    for input_value, prediction in zip(
        XOR_X,
        predicted_classes
    ):
        print(
            input_value,
            "->",
            prediction
        )
    return model


# MNIST dataset
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
def download_mnist_file(filename):
    url = MNIST_URL + filename
    path = filename
    if not os.path.exists(path):
        print("Downloading:", filename)
        urllib.request.urlretrieve(
            url,
            path
        )
    return path

def load_mnist_images(filename):
    with gzip.open(filename, "rb") as file:
        magic, number, rows, columns = struct.unpack(
            ">IIII",
            file.read(16)
        )
        # Read all pixel data.
        data = np.frombuffer(
            file.read(),
            dtype=np.uint8
        )
        # Reshape into images.
        images = data.reshape(
            number,
            rows * columns
        )
        # Convert pixels from 0-255 to 0-1.
        images = images.astype(np.float64) / 255.0
    return images

def load_mnist_labels(filename):
    with gzip.open(filename, "rb") as file:

        # Read header.
        magic, number = struct.unpack(
            ">II",
            file.read(8)
        )
        # Read label values.
        labels = np.frombuffer(
            file.read(),
            dtype=np.uint8
        )
    return labels

def load_mnist(limit=2000):
    train_images_file = download_mnist_file(
        "train-images-idx3-ubyte.gz"
    )
    train_labels_file = download_mnist_file(
        "train-labels-idx1-ubyte.gz"
    )
    images = load_mnist_images(
        train_images_file
    )

    labels = load_mnist_labels(
        train_labels_file
    )
    images = images[:limit]
    labels = labels[:limit]
    return images, labels

def train_mnist():

    X, y = load_mnist(limit=2000)
    model = MLP(
        input_size=784,
        hidden_size=64,
        output_size=10
    )
    optimizer = SGD(
        model.parameters(),
        learning_rate=0.05
    )
    epochs = 5
    batch_size = 32
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        total_loss = 0.0
        number_of_batches = 0
        
        for start in range(
            0,
            len(X),
            batch_size
        ):
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]
            x_tensor = Tensor(X_batch)
            logits = model(x_tensor)
            loss = cross_entropy(
                logits,
                y_batch
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            number_of_batches += 1
        average_loss = (
            total_loss / number_of_batches
        )
        predictions = model(
            Tensor(X)
        )


        predicted_classes = np.argmax(
            predictions.data,
            axis=1
        )
        accuracy = np.mean(
            predicted_classes == y
        )
        print(
            "Epoch:",
            epoch + 1,
            "| Loss:",
            round(average_loss, 4),
            "| Accuracy:",
            round(accuracy * 100, 2),
            "%"
        )
    return model

# Define a small example to demonstrate backpropagation.
def autodiff_demo():
    x = Tensor(
        [2.0, 3.0],
        requires_grad=True
    )
    w = Tensor(
        [4.0, 5.0],
        requires_grad=True
    )
    multiplication = x * w
    result = multiplication.sum()
    result.backward()
    print("x =", x.data)
    print("w =", w.data)
    print("x * w =", multiplication.data)
    print("result =", result.data)
    print("Gradient of x =", x.grad)
    print("Gradient of w =", w.grad)



def compare_with_pytorch():
    try:
        import torch
        import torch.nn as nn
    except ImportError:

        # Inform the user.
        print(
            "PyTorch is not installed."
        )
        return
    np.random.seed(123)
    input_data = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    our_model = Linear(
        2,
        2
    )
    torch_model = nn.Linear(
        2,
        2
    )
    torch_model.weight.data = torch.tensor(
        our_model.weight.data.T,
        dtype=torch.float64
    )
    torch_model.bias.data = torch.tensor(
        our_model.bias.data,
        dtype=torch.float64
    )
    our_input = Tensor(
        input_data,
        requires_grad=True
    )
    our_output = our_model(
        our_input
    )
    torch_input = torch.tensor(
        input_data,
        dtype=torch.float64,
        requires_grad=True
    )
    torch_output = torch_model(
        torch_input
    )
    print(
        "Forward outputs match:",
        np.allclose(
            our_output.data,
            torch_output.detach().numpy(),
            atol=1e-8
        )
    )
    our_loss = (
        our_output * our_output
    ).mean()
    our_loss.backward()
    torch_loss = (
        torch_output * torch_output
    ).mean()
    torch_loss.backward()
    print(
        "Input gradients match:",
        np.allclose(
            our_input.grad,
            torch_input.grad.numpy(),
            atol=1e-8
        )
    )
    print(
        "Weight gradients match:",
        np.allclose(
            our_model.weight.grad.T,
            torch_model.weight.grad.numpy(),
            atol=1e-8
        )
    )
    print(
        "Bias gradients match:",
        np.allclose(
            our_model.bias.grad,
            torch_model.bias.grad.numpy(),
            atol=1e-8
        )
    )    