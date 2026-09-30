import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import urllib.request                 # urllib -> to download MNIST data
import gzip                           #gzip -> to unzip the MNIST data
import struct                         #struct ->help me to read binary data


#Unbroadcast function is basically to convert the shape of the gradient to the shape of the original tensor after broadcasting.
def unbroadcast(gradient, original_shape):
    while(len(gradient.shape)>len(original_shape)):
        gradient = gradient.sum(axis=0)
    for axis,size in enumerate(original_shape):
        if size==1 and gradient.shape[axis]!=1:
            gradient = gradient.sum(axis = axis, keepdims=True)
    return gradient

class Tensor:
    def __init__(sel,data,required_grad = False,_children = (),_op = ''):
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
                        required_grad = self.required_grad orother.required_grad,
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
        other = other if isinstance(Self, Tensor) else Tensor(other)
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
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"pow({exponent})"
        )
        def _backward():

            if self.requires_grad:
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
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="@"
        )
        def _backward():
            if self.requires_grad:

                # d(A@B)/dA = output_gradient @ B.T
                self.grad += output.grad @ other.data.T
                
            if other.requires_grad:

                # d(A@B)/dB = A.T @ output_gradient
                other.grad += self.data.T @ output.grad
                
        output._backward = _backward
        return output
    __matmul__ = matmul


    # Define sum operation.
    def sum(self, axis=None, keepdims=False):
        output = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum"
        )
        def _backward():
            if self.requires_grad:

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
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp"
        )
        def _backward():
            # The derivative of exp(x) is exp(x).
            if self.requires_grad:
                self.grad += unbroadcast(output.grad * output.data, self.data.shape)
        output._backward = _backward
        return output


    # Define natural logarithm.
    def log(self):
        output = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log"
        )

        # Define backward operation.
        def _backward():
            # Derivative of log(x) is 1/x.
            if self.requires_grad:
                self.grad += unbroadcast(output.grad / self.data, self.data.shape)

        output._backward = _backward
        return output

    # Define reshape.
    def reshape(self, *shape):
        output = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape"
        )
        def _backward():
            if self.requires_grad:
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



    
    
    
    
    
         