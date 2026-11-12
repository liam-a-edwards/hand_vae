import numpy as np


def unbroadcast(grad, shape):
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


# Adapted for numpy arrays from micrograd (https://github.com/karpathy/micrograd)
class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Shape={self.data.shape}"

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        stack = [(self, False)]
        while stack:
            v, processed = stack.pop()
            if processed:
                topo.append(v)
            elif v not in visited:
                visited.add(v)
                stack.append((v, True))
                for child in v._prev:
                    stack.append((child, False))

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __radd__(self, other):  # other + self
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __sub__(self, other):  # self - other
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), "-")

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad -= unbroadcast(out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __rsub__(self, other):  # other - self
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(other.data - self.data, (self, other), "-")

        def _backward():
            self.grad -= unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __truediv__(self, other):  # self / other
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), "/")

        def _backward():
            self.grad += unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += unbroadcast(
                -self.data / (other.data) ** 2 * out.grad, other.data.shape
            )

        out._backward = _backward

        return out

    def __rtruediv__(self, other):  # other / self
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(other.data / self.data, (self, other), "/")

        def _backward():
            other.grad += unbroadcast(out.grad / self.data, other.data.shape)
            self.grad += unbroadcast(
                -other.data / (self.data) ** 2 * out.grad, self.data.shape
            )

        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += unbroadcast(out.grad @ other.data.T, self.data.shape)
            other.grad += unbroadcast(self.data.T @ out.grad, other.data.shape)

        out._backward = _backward

        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad += unbroadcast(
                out.grad * np.ones_like(self.data), self.data.shape
            )

        out._backward = _backward

        return out

    def __neg__(self):  # -self
        out = Tensor(-self.data, (self,), "neg")

        def _backward():
            self.grad -= out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        out = Tensor(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward

        return out

    def cos(self):
        out = Tensor(np.cos(self.data), (self,), "cos")

        def _backward():
            self.grad -= np.sin(self.data) * out.grad

        out._backward = _backward

        return out

    def sin(self):
        out = Tensor(np.sin(self.data), (self,), "sin")

        def _backward():
            self.grad += np.cos(self.data) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += 1 / self.data * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        e = np.exp(-np.abs(self.data))  # Prevent overflow
        s = np.where(self.data >= 0, 1 / (1 + e), e / (1 + e))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad

        out._backward = _backward

        return out
