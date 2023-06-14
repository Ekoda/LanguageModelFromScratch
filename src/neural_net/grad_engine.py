import numpy as np

class ValueNode:
    def __init__(self, data: float, children: tuple = (), operation: str = ''):
        self.data = data
        self.gradient = 0
        self.previous = set(children)
        self.operation = operation
        self._backward = lambda: None
    
    def ensure_other_node(self, other):
        return other if isinstance(other, ValueNode) else ValueNode(other)
        
    def __add__(self, other):
        other = self.ensure_other_node(other)
        result_node = ValueNode(self.data + other.data, (self, other), '+')
        def _backward():
            self.gradient += result_node.gradient
            other.gradient += result_node.gradient
        result_node._backward = _backward
        return result_node

    def __mul__(self, other):
        other = self.ensure_other_node(other)
        result_node = ValueNode(self.data * other.data, (self, other), '*')
        def _backward():
            self.gradient += other.data * result_node.gradient
            other.gradient += self.data * result_node.gradient
        result_node._backward = _backward
        return result_node

    def __pow__(self, other):
        result_node = ValueNode(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.gradient += (other * self.data ** (other - 1)) * result_node.gradient
        result_node._backward = _backward
        return result_node

    def log(self):
        assert self.data > 0, "Logarithm not defined for zero or negative values."
        result_node = ValueNode(np.log(self.data), (self,), 'log')
        def _backward():
            self.gradient += (1 / self.data) * result_node.gradient
        result_node._backward = _backward
        return result_node

    def relu(self):
        result_node = ValueNode(0 if self.data < 0 else self.data, (self,), 'relu')
        def _backward():
            self.gradient += (result_node.data > 0) * result_node.gradient
        result_node._backward = _backward
        return result_node

    def sigmoid(self):
        sigmoid_value = 1 / (1 + np.exp(-self.data))
        result_node = ValueNode(sigmoid_value, (self,), 'sigmoid')
        def _backward():
            self.gradient += result_node.data * (1 - result_node.data) * result_node.gradient
        result_node._backward = _backward
        return result_node

    def backward(self):
        visited, ordered_nodes = set(), []
        def order_nodes(node):
            if node not in visited:
                visited.add(node)
                for child in node.previous:
                    order_nodes(child)
                ordered_nodes.append(node)
        order_nodes(self)
        self.gradient = 1
        for node in reversed(ordered_nodes):
            node._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __repr__(self):
        return f"ValueNode(data={self.data}, gradient={self.gradient})"
