import numpy as np
from collections import deque, defaultdict
import graph.api as graph_fn
import autograd.grads as gpi

class Node(np.ndarray):
    def __new__(
        subtype, shape, dtype=float, buffer=None, offset=0, strides=None,
        order=None
    ):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset,
                                  strides, order)
        obj.grad = None
        return obj 

    def _create_node(self, method, node, name, self_first=True):
        if not isinstance(node, Node):
            node = Constant.create(node)
        val = getattr(np.ndarray, method)(self, node)
        return Operation.create(name, val, self if self_first else node,
                                node if self_first else self)

    def __add__(self, node):
        return self._create_node('__add__', node, 'add')

    def __radd__(self, node):
        return self._create_node('__radd__', node, 'add', False)

    def __sub__(self, node):
        return self._create_node('__sub__', node, 'sub')

    def __rsub__(self, node):
        return self._create_node('__rsub__', node, 'sub', False)

    def __mul__(self, node):
        return self._create_node('__mul__', node, 'mul')

    def __rmul__(self, node):
        return self._create_node('__rmul__', node, 'mul', False)

    def __div__(self, node):
        return self._create_node('__div__', node, 'div')

    def __rdiv_(self, node):
        return self._create_node('__rdiv__', node, 'div', False)

    def __truediv__(self, node):
        return self._create_node('__truediv__', node, 'div')

    def __rtruediv__(self, node):
        return self._create_node('__rtruediv__', node, 'div', False)

    def __pow__(self, node):
        return self._create_node('__pow__', node, 'pow')

    def __rpow__(self, node):
        return self._create_node('__rpow__', node, 'pow', False)

    @property
    def T(self):
        val = np.transpose(self)
        return Operation.create('transpose', val, self)
    
    def correct_grad(self, array: np.ndarray):
        if self.shape == array.shape:
            return array
        
        my_shape = list(self.shape)
        ar_shape = list(array.shape)

        if self.ndim != array.ndim:
            my_shape = [-1] * np.abs(self.shape - array.shape) + self.shape
        
        sum_axes = []
        squeeze_axes = []

        for i, (dim2, dim1) in enumerate(zip(ar_shape, my_shape)):
            if dim2 != dim1:
                sum_axes.append(i)
                if dim1 == -1:
                    squeeze_axes.append(i)
        
        narray = graph_fn.sum(array, axis=tuple(sum_axes), keepdims=True)
        return graph_fn.squeeze(narray, axis=tuple(squeeze_axes))

    def backward(self):
        frontier = NodeQueue()
        future_grads = defaultdict(int)
        future_grads[self.name] = Constant.create(np.ones(self.shape))
        frontier.push(self)

        grads = {}
        
        while len(frontier):
            vertex = frontier.pop()

            # Calculate gradient only from operation node.
            if isinstance(vertex, Variable):
                vertex.grad = future_grads[vertex.name]
                grads[vertex.name] = future_grads[vertex.name]
                continue
            elif isinstance(vertex, Constant):
                vertex.grad = graph_fn.constant(0)
                continue
            
            adj = future_grads[vertex.name]
            op_name = vertex.op_name

            op_grad = getattr(gpi, 'grad_{}'.format(op_name))
            grad = op_grad(vertex, adj)    # Get grad of operation

            future_grads[vertex.op1.name] = vertex.op1.correct_grad(future_grads[vertex.op1.name] + grad[0])
            if vertex.op1 not in frontier:
                frontier.push(vertex.op1)
            
            if vertex.op2:
                future_grads[vertex.op2.name] = vertex.op2.correct_grad(future_grads[vertex.op2.name] + grad[1])
                if vertex.op2 not in frontier:
                    frontier.push(vertex.op2)

        self.grad = grads
        return grads


class Operation(Node):
    unknown_nodes = {}

    @staticmethod
    def create(op_name, op_result, op1, op2=None, name=None):
        obj = Operation(strides=op_result.strides, shape=op_result.shape,
                     dtype=op_result.dtype, buffer=np.copy(op_result))
        obj.op_name = op_name
        obj.op1 = op1
        obj.op2 = op2

        if name is not None:
            obj.name = name
        else:
            if op_name not in Operation.unknown_nodes:
                Operation.unknown_nodes[op_name] = 0
            nodeID = Operation.unknown_nodes[op_name]
            Operation.unknown_nodes[op_name] += 1
            obj.name = op_name + '_' + str(nodeID)

        return obj


class Constant(Node):
    unknown_count = 0

    @staticmethod
    def create(val, name=None):
        if not isinstance(val, np.ndarray):
            val = np.array(val, dtype=float)
        obj = Constant(strides=val.strides, shape=val.shape, dtype=val.dtype,
                       buffer=np.copy(val))
        if name is None:
            obj.name = "const_" + str(Constant.unknown_count)
            Constant.unknown_count += 1
        else:
            obj.name = name

        return obj

class Variable(Node):
    unknown_count = 0

    @staticmethod
    def create(val, name=None):
        if not isinstance(val, np.ndarray):
            val = np.array(val, dtype=float)
        obj = Variable(strides=val.strides, shape=val.shape, dtype=val.dtype,
                       buffer=np.copy(val))
        if name is None:
            obj.name = "var_" + str(Constant.unknown_count)
            Constant.unknown_count += 1
        else:
            obj.name = name

        return obj

class NodeQueue:
    def __init__(self):
        self.nodes = deque()
        self.node_names = deque()

    def push(self, node):
        self.nodes.append(node)
        self.node_names.append(node.name)

    def pop(self):
        self.node_names.popleft()
        return self.nodes.popleft()

    def __contains__(self, node):
        return node.name in self.node_names

    def __len__(self):
        return len(self.node_names)
