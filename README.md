# backpropDAG



Show me how to code in python step by step: "Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes."
I can certainly guide you through the process of implementing backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API.
First, we need to import the necessary libraries. We will be using numpy for array computations and matplotlib for plotting the results.
import numpy as np
import matplotlib.pyplot as plt


Next, we will define the DAG (Directed Acyclic Graph) class. This class will have methods for adding nodes and edges, as well as for performing forward and backward computations. The forward computation will compute the output of the graph given an input, while the backward computation will compute the gradients of the graph with respect to its inputs.
class DAG:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def forward(self, input):
        for node in self.nodes:
            node.forward()
        return self.nodes[-1].output

    def backward(self, grad_output):
        for node in reversed(self.nodes):
            node.backward(grad_output)


Now, we will define the node class which will hold the computation of each node and its gradient.
class Node:
    def __init__(self, forward_fn, backward_fn):
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn
        self.inputs = []
        self.output = None
        self.grad_output = None
        self.grad_inputs = []

    def forward(self):
        inputs = [input.output for input in self.inputs]
        self.output = self.forward_fn(*inputs)

    def backward(self, grad_output):
        self.grad_output = grad_output
        grad_inputs = self.backward_fn(grad_output)
        for i, input in enumerate(self.inputs):
            input.grad_output += grad_inputs[i]


Next, we will define the small neural network library on top of the DAG. This library will have classes for building layers such as Linear and Sigmoid and also some helper functions for training and evaluating the model.
class Linear(Node):
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)

        def forward_fn(x):
            self.x = x
            return np.dot(x, self.W) + self.b

        def backward_fn(grad_output):
            grad_input = np.dot(grad_output, self.W.T)
            grad_W = np.dot(self.x.T, grad_output)
            grad_b = np.sum(grad_output, axis

xis


Add a class called value with methods for arithmetic operations


Sure, we can add a class called Value with methods for arithmetic operations such as addition, subtraction, multiplication, and division. This class will inherit from the Node class, and the forward and backward functions will be defined for each arithmetic operation. Here's an example of how the Value class could be implemented:
Note that we have defined the __add__, __sub__, __mul__, and __truediv__ methods to overload the corresponding arithmetic operators. These methods create a new Node object with the appropriate forward and backward functions for each arithmetic operation and set its inputs to be the Value objects on which the operator is being applied.
class Value(Node):
    def __init__(self, value):
        self.value = value
        self.output = value
        self.grad_output = 0

    def forward(self):
        pass

    def backward(self, grad_output):
        self.grad_output = grad_output

    def __add__(self, other):
        def forward_fn(x, y):
            return x + y

        def backward_fn(grad_output):
            return grad_output, grad_output

        node = Node(forward_fn, backward_fn)
        node.inputs = [self, other]
        return node

    def __sub__(self, other):
        def forward_fn(x, y):
            return x - y

        def backward_fn(grad_output):
            return grad_output, -grad_output

        node = Node(forward_fn, backward_fn)
        node.inputs = [self, other]
        return node

    def __mul__(self, other):
        def forward_fn(x, y):
            return x * y

        def backward_fn(grad_output):
            return grad_output * other.output, grad_output * self.output

        node = Node(forward_fn, backward_fn)
        node.inputs = [self, other]
        return node

    def __truediv__(self, other):
        def forward_fn(x, y):
            return x / y

        def backward_fn(grad_output):
            return grad_output / other.output, -grad_output * self.output / (other.output ** 2)

        node = Node(forward_fn, backward_fn)
        node.inputs = [self, other]
        return node




Add a method for the power function

Sure, we can add a method for the power function to the Value class. Here's an example of how it could be implemented:
class Value(Node):
    ...
    def __pow__(self, other):
        def forward_fn(x, y):
            return x ** y

        def backward_fn(grad_output):
            return grad_output * other.output * (self.output ** (other.output - 1)), grad_output * np.log(self.output) * (self.output ** other.output)
        node = Node(forward_fn, backward_fn)
        node.inputs = [self, other]
        return node

node
In this implementation, we have defined the __pow__ method to overload the ** operator. This method creates a new Node object with the appropriate forward and backward functions for the power operation, and set its inputs to be the Value objects on which the operator is being applied.
The forward function calculates the power between the two input values, while the backward function calculates the gradient with respect to the inputs.
You can also use other mathematical functions in the same manner.


Add a method for relu

Sure, we can add a method for the ReLU (Rectified Linear Unit) activation function to the Value class. Here's an example of how it could be implemented:
class Value(Node):
    ...
    def relu(self):
        def forward_fn(x):
            return np.maximum(x, 0)

        def backward_fn(grad_output):
            return grad_output * (self.output > 0)

        node = Node(forward_fn, backward_fn)
        node.inputs = [self]
        return node



In this implementation, the relu method creates a new Node object with the appropriate forward and backward functions for the ReLU activation function, and set its input to be the Value object on which the method is being applied.
The forward function applies the ReLU activation function on the input value, while the backward function calculates the gradient with respect to the input by multiplying the gradients with the derivative of the ReLU function which is 1 for positive values and 0 for negative values.
It's also worth mentioning that some deep learning frameworks like Pytorch, Tensorflow and others, already have built-in implementations for activation functions such as ReLU, so you don't need to implement it by yourself.


Add a reverse method for all the methods we have implemented
Sure, we can add a reverse method for all the methods we have implemented in the Value class. The reverse method will simply swap the inputs of the operation and negate the gradients. Here's an example of how it could be implemented:
class Value(Node):
    ...
    def __add__(self, other):
        ...
        return node

    def __sub__(self, other):
        ...
        return node

    def __mul__(self, other):
        ...
        return node

    def __truediv__(self, other):
        ...
        return node
    
    def __pow__(self, other):
        ...
        return node
    
    def relu(self):
        ...
        return node
    
    def reverse(self):
        self.grad_output = -self.grad_output
        self.inputs[0], self.inputs[1] = self.inputs[1], self.inputs[0]


As you can see in the reverse method, we simply negate the gradients by multiplying it with -1, and swap the inputs of the operation. This way, the backward computation will compute the gradients with respect to the reversed operation.
Keep in mind, this reverse method only works for the binary operations such as addition, subtraction, multiplication, division, power and relu. For unary operations or other types of operation you may have to implement the reverse method accordingly.


Recode the classes and methods so the work with a file called nn.py which contains the following code:"import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self. Layers)}]" "
The code you provided is creating a neural network library using classes and modules defined in the micrograd library. The Value class from the micrograd library is used to define the weights and biases of the neurons, and the Module class is used as a base class for other classes such as Neuron, Layer, and MLP.
The Neuron class represents a single neuron in the neural network and it is initialized with the number of inputs and an optional non-linearity flag. The __call__ method is used to calculate the output of the neuron given an input and the parameters method returns a list of the neuron's weights and biases.
The Layer class represents a layer of neurons in the neural network and it is initialized with the number of inputs and outputs, and any additional keyword arguments. The __call__ method is used to calculate the outputs of the layer given an input and the parameters method returns a list of all the weights and biases of the neurons in the layer.
The MLP class represents a multi-layer perceptron and it is initialized with the number of inputs and a list of the number of outputs for each layer. The __call__ method is used to calculate the output of the network given an input and the parameters method returns a list of all the weights and biases of all the neurons in the network.
So to make this code work with the nn.py file, you need to import the micrograd library, and the use the classes and methods provided in the file.
import micrograd.engine as engine
from nn import Module, Neuron, Layer, MLP
You can then create instances of these classes and use them to build your neural network. For example, you can create a 2-layer perceptron with 2 inputs and 2 outputs as follows:
mlp = MLP(2, [2])


and then you can use the call method to forward the inputs through the network
output = mlp([1,2])


You can also use the zero_grad() method to set the gradients of all the parameters to 0 before running the backward pass, and the parameters() method to retrieve the list of all the parameters of the network, which you can use to update the parameters of the network using optimizer.


Modify the code in the class value for the method backward using this example:" 


def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward() "
Sure, we can modify the backward method in the Value class to use the topological sorting approach you provided. Here's an example of how it could be implemented:
class Value(Node):
    ...
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

This backward method uses a depth-first search algorithm to build a topological ordering of the graph. This is necessary because the gradients must be computed in the correct order to ensure that the gradients of the parents are correctly computed before the gradients of the children.
It starts by initializing the gradients of the output node to 1, and then iterates through the topological ordering of the graph, calling the _backward method on each node to compute its gradient.
In the above example, _prev and _backward are assumed to be properties of


Continue your response
the Value class that contain the previous nodes in the graph and the backward computation method respectively. You will need to make sure that these properties are correctly set and implemented for the Value class for this method to work correctly. Also, this approach assumes that the graph is well formed and acyclic, otherwise the algorithm will get stuck in an infinite loop.


What do you mean when you say " In the above example, _prev and _backward are assumed to be properties of "
In the example I provided, I assumed that the Value class has two properties, _prev and _backward, that are used in the implementation of the backward method. _prev is assumed to be a property that contains a list of the previous nodes in the computation graph, and _backward is assumed to be a method that performs the backward computation for the node.
In the example _prev is used in the topological sorting algorithm to traverse the graph, and _backward is used to compute the gradients for each node.
It's important to note that these properties are not part of the standard implementation of the Value class and it's just an assumption I made to provide an example of how the backward method could be implemented, in order to make it work, the implementation of the Value class should include those properties.


rewrite all the code into two files one called engine.py and the other file called nn.py
Sure, here's an example of how the code could be split into two files, engine.py and nn.py:
engine.py
import random

class Value:
    def __init__(self, value, grad=0):
        self.value = value
        self.grad = grad
        self.prev = []

    def __add__(self, other):
        return Value(self.value + other.value, 0)

    def __sub__(self, other):
        return Value(self.value - other.value, 0)

    def __mul__(self, other):
        return Value(self.value * other.value, 0)

    def __truediv__(self, other):
        return Value(self.value / other.value, 0)
    
    def __pow__(self, other):
        return Value(self.value ** other.value, 0)

    def relu(self):
        return Value(max(0, self.value), 0)
    
    def backward(self):
        self.grad = 1
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for v in reversed(topo):
            v._backward()
            
    def _backward(self):
        pass

nn.py
from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def






Finish writing the code for nn.py
Sure, here's how the rest of the code for nn.py could look like:
from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
]

