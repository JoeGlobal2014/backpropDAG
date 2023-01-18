import micrograd.engine as engine
from nn import Module, Neuron, Layer, MLP

from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            
#class Layer(Module):
    def
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
  
 #class MLP(Module):


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
    return f"MLP of [{', '.join(str(layer) for layer in self. Layers)}]"


#In this example, I've moved the `Value` class and its related methods to a separate file called




