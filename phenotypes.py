from __future__ import annotations

from typing import List, Set, Dict, Tuple, Optional, Any, Callable
from icontract import invariant, require, ensure
from enum import Enum

import math
from math import cos, sin, atan, ceil, floor
from queue import Queue

from scipy import special

import numpy as np

from neat.genes import NeuronGene, NeuronType
# import neat.genes as genes

class SNeuron:
    
    def __init__(self, neuronGene: NeuronGene) -> None:
        self.linksIn: List[SLink] = []

        self.activation = neuronGene.activation
        self.output: float = 1.0 if neuronGene.neuronType == NeuronType.BIAS else 0.0

        self.neuronType = neuronGene.neuronType

        self.ID: int = neuronGene.ID

        self.y: float = neuronGene.y
        self.x: float = neuronGene.x

    def activate(self, x: float) -> float:
        return self.activation(x)

    def __repr__(self):
        return "SNeuron(Type={0}, ID={1}, x={2}, y={3})".format(self.neuronType, self.ID, self.x, self.y)

class SLink:

    def __init__(self, fromNeuron: SNeuron, toNeuron: SNeuron, weight: float, recurrent: bool = False) -> None:
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron

        self.weight = weight

        self.recurrent = recurrent

# @invariant(lambda self: all(isinstance(x, SNeuron) for x in self.neurons), ValueError("Some neurons are not of type SNeuron."))
class Phenotype:

    def __init__(self, neurons: List[SNeuron], ID: int) -> None:
        self.neurons = neurons
        self.ID = ID

    # def sigmoid(self, x):
    #     return 1.0 / (1.0 + math.exp(-x))

    # def tanh(self, x: float) -> float:
    #     return np.tanh(x)

    # def relu(self, x: float) -> float:
    #     return np.maximum(x, 0)

    # def leakyRelu(self, x: float) -> float:
    #     return x if x > 0.0 else x * 0.01

    # def activation(self, x: float) -> float:
    #     # return self.leakyRelu(x)
    #     return self.tanh(x)

    def calcOutput(self, neuron: SNeuron) -> float:
        linksIn = neuron.linksIn
        if (len(linksIn) > 0):
            return neuron.activate(np.sum([self.calcOutput(linkIn.fromNeuron) * linkIn.weight for linkIn in linksIn]))
        else:
            return neuron.output

    def update(self, inputs: List[float]) -> List[float]:
        # return self.updateRecursively(inputs)
        return self.updateIteratively(inputs)

    def updateRecursively(self, inputs: List[float]) -> List[float]:
        inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
        for value, neuron in zip(inputs, inputNeurons):
            neuron.output = value

        outputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.OUTPUT]

        return [self.calcOutput(outputNeuron) for outputNeuron in outputNeurons]

    def updateIteratively(self, inputs: List[float]) -> List[float]:
        queue: Queue = Queue()

        # Set input neurons values
        inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
        for value, neuron in zip(inputs, inputNeurons):
            neuron.output = value

        for n in [neuron for neuron in self.neurons if neuron.neuronType in [NeuronType.INPUT, NeuronType.BIAS]]:
            queue.put(neuron)

        depths = sorted(set([n.y for n in self.neurons]))

        # for currentNeuron in self.neurons[len(inputNeurons):]:
        for depth in depths:
            neurons = [n for n in self.neurons if n.y == depth]

            for n in neurons:
                if len(n.linksIn) == 0:
                    n.output = 0.0
                else:
                    output = [link.fromNeuron.output * link.weight for link in n.linksIn]
                    n.output = n.activate(np.sum(output))
                
        # print(table)
        return [n.output for n in self.neurons if n.neuronType == NeuronType.OUTPUT]