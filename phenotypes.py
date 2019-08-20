from __future__ import annotations

from typing import List, Set, Dict, Tuple, Optional, Any, Callable
from enum import Enum

import math
from math import cos, sin, atan, ceil, floor

from scipy import special

import numpy as np

# from neat.genes import NeuronGene
import neat.genes as genes

class NeuronType(Enum):
    INPUT = 0
    HIDDEN = 1
    BIAS = 2
    OUTPUT = 3
    LINK = 4


class SNeuron:
    # def __init__(self, neuronType: NeuronType, neuronID: int, activation: Callable[[float], float],  y: float, x: float) -> None:
    def __init__(self, neuronGene: genes.NeuronGene) -> None:
        self.linksIn: List[SLink] = []

        self.activation = neuronGene.activation
        self.output = 1.0 if neuronGene.neuronType == NeuronType.BIAS else 0.0

        self.neuronType = neuronGene.neuronType

        self.ID = neuronGene.ID

        self.y = neuronGene.y
        self.x = neuronGene.x

    def activate(self, x: float) -> float:
        return self.activation(x)

class SLink:

    def __init__(self, fromNeuron: SNeuron, toNeuron: SNeuron, weight: float, recurrent: bool = False) -> None:
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron

        self.weight = weight

        self.recurrent = recurrent

class CNeuralNet:

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
        return self.updateRecursively(inputs)

    def updateRecursively(self, inputs: List[float]) -> List[float]:
        inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
        for value, neuron in zip(inputs, inputNeurons):
            neuron.output = value

        outputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.OUTPUT]

        return [self.calcOutput(outputNeuron) for outputNeuron in outputNeurons]

    def updateIteratively(self, inputs: List[float]) -> List[float]:
        # Set input neurons values
        inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
        for value, neuron in zip(inputs, inputNeurons):
            neuron.output = value

        inputNeurons += [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.BIAS]

        for currentNeuron in self.neurons[len(inputNeurons):]:
            linksIn = currentNeuron.linksIn

            if len(linksIn) == 0:
                currentNeuron.output = 0.0
            else:
                # output = np.sum(np.array([link.fromNeuron.output * link.weight for link in linksIn]))
                output = np.array([link.fromNeuron.output * link.weight for link in linksIn])
                # currentNeuron.output = self.activation(currentNeuron.bias + np.sum(output))
                # currentNeuron.output = self.activation(currentNeuron.bias + np.sum(output))
                currentNeuron.output = currentNeuron.activate(np.sum(output))
                
        # print(table)
        return [n.output for n in self.neurons if n.neuronType == NeuronType.OUTPUT]