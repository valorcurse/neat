from __future__ import annotations

from typing import List, Set, Dict, Tuple, Optional, Any, Callable
from icontract import invariant, require, ensure
from enum import Enum

import math
from math import cos, sin, atan, ceil, floor
from queue import Queue

import numpy as np
from scipy import special
import numba
from numba import types
from numba import jit, njit, generated_jit, int64, float64, cuda

import networkx as nx

from neat.genes import NeuronGene, FuncsEnum
from neat.types import NeuronType

from timeit import default_timer as timer

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

    # def __init__(self, neurons: List[SNeuron], ID: int) -> None:
    #     self.neurons = neurons
    #     self.ID = ID
    #     self.graph = nx.DiGraph()

    #     # for n in self.neurons:
    #     self.graph.add_nodes_from([n.ID for n in self.neurons])
    #     self.graph.add_weighted_edges_from()

    def __init__(self, graph, ID: int) -> None:
        # self.neurons = neurons
        self.ID = ID
        self.graph = graph

        # for n in self.neurons:
        # self.graph.add_nodes_from([n.ID for n in self.neurons])
        # self.graph.add_weighted_edges_from()


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

    # def calcOutput(self, neuron: SNeuron) -> float:
    #     linksIn = neuron.linksIn
    #     if (len(linksIn) > 0):
    #         return neuron.activate(np.sum([self.calcOutput(linkIn.fromNeuron) * linkIn.weight for linkIn in linksIn]))
    #     else:
    #         return neuron.output

    # def update(self, inputs: List[float]) -> List[float]:
    #     # return self.updateRecursively(inputs)
    #     return self.updateIteratively(inputs)

    # def updateRecursively(self, inputs: List[float]) -> List[float]:
    #     inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
    #     for value, neuron in zip(inputs, inputNeurons):
    #         neuron.output = value

    #     outputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.OUTPUT]

    #     return [self.calcOutput(outputNeuron) for outputNeuron in outputNeurons]

    # def updateIteratively(self, inputs: List[float]) -> List[float]:
    #     queue: Queue = Queue()

    #     # Set input neurons values
    #     inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
    #     for value, neuron in zip(inputs, inputNeurons):
    #         neuron.output = value

    #     for n in [neuron for neuron in self.neurons if neuron.neuronType in [NeuronType.INPUT, NeuronType.BIAS]]:
    #         queue.put(neuron)

    #     depths = sorted(set([n.y for n in self.neurons]))

    #     # for currentNeuron in self.neurons[len(inputNeurons):]:
    #     for depth in depths:
    #         neurons = [n for n in self.neurons if n.y == depth]

    #         for n in neurons:
    #             if len(n.linksIn) == 0:
    #                 n.output = 0.0
    #             else:
    #                 output = [link.fromNeuron.output * link.weight for link in n.linksIn]
    #                 n.output = n.activate(np.sum(output))
                
    #     # print(table)
    #     return [n.output for n in self.neurons if n.neuronType == NeuronType.OUTPUT]


    def update(self, inputs: List[float]) -> List[float]:
        start_nodes = [n for n,d in self.graph.in_degree() if d == 0]

        adj = nx.adjacency_matrix(self.graph).todense()
        
        activations = np.array([FuncsEnum[self.graph.nodes()[n]['activation'].__name__].value for n in self.graph.nodes()])

        num_of_nodes = adj.shape[0]
        mem = np.full(num_of_nodes, np.nan)

        in_nodes = [n for n,d in self.graph.in_degree() if d == 0]
        for e in range(len(in_nodes)):
            mem[e] = inputs[e]
        
        out_nodes = [n for n,d in self.graph.out_degree() if d == 0]

        for i in range(len(out_nodes)):
            calcResults(num_of_nodes - i - 1, adj, activations, mem)
        
        outputs = [mem[n - i - 1] for n in range(len(out_nodes))]

        return outputs


@njit(float64(int64, float64[:, :], int64[:], float64[:]))
def calcResults(n, adj, acts, mem):
    preds = adj.T[n].nonzero()[0]
    
    if len(preds) == 0:
        return mem[n]

    inputs = np.array([mem[p] if not np.isnan(mem[p]) else calcResults(p, adj, acts, mem) for p in preds])
    
    weights = adj[preds][:, n]

    result = np.sum(weights * inputs)

    function = acts[n]
    if function == FuncsEnum.tanh.value:
        mem[n] = math.tanh(result)
    elif function == FuncsEnum.sin.value:
        mem[n] = math.sin(result)
    elif function == FuncsEnum.cos.value:
        mem[n] = math.cos(result)

    return mem[n]