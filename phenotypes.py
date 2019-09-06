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
from numba import jit, njit, generated_jit, int64, float64, cuda, jitclass

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

spec = [('adjacency_matrix', float64[:, :]), ('activations', int64[:]), ('num_of_nodes', int64), ('in_nodes', int64), ('out_nodes', int64)]
@jitclass(spec)
class Execution(object):

    def __init__(self, adjacency_matrix, activations, num_of_nodes, in_nodes, out_nodes):
        self.adjacency_matrix = adjacency_matrix
        self.activations = activations
        self.num_of_nodes = num_of_nodes
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

    def update(self, inputs):
        num_of_nodes = self.adjacency_matrix.shape[0]
        mem = np.full(num_of_nodes, np.nan)
        # print(inputs)
        for e in range(self.in_nodes):
            mem[e] = inputs[e]
        
        for i in range(self.out_nodes):
            calcResults(num_of_nodes - i - 1, self.adjacency_matrix, self.activations, mem)
        
        outputs = np.array([mem[n - i - 1] for n in range(self.out_nodes)])

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

        self.start_nodes = [n for n,d in self.graph.in_degree() if d == 0]
        self.in_nodes = [n for n in self.graph.nodes.data() if n[1]['type'] == NeuronType.INPUT]
        self.out_nodes = [n for n in self.graph.nodes.data() if n[1]['type'] == NeuronType.OUTPUT]
        self.adj = nx.adjacency_matrix(self.graph).todense()
        
        self.activations = np.array([FuncsEnum[self.graph.nodes()[n]['activation'].__name__].value for n in self.graph.nodes()])

        self.execution = Execution(self.adj, self.activations, self.adj.shape[0], len(self.in_nodes), len(self.out_nodes))
    

    # @staticmethod
    # @njit(float64(int64, float64[:]))
    # def update(inputs: List[float]) -> List[float]:
    #     num_of_nodes = self.adj.shape[0]
    #     mem = np.full(num_of_nodes, np.nan)
    #     # print(inputs)
    #     for e in range(len(self.in_nodes)):
    #         mem[e] = inputs[e]
        
    #     for i in range(len(self.out_nodes)):
    #         calcResults(num_of_nodes - i - 1, self.adj, self.activations, mem)
        
    #     outputs = [mem[n - i - 1] for n in range(len(self.out_nodes))]

    #     return outputs


# @njit(float64(int64, float64[:, :], int64[:], float64[:]))
# def calcResults(n, adj, acts, mem):
#     preds = adj.T[n].nonzero()[0]
    
#     if len(preds) == 0:
#         return mem[n]

#     inputs = np.array([mem[p] if not np.isnan(mem[p]) else calcResults(p, adj, acts, mem) for p in preds])
    
#     weights = adj[preds][:, n]

#     result = np.sum(weights * inputs)

#     function = acts[n]
#     if function == FuncsEnum.tanh.value:
#         mem[n] = math.tanh(result)
#     elif function == FuncsEnum.sin.value:
#         mem[n] = math.sin(result)
#     elif function == FuncsEnum.cos.value:
#         mem[n] = math.cos(result)

#     return mem[n]