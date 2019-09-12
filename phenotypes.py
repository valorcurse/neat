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
from numba import jit, njit, generated_jit, int64, float64, cuda, jitclass, void, vectorize

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

# spec = [('adjacency_matrix', float64[:, :]), ('activations', int64[:]), ('num_of_nodes', int64), ('in_nodes', int64), ('out_nodes', int64)]
# @jitclass(spec)
class Execution(object):

    def __init__(self, adjacency_matrix, activations, num_of_nodes, in_nodes, out_nodes):
        self.adjacency_matrix = adjacency_matrix
        self.activations = activations
        self.num_of_nodes = num_of_nodes
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

    # @vectorize
    # def update(self, inputs):
    def update(self, X):
        Y = np.empty(X.shape)
        num_of_nodes = self.adjacency_matrix.shape[0]
        mem = np.zeros((X.shape[0], num_of_nodes))
        mem[:, :X.shape[1]] = X
        
        print(mem)

        cuda_adj = cuda.to_device(self.adjacency_matrix)
        cuda_acts = cuda.to_device(self.activations)
        cuda_mem = cuda.to_device(mem)

        print(self.adjacency_matrix)
        
        threadsperblock = 32
        blockspergrid = (X.shape[0] + (threadsperblock - 1)) // threadsperblock

        print("cuda", blockspergrid, threadsperblock)

        print("mem1", mem)
        calcResults[blockspergrid, threadsperblock](cuda_adj, cuda_acts, cuda_mem)

        mem = cuda_mem.copy_to_host()
        print("mem2", mem)

        outputs = mem.T[-self.out_nodes:]

        print("outputs:", outputs)

        return outputs

@cuda.jit(device=True)
def feedForward(adj, acts, mem):
    adj_t = adj.T
    for col_i in range(adj_t.shape[0]):

        weights = adj_t[col_i]
        result = 0.0
        for k in range(weights.shape[0]):
            result += weights[k]*mem[k]

        function = acts[col_i]
        if function == 0:
            mem[col_i] = math.tanh(result)
        elif function == 1:
            mem[col_i] = math.sin(result)
        elif function == 2:
            mem[col_i] = math.cos(result)

@cuda.jit(void(float64[:, :], int64[:], float64[:, :]))
def calcResults(adj, acts, mem):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if i >= mem.shape[0]:
        return

    feedForward(adj, acts, mem[i])



# @invariant(lambda self: all(isinstance(x, SNeuron) for x in self.neurons), ValueError("Some neurons are not of type SNeuron."))
class Phenotype:

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