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
class SubstrateCUDA(object):

    def __init__(self, phenotype: Phenotype):
        self.phenotype = phenotype
        self.adjacency_matrix = self.phenotype.adjacency_matrix
        self.activations = self.phenotype.activations
        self.num_of_nodes = len(self.phenotype.graph.nodes)
        self.num_in_nodes = len(self.phenotype.in_nodes)
        self.num_out_nodes = len(self.phenotype.out_nodes)

        self.batch_size = 50 * 10**6
    # def update(self, X):
    #     # Y = np.empty(X.shape)
    #     num_of_nodes = self.adjacency_matrix.shape[0]
    #     mem = np.zeros((X.shape[0], num_of_nodes))
    #     mem[:, :X.shape[1]] = X
    #     # print(mem)

    #     cuda_adj = cuda.to_device(self.adjacency_matrix)
    #     cuda_acts = cuda.to_device(self.activations)
    #     cuda_mem = cuda.to_device(mem)

    #     threadsperblock = 32
    #     blockspergrid = (X.shape[0] + (threadsperblock - 1)) // threadsperblock

    #     calcResults[blockspergrid, threadsperblock](cuda_adj, cuda_acts, cuda_mem)

    #     mem = cuda_mem.copy_to_host()
    #     outputs = mem.T[-self.num_out_nodes:].T

    #     return outputs


    def update(self, X, Y):
        cuda_adj = cuda.to_device(self.adjacency_matrix)
        cuda_acts = cuda.to_device(self.activations)

        threadsperblock = 32
        blockspergrid = (self.batch_size + (threadsperblock - 1)) // threadsperblock

        num_of_nodes = self.adjacency_matrix.shape[0]

        total_size = Y.shape[0]*X.shape[0]
        num_of_batches = int(math.ceil(total_size/self.batch_size))
        print("num_of_batches", num_of_batches)
        for batch_i in range(num_of_batches):
            t1 = timer()

            start_index = batch_i*self.batch_size

            array = np.empty((self.batch_size, 4))
            mem = np.empty((self.batch_size, num_of_nodes))
            cuda_mem = cuda.to_device(mem)
            results = np.empty((self.batch_size, self.num_out_nodes))
            cuda_results = cuda.to_device(results)

            calcResults2[blockspergrid, threadsperblock](X, Y, cuda_adj, cuda_acts, cuda_mem, start_index, cuda_results)

            # mem = cuda_mem.copy_to_host()
            results = cuda_results.copy_to_host()
            # outputs = mem.T[-self.num_out_nodes:].T

            t2 = timer()
            print("Batch: {} | Time: {}".format(batch_i, t2-t1))
            print(results)

        return outputs


# print(i%X.shape[0], abs(i%Y.shape[0]-math.floor(i/X.shape[0]))) 

# @njit
@cuda.jit(device=True)
def calculateLinks(X, Y, start_index, array):
    x_size = X.shape[0]
    y_size = Y.shape[0]
    array_size = array.shape[0]
    range_end = start_index + array_size

    for i in range(array_size):
        if range_end >= x_size*y_size:
            break

        batch_i = start_index+i
        x = int(batch_i%x_size)
        y = int(abs(batch_i%y_size-math.floor(batch_i/x_size)))
        
        # print(x, y)
        # print(array.shape, X[x].shape)
        array[0] = X[x][0]
        array[1] = X[x][1]
        array[2] = Y[y][0]
        array[3] = Y[y][1]


# @cuda.jit
# def calculateLinks(X, Y):
#     array = np.empty((Y.shape[0], 4))
#     # for i, x in enumerate(X):
#     for i, y in enumerate(Y):
#         coords = np.array([x, y]).flatten()
#         # print(array[i, :], coords)
#         array[i, :] = coords


@cuda.jit(device=True)
def feedForward(adj, acts, mem):
    adj_t = adj.T
    for col_i in range(adj_t.shape[0]):

        weights = adj_t[col_i]
        result = 0.0
        for k in range(weights.shape[0]):
            result += weights[k]*mem[k]

        # print(result)
        function = acts[col_i]
        if function == 0:
            mem[col_i] = math.tanh(result)
        elif function == 1:
            mem[col_i] = math.sin(result)
        elif function == 2:
            mem[col_i] = math.cos(result)

        # print(mem[col_i])

@cuda.jit(void(float64[:, :], int64[:], float64[:, :]))
def calcResults(adj, acts, mem):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if i >= mem.shape[0]:
        return

    feedForward(adj, acts, mem[i])

@cuda.jit
def calcResults2(X, Y, adj, acts, mem, start_index, results):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    calculateLinks(X, Y, start_index, mem[i])
    feedForward(adj, acts, mem[i])

    results[i][0] = mem[i][-1]
    results[i][1] = mem[i][-2]



# @invariant(lambda self: all(isinstance(x, SNeuron) for x in self.neurons), ValueError("Some neurons are not of type SNeuron."))
class Phenotype:

    def __init__(self, graph, ID: int) -> None:
        # self.neurons = neurons
        self.ID = ID
        self.graph = graph

        self.start_nodes = [n for n,d in self.graph.in_degree() if d == 0]
        self.in_nodes = [n for n in self.graph.nodes.data() if n[1]['type'] == NeuronType.INPUT]
        self.out_nodes = [n for n in self.graph.nodes.data() if n[1]['type'] == NeuronType.OUTPUT]
        self.adjacency_matrix = nx.adjacency_matrix(self.graph).todense()
        
        self.activations = np.array([FuncsEnum[self.graph.nodes()[n]['activation'].__name__].value for n in self.graph.nodes()])

        # self.execution = Execution(self.adj, self.activations, self.adj.shape[0], len(self.in_nodes), len(self.out_nodes))