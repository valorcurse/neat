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

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

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

        self.batch_size = 10 * 10**7

    def update(self, X, Y):
        
        X_data = X["data"]
        Y_data = Y["data"]

        custom_sized_kernel = kernel_maker(self.adjacency_matrix.shape[0])
        calculateSubstrate = cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], int64[:], int64, float64[:,:])')(custom_sized_kernel)
        assert '.local' in calculateSubstrate.ptx

        num_of_nodes = self.adjacency_matrix.shape[0]
        cuda_adj = cuda.to_device(self.adjacency_matrix)
        cuda_acts = cuda.to_device(self.activations)

        threadsperblock = 32
        blockspergrid = (self.batch_size + (threadsperblock - 1)) // threadsperblock

        links = []
        total_size = Y_data.shape[0]*X_data.shape[0]
        num_of_batches = int(math.ceil(total_size/self.batch_size))
        print("num_of_batches", num_of_batches)
        for batch_i in range(num_of_batches):
            t1 = timer()

            start_index = batch_i*self.batch_size

            results = np.empty((self.batch_size, self.num_out_nodes))
            cuda_results = cuda.to_device(results)

            calculateSubstrate[blockspergrid, threadsperblock](X_data, Y_data, cuda_adj, cuda_acts, start_index, cuda_results)

            results = cuda_results.copy_to_host()


            x_size = X_data.shape[0]
            y_size = Y_data.shape[0]
            nonzero = results.nonzero()
            print(nonzero)
            for r in nonzero[1]:
                data_i = start_index+r - 1
                # print(data_i)
                x = int(data_i%x_size) - 1
                y = int(abs(data_i%y_size-math.floor(data_i/x_size))) - 1
                print(r, X["IDs"][x], Y["IDs"][y])
                links.append((X["IDs"][x], Y["IDs"][y], results[0][r]))

            t2 = timer()
            print("Batch: {} | Time: {}".format(batch_i, t2-t1))

        return links

@cuda.jit(device=True)
def calculateLinks(X, Y, start_index, i, mem):
    x_size = X.shape[0]
    y_size = Y.shape[0]
    array_size = mem.shape[0]
    range_end = start_index + array_size

    batch_i = start_index+i
    x = int(batch_i%x_size)
    y = int(abs(batch_i%y_size-math.floor(batch_i/x_size)))
    
    mem[0] = X[x][0]
    mem[1] = X[x][1]
    mem[2] = Y[y][0]
    mem[3] = Y[y][1]

@cuda.jit(device=True)
def feedForward(adj, acts, mem):
    adj_t = adj.T
    for col_i in range(adj_t.shape[0]):

        weights = adj_t[col_i]
        result = 0.0
        for k in range(weights.shape[0]):
            result += weights[k]*mem[k]
            # print(weights[k], mem[k])

        function = acts[col_i]
        if function == 0:
            mem[col_i] = math.tanh(result)
        elif function == 1:
            mem[col_i] = math.sin(result)
        elif function == 2:
            mem[col_i] = math.cos(result)

        # print(mem[col_i])

# @cuda.jit(void(float64[:, :], int64[:], float64[:, :]))
# def calcResults(adj, acts, mem):
#     i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

#     if i >= mem.shape[0]:
#         return

#     feedForward(adj, acts, mem[i])

# some_size = 512
# @cuda.jit("void(float64[:,:], float64[:,:], float64[:,:], int64[:], int64, int64, float64[:])")
@cuda.jit
def calcResults2(X, Y, adj, acts, mem, start_index, results):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    calculateLinks(X, Y, start_index, mem)
    feedForward(adj, acts, mem[i])

    results[i][0] = mem[i][-1]
    results[i][1] = mem[i][-2]

def kernel_maker(size):
    def impl(X, Y, adj, acts, start_index, results):
        mem = cuda.local.array(size, dtype=float64)
        
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        calculateLinks(X, Y, start_index, i, mem)
        # print(mem)
        feedForward(adj, acts, mem)


        # print(results[i][0], mem[size-1])
        results[i][0] = mem[-1]
        # results[i][1] = mem[-2]
        # calcResults2(X, Y, adj, acts, mem, start_index, results)

    return impl



# @invariant(lambda self: all(isinstance(x, SNeuron) for x in self.neurons), ValueError("Some neurons are not of type SNeuron."))
class Phenotype:

    def __init__(self, graph, ID: int) -> None:
        self.ID = ID
        self.graph = graph

        self.start_nodes = [n for n,d in self.graph.in_degree() if d == 0]
        self.in_nodes = [n for n in self.graph.nodes.data() if n[1]['type'] == NeuronType.INPUT]
        self.out_nodes = [n for n in self.graph.nodes.data() if n[1]['type'] == NeuronType.OUTPUT]
        self.adjacency_matrix = nx.adjacency_matrix(self.graph).todense()
        
        self.activations = np.array([FuncsEnum[self.graph.nodes()[n]['activation'].__name__].value for n in self.graph.nodes()])
