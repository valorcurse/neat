from __future__ import annotations

from typing import List

import math
import numpy as np
import networkx as nx
import numba
from numba import cuda, jit, vectorize, guvectorize, int64, float32
from numba.types import List
import multiprocessing as mp
# from pathos.multiprocessing import ProcessPool
from joblib import Parallel, delayed

from neat.neatTypes import NeuronType
from neat.genes import NeuronGene, FuncsEnum
# from neat.cuda_matmult import cu_square_matrix_mul

from copy import deepcopy

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

# @invariant(lambda self: all(isinstance(x, SNeuron) for x in self.neurons), ValueError("Some neurons are not of type SNeuron."))
class Phenotype:

    def __init__(self, graph, ID: int) -> None:
        self.ID = ID
        self.graph = graph

        self.start_nodes = [n for n,d in self.graph.in_degree() if d == 0]
        self.in_nodes = [n for n in self.graph.nodes.data() if n[1]['type'] == NeuronType.INPUT]
        self.out_nodes = [n for n in self.graph.nodes.data() if n[1]['type'] == NeuronType.OUTPUT]

        # self.adjacency_matrix = nx.adjacency_matrix(self.graph).todense()
        self.adjacency_matrix = nx.to_numpy_matrix(self.graph)

        self.activations = np.array([FuncsEnum[self.graph.nodes()[n]['activation'].__name__].value for n in self.graph.nodes()])

    def update(self, X):
        pass

class FeedforwardCUDA(object):


    def __init__(self, phenotypes: List[Phenotype]):
        self.kernel_spec = 'void(float32[:], float32[:])'
        # self.cuda_kernels = [self.init_kernel(p) for p in phenotypes]

        self.threadsperblock = 32
        self.blockspergrid = (len(phenotypes) + (self.threadsperblock - 1)) // self.threadsperblock

        self.phenotypes = phenotypes

        self.num_of_outputs = len(self.phenotypes[0].out_nodes)

    def kernel_maker(self, adj_size, adj, acts):
        def impl(X, results):
            # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            # i = cuda.grid(1)

            mem = cuda.local.array(adj_size, dtype=float32)
            for x_i in range(X.shape[0]):
                x = X[x_i]
                mem[x_i] = x

            feedForward(adj, acts, mem)

            for j in range(results.shape[0]):
                results[j] = mem[-j - 1]

        return impl

    def init_kernel(self, phenotype):
        kernel_spec = self.kernel_maker(phenotype.adjacency_matrix.shape[0], phenotype.adjacency_matrix, phenotype.activations)

        kernel = cuda.jit(self.kernel_spec)(kernel_spec)
        kernel.total_size = phenotype.adjacency_matrix.shape[0] + phenotype.adjacency_matrix.size + phenotype.activations.size

        return kernel


    def update(self, X):
        mem = np.array([np.zeros(p.adjacency_matrix.shape[0], dtype=np.float32) for p in self.phenotypes])
        mem[:, :X.shape[1]] = X

        adjs = np.array([p.adjacency_matrix for p in self.phenotypes])
        acts = np.array([p.activations for p in self.phenotypes])
        results = np.array([np.empty(self.num_of_outputs, dtype=np.float32) for _ in self.phenotypes])

        cuda_mem = cuda.to_device(mem)
        cuda_adj = cuda.to_device(adjs)
        cuda_acts = cuda.to_device(acts)
        cuda_results = cuda.to_device(results)
        execute_network[self.blockspergrid, self.threadsperblock](cuda_mem, cuda_adj, cuda_acts, cuda_results)

        results = cuda_results.copy_to_host()

        return results

class SubstrateCUDA(object):


    def __init__(self, phenotype: Phenotype):
        self.phenotype = phenotype
        self.adjacency_matrix = self.phenotype.adjacency_matrix
        self.activations = self.phenotype.activations
        self.num_of_nodes = len(self.phenotype.graph.nodes)
        self.num_in_nodes = len(self.phenotype.in_nodes)
        self.num_out_nodes = len(self.phenotype.out_nodes)

        self.batch_size = 10 * 10**7

    def kernel_maker(self, size):
        def impl(X, Y, adj, acts, start_index, results):
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if i >= results.shape[0]:
                return

            mem = cuda.local.array(size, dtype=float32)

            calculateLinks(X, Y, start_index, i, mem)
            feedForward(adj, acts, mem)

            results[i][0] = mem[-1]

        return impl

    def update(self, X, Y):
        
        X_data = X["data"]
        Y_data = Y["data"]

        custom_sized_kernel = self.kernel_maker(self.adjacency_matrix.shape[0])
        calculateSubstrate = cuda.jit('void(float32[:,:], float32[:,:], float32[:,:], int64[:], int64, float32[:,:])')(custom_sized_kernel)
        assert '.local' in calculateSubstrate.ptx

        num_of_nodes = self.adjacency_matrix.shape[0]
        cuda_adj = cuda.to_device(self.adjacency_matrix)
        cuda_acts = cuda.to_device(self.activations)

        total_size = Y_data.shape[0]*X_data.shape[0]
        b_size = min(self.batch_size, total_size)

        threads_per_block = 32
        blockspergrid = (b_size + (threads_per_block - 1)) // threads_per_block

        links = []
        num_of_batches = int(math.ceil(total_size/b_size))
        for batch_i in range(num_of_batches):
            start_index = batch_i*b_size

            results = np.empty((b_size, self.num_out_nodes))
            cuda_results = cuda.to_device(results)

            calculateSubstrate[blockspergrid, threads_per_block](X_data, Y_data, cuda_adj, cuda_acts, start_index, cuda_results)

            results = cuda_results.copy_to_host()

            x_size = X_data.shape[0]
            y_size = Y_data.shape[0]
            nonzero = results.nonzero()
            for r in nonzero[0]:
                data_i = start_index+r - 1
                x = int(data_i % x_size) - 1
                y = int(abs(data_i % y_size-math.floor(data_i/x_size))) - 1

                out_node = X["IDs"][x]
                in_node = Y["IDs"][y]
                weight = results[r][0]
                edge = (out_node, in_node, weight)
                links.append(edge)


        return links

@cuda.jit(device=True)
def calculateLinks(X, Y, start_index, i, mem):
    x_size = X.shape[0]
    y_size = Y.shape[0]
    array_size = mem.shape[0]

    batch_i = start_index+i
    x = int(batch_i % x_size)
    y = int(abs((batch_i + math.floor(batch_i/x_size)) % y_size))

    mem[0] = X[x][0]
    mem[1] = X[x][1]
    mem[2] = Y[y][0]
    mem[3] = Y[y][1]

@cuda.jit(device=True)
def feedForward(adj, acts, mem):
    adj_t = adj.T
    for col_i in range(adj_t.shape[0]):
        weights = adj_t[col_i].T

        result = 0.0
        for i in mem:
            # if i != 0.0:
            #     continue

            added = 0.0
            for j in weights:
                added += i*j
            result += added

        function = acts[col_i]
        if function == 0:
            mem[col_i] += math.tanh(result)
        elif function == 1:
            mem[col_i] += math.sin(result)
        elif function == 2:
            mem[col_i] += math.cos(result)

@cuda.jit()
def execute_network(all_mem, all_adj, all_acts, all_results):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if i >= all_mem.shape[0]:
        return

    mem = all_mem[i]
    adj = all_adj[i]
    acts = all_acts[i]
    results = all_results[i]


    feedForward(adj, acts, mem)

    for j in range(results.shape[0]):
        results[j] = mem[-j - 1]