from __future__ import annotations

from typing import List

import math
import numpy as np
import numpy.ma as ma
import networkx as nx
import numba
from numba import cuda
from numba.types import List

import sys

import neat.genes
from neat.neatTypes import NeuronType
from neat.utils import chunks

from copy import deepcopy

np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))

class SNeuron:
    
    def __init__(self, neuronGene: neat.phenotypes.NeuronGene) -> None:
        self.linksIn: List[SLink] = []

        self.activation = neuronGene.activation
        self.bias = 1.0
        # self.output: float = 1.0 if neuronGene.neuronType == NeuronType.BIAS else 0.0
        self.output: float = 0.0

        self.neuronType = neuronGene.neuronType

        self.ID: int = neuronGene.ID

        self.y: float = neuronGene.y
        self.x: float = neuronGene.x

    def activate(self, x: float) -> float:
        return self.activation(x + self.bias)

    def __repr__(self):
        return "SNeuron(Type={0}, ID={1}, x={2}, y={3})".format(self.neuronType, self.ID, self.x, self.y)

class SLink:

    def __init__(self, fromNeuron: SNeuron, toNeuron: SNeuron, weight: float, recurrent: bool = False) -> None:
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron

        self.weight = weight

        self.recurrent = recurrent

class Phenotype:

    def __init__(self, graph, ID: int) -> None:
        self.ID = ID
        self.graph = graph

        self.start_nodes = [n for n,d in self.graph.in_degree() if d == 0]
        self.in_nodes = [n for n in self.graph.nodes.data() if n[1]['type'] == NeuronType.INPUT]
        self.out_nodes = [n for n in self.graph.nodes.data() if n[1]['type'] == NeuronType.OUTPUT]

        sorted_nodes = list(nx.topological_sort(self.graph))

        # Adjacency matrix sorted topologically
        self.adjacency_matrix = nx.adjacency_matrix(self.graph, nodelist=sorted_nodes).todense()
        # self.adjacency_matrix = nx.to_numpy_matrix(self.graph, dtype=np.float32)
        self.activations = []
        for i in sorted_nodes:
            node = self.graph.nodes()[i]
            self.activations.append(neat.genes.FuncsEnum[node['activation'].__name__].value)
        self.activations = np.array(self.activations, dtype=np.int32)

        # self.activations = np.array([neat.genes.FuncsEnum[self.graph.nodes()[n]['activation'].__name__].value for n in self.graph.nodes()], dtype=np.int32)

        self.bias = []
        for i in sorted_nodes:
            node = self.graph.nodes()[i]
            self.bias.append(node['bias'])
        self.bias = np.array(self.bias, dtype=np.float32)

        # self.bias = np.array([self.graph.nodes()[n]['bias'] for n in self.graph.nodes()], dtype=np.float32)

        self.fitness = 0.0

    def update(self, X):
        pass

'''
Executes neuralnetworks on the GPU for environments that
need to be executed sequentially
'''
class SequentialCUDA(object):


    def __init__(self):
        self.threadsperblock = 32

    def update(self, phenotypes, X):

        if type(X) is not np.ndarray:
            X = np.array(X)

        blockspergrid = (len(phenotypes) + (self.threadsperblock - 1)) // self.threadsperblock

        num_of_outputs = len(phenotypes[0].out_nodes)

        assert X.shape == (len(phenotypes), len(phenotypes[0].in_nodes)), \
            "Incorrect number of input values. There are {} instead of {}: {}".format(
                X.shape,
                (len(phenotypes), len(phenotypes[0].in_nodes)),
                phenotypes[0].graph.nodes.data()
            )



        adj_matrices = [p.adjacency_matrix for p in phenotypes]
        acts = [p.activations for p in phenotypes]
        bias = [p.bias for p in phenotypes]

        largest_adj_size = max([n.shape[0] for n in adj_matrices])
        original_sizes = np.zeros(len(phenotypes), dtype=np.int32)
        # Pad all matrices to conform with the largest network
        for i, (adj, act, bia) in enumerate(zip(adj_matrices, acts, bias)):
            original_sizes[i] = adj.shape[0]
            new_size = largest_adj_size - adj.shape[0]

            adj_matrices[i] = np.pad(adj, [(0, new_size), (0, new_size)], mode='constant')
            acts[i] = np.pad(act, [(0, new_size)], mode='constant')
            bias[i] = np.pad(bia, [(0, new_size)], mode='constant')

        adj_matrices = np.array(adj_matrices, dtype=np.float32)
        acts = np.array(acts, dtype=np.int32)
        bias = np.array(bias, dtype=np.float32)

        mem = np.array([np.zeros(largest_adj_size, dtype=np.float32) for _ in phenotypes])

        # Copy inputs to mem
        for row_i, row in enumerate(mem):
            row[:X.shape[1]] = X[row_i]


        results = np.array([np.zeros(num_of_outputs, dtype=np.float32) for _ in adj_matrices])

        cuda_mem = cuda.to_device(mem)
        cuda_adj = cuda.to_device(adj_matrices)
        cuda_acts = cuda.to_device(acts)
        cuda_bias = cuda.to_device(bias)
        cuda_original_sizes = cuda.to_device(original_sizes)
        cuda_results = cuda.to_device(results)

        execute_network[blockspergrid, self.threadsperblock](cuda_mem, cuda_adj, cuda_acts, cuda_bias, cuda_original_sizes, cuda_results)
        results = cuda_results.copy_to_host()

        return results

'''
Executes neural networks on the GPU for environments that
can be parallelized
'''
class ParallelCUDA(object):


    def __init__(self, X):
        self.threadsperblock = 891
        self.max_global_data = 0x10000

        print("Data size: {}".format(sys.getsizeof(X)))

        # self.kernels = [cuda.jit()(self.create_kernel(x)) for x in chunks(X, self.max_global_data)]
        chunk_size = self.max_global_data / (X.shape[1] * X.itemsize)
        chunk_size = int(chunk_size / 100.0) * 100
        self.kernels = [cuda.jit()(self.create_kernel(x)) for x in chunks(X, chunk_size)]



    def create_kernel(self, data):
        print("Created kernel: {}".format(data.size))
        def impl(all_mem, all_adj, all_acts, all_bias, all_original_sizes, all_results):
            # i, j = cuda.grid(2)

            # i is de row of the data
            i = numba.cuda.blockIdx.x
            # j is de phenotype
            j = numba.cuda.blockIdx.y

            # i = numba.cuda.threadIdx

            X = cuda.const.array_like(data)[i]

            if j >= all_mem.shape[0]:
                return

            mem = all_mem[j][i]
            adj = all_adj[j]
            acts = all_acts[j]
            bias = all_bias[j]
            original_size = all_original_sizes[j]
            results = all_results[j][i]

            # for x in range(X.shape[0]):
            #     mem[x] = X[x]
            #
            feedForward_parallel(adj, acts, bias, mem, X)

            size_diff = mem.shape[0] - original_size
            for k in range(results.shape[0]):
                results[i] = mem[-(results.shape[0] + size_diff) + k]

        return impl

    def update(self, phenotypes):

        # if type(X) is not np.ndarray:
        #     X = np.array(X)

        blockspergrid = (len(phenotypes) + (self.threadsperblock - 1)) // self.threadsperblock

        # num_of_outputs = len(phenotypes[0].out_nodes)
        num_of_outputs = len(phenotypes[0].out_nodes)

        # assert X.shape == (len(phenotypes), len(phenotypes[0].in_nodes)), \
        #     "Incorrect number of input values. There are {} instead of {}: {}".format(
        #         X.shape,
        #         (len(phenotypes), len(phenotypes[0].in_nodes)),
        #         phenotypes[0].graph.nodes.data()
        #     )



        adj_matrices = [p.adjacency_matrix for p in phenotypes]
        acts = [p.activations for p in phenotypes]
        bias = [p.bias for p in phenotypes]

        largest_adj_size = max([n.shape[0] for n in adj_matrices])
        original_sizes = np.zeros(len(phenotypes), dtype=np.int32)
        # Pad all matrices to conform with the largest network
        for i, (adj, act, bia) in enumerate(zip(adj_matrices, acts, bias)):
            original_sizes[i] = adj.shape[0]
            new_size = largest_adj_size - adj.shape[0]

            adj_matrices[i] = np.pad(adj, [(0, new_size), (0, new_size)], mode='constant')
            acts[i] = np.pad(act, [(0, new_size)], mode='constant')
            bias[i] = np.pad(bia, [(0, new_size)], mode='constant')

        adj_matrices = np.array(adj_matrices, dtype=np.float32)
        acts = np.array(acts, dtype=np.int32)
        bias = np.array(bias, dtype=np.float32)

        mem = np.array([np.zeros((largest_adj_size, X.shape[0]), dtype=np.float32) for _ in phenotypes])

        # Copy inputs to mem
        # for row_i, row in enumerate(mem):
        #     row[:X.shape[1]] = X[row_i]


        results = np.array([np.zeros((len(phenotypes), X.shape[0]), dtype=np.float32) for _ in adj_matrices])

        cuda_mem = cuda.to_device(mem)
        cuda_adj = cuda.to_device(adj_matrices)
        cuda_acts = cuda.to_device(acts)
        cuda_bias = cuda.to_device(bias)
        cuda_original_sizes = cuda.to_device(original_sizes)
        cuda_results = cuda.to_device(results)

        results_head = 0
        for kernel in self.kernels:
            kernel[blockspergrid, self.threadsperblock](cuda_mem, cuda_adj, cuda_acts, cuda_bias, cuda_original_sizes, cuda_results)
            copy_results = cuda_results.copy_to_host()
            next_head = results_head + copy_results.shape[0]
            results[results_head:next_head] = copy_results
            results_head = next_head
        # self.kernel[blockspergrid, self.threadsperblock](cuda_mem, cuda_adj, cuda_acts, cuda_bias, cuda_original_sizes, cuda_results)
        #     results = cuda_results.copy_to_host()

        return results

@cuda.jit(device=True)
def feedForward(adj, acts, bias, mem):
    adj_t = adj.T

    for i in range(mem.size):
        # print("###############")
        function = acts[i]
        weights = adj_t[i].T

        sum = mem[i]
        for j in range(weights.shape[0]):
            weight = weights[j]
            x = mem[j]

            # print("x", x)
            if weight == 0.0:
                continue

            # print(j, i, weight, x)
            sum += x*weight
            # print("sum", sum)

        # print("bias:", bias[i])
        sum += bias[i]
        # print("sum", sum)
        if function == 0: # Tanh
            mem[i] = math.tanh(sum)
        elif function == 1: # Sine
            mem[i] = math.sin(sum)
        elif function == 2: # Cosine
            mem[i] = math.cos(sum)
        elif function == 3: # Sigmoid
            # print("sum", sum)
            mem[i] = 1 / (1 + math.exp(-sum))
            # print("sigmoid", mem[i])
        elif function == 4: # Leaky ReLu
            mem[i] = sum if sum > 0.0 else sum * 0.01
        elif function == 5: # Linear
            mem[i] = sum
            # print("linear", mem[i])
        elif function == 6: # Inverse
            mem[i] = -sum
        elif function == 7: # Absolute
            mem[i] = abs(sum)
        elif function == 8: # Step
            mem[i] = 1.0 if sum > 0.0 else 0.0

@cuda.jit(device=True)
def feedForward_parallel(adj, acts, bias, mem, X):
    adj_t = adj.T

    for i in range(mem.shape[0]):
        function = acts[i]
        weights = adj_t[i].T

        sum = 0.0
        for j in range(weights.shape[0]):
            x = X[i] if i < X.size else mem[i]
            weight = weights[j]

            if function == 6:
                sum = max(sum, x*weight)
            else:
                sum += x*weight

        sum += bias[i]
        if function == 0:
            mem[i] = math.tanh(sum)
        elif function == 1:
            mem[i] = math.sin(sum)
        elif function == 2:
            mem[i] = math.cos(sum)
        elif function == 3:
            mem[i] = 1 / (1 + math.exp(-sum))  # Sigmoid
        elif function == 4:
            mem[i] = sum if sum > 0.0 else sum * 0.01  # Leaky ReLu
        elif function == 5:
            mem[i] = sum  # Linear

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

            results = np.zeros((b_size, self.num_out_nodes))
            cuda_results = cuda.to_device(results)

            mem = np.array([np.zeros(self.adjacency_matrix.shape[0], dtype=np.float32) for _ in range(len(X))])

            calc_substrate[blockspergrid, threads_per_block](X_data, Y_data, cuda_adj, cuda_acts, mem, cuda_results)

            results = cuda_results.copy_to_host()
            results = np.around(results, 2)

            x_size = X_data.shape[0]
            y_size = Y_data.shape[0]

            nonzero = np.where(results >= 0.25)
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


@cuda.jit()
def execute_network(all_mem, all_adj, all_acts, all_bias, all_original_sizes, all_results):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= all_mem.shape[0]:
        return

    mem = all_mem[i]
    adj = all_adj[i]
    acts = all_acts[i]
    bias = all_bias[i]
    original_size = all_original_sizes[i]
    results = all_results[i]

    feedForward(adj, acts, bias, mem)

    size_diff = results.shape[0] - original_size
    for j in range(results.shape[0]):
        results[j] = mem[-(results.shape[0] + size_diff) + j]

@cuda.jit()
def calc_substrate(X, Y, adj, acts, mems, results):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= results.shape[0]:
        return

    mem = mems[i]
    x_size = X.shape[0]
    y_size = Y.shape[0]

    x = int(i % x_size)
    y = int(abs((i + math.floor(i / x_size)) % y_size))

    mem[0] = X[x][0]
    mem[1] = X[x][1]
    mem[2] = Y[y][0]
    mem[3] = Y[y][1]

    feedForward(adj, acts, mem)

    for j in range(results.shape[0]):
        results[j] = mem[-j - 1]

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