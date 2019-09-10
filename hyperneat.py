from typing import List, Set, Dict, Tuple, Optional

import numpy as np
from copy import deepcopy

from neat.neat import NEAT
from neat.utils import fastCopy
from neat.types import NeuronType
from neat.phenotypes import Phenotype
from neat.population import PopulationUpdate, PopulationConfiguration
from neat.genes import Genome, LinkGene, NeuronGene, MutationRates, Phase, SpeciationType

import networkx as nx
from numba import jit, njit, generated_jit, int64, float64, cuda, jitclass, void

import math

from itertools import groupby

from timeit import default_timer as timer
from visualize import Visualize


class HyperNEAT(NEAT):

    def __init__(self, population_configuration: PopulationConfiguration, mutation_rates: MutationRates=MutationRates()):
        
        # CPPNs take 4 inputs, gotta move this somewhere else
        nrOfLayers: int = 3
        self.layers = np.linspace(-1.0, 1.0, num=nrOfLayers)
        cppnInputs: int = 4
        # cppnOutputs: int = nrOfLayers - 1
        cppnOutputs = 1

        self.n_inputs = population_configuration.n_inputs
        self.n_outputs = population_configuration.n_outputs


        # Substrate
        hiddenLayersWidth: int = self.n_inputs/2

        self.substrateNeurons: List[NeuronGene] = []
        for y in self.layers:
            
            if y == -1.0:
                for x in np.linspace(-1.0, 1.0, num=self.n_inputs):
                    self.substrateNeurons.append(NeuronGene(NeuronType.INPUT, len(self.substrateNeurons), y, x))        

            elif y == 1.0:
                for x in np.linspace(-1.0, 1.0, num=self.n_outputs):
                    self.substrateNeurons.append(NeuronGene(NeuronType.OUTPUT, len(self.substrateNeurons), y, x))
            
            else:
                for x in np.linspace(-1.0, 1.0, num=hiddenLayersWidth):
                    self.substrateNeurons.append(NeuronGene(NeuronType.HIDDEN, len(self.substrateNeurons), y, x))


        population_configuration._data["n_inputs"] = 4
        population_configuration._data["n_outputs"] = 1
        self.neat = NEAT(population_configuration, mutation_rates)


    def epoch(self, update_data: PopulationUpdate) -> List[Phenotype]:
        self.neat.population.updatePopulation(update_data)

        print("Reproducing")
        cppns = self.neat.population.reproduce()
        substrates = []
        for i, c in enumerate(cppns):
            print("epoch:", i)
            substrates.append(self.createSubstrate(c).createPhenotype())
            
            print("\rCreated substrates: %d/%d"%(i, len(cppns)), end='')

        print("")

        return substrates


    @staticmethod
    # @njit
    def calculateLinks(x, Y):
        array = np.empty((Y.shape[0], 4))
        # for i, x in enumerate(X):
        for i, y in enumerate(Y):
            coords = np.array([x, y]).flatten()
            # print(array[i, :], coords)
            array[i, :] = coords

        return array

    @staticmethod
    # @njit
    @cuda.jit(void(float64[:], float64, float64[:], float64, int64[:], float64[:]))
    def calculateLinks2(X, y1, Y, y2, execution, links):
        # array = np.empty(X.shape[0])
        # links = np.empty((array.shape[0], 3))

        # startX, startY = cuda.grid(1)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x


        x = i%X.shape[0]
        y = abs(i%2-math.floor(i/Y.shape[0]))

        x1 = X[int(x)]
        x2 = Y[int(y)]

        inputs = [x1, y1, x2, y2]
        # value = execution.update(inputs)
        # gridX = cuda.gridDim.x * cuda.blockDim.x
        # gridY = cuda.gridDim.y * cuda.blockDim.y

        # for i, x1 in enumerate(X.T[0]):
            # for j, x2, in enumerate(Y.T[0]):
                # inputs = [x1, y1, x2, y2]

                # value = execution.update(inputs)
                # links[len(links)-1] = [X[i][1], X[j][1], value[0]]

        # return links


        # return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    def createSubstrate(self, cppn: Genome) -> Phenotype:
            
        cppnPheno = cppn.createPhenotype()
        print("cppn outputs:", len(cppnPheno.out_nodes))
        cppn_exec = cppnPheno.execution
        
        graph = nx.DiGraph()
        graph.add_nodes_from([(n.ID,  {"activation": n.activation}) for n in self.substrateNeurons])
        nrOfInputs = self.n_inputs
        nrOfOutputs = self.n_outputs

        # substrateGenome: Genome = Genome(cppn.ID, nrOfInputs, nrOfOutputs, neurons=fastCopy(self.substrateNeurons))

        # layers = [list(g) for k, g in groupby(substrateGenome.neurons, lambda n: n.y)]
        layers = [list(g) for k, g in groupby(self.substrateNeurons, lambda n: n.y)]

        coordinates = []
        for l in layers:
            singleLayer = np.array([n for n in l])
            coordinates.append(singleLayer)
        coordinates = np.array(coordinates)

        links = []
        for i in range(coordinates.shape[0] - 1):
            leftNeuronLayer = coordinates[i]

            X = np.array([(n.x, n.ID) for n in leftNeuronLayer])
            leftDepth = leftNeuronLayer[0].y



            for j in range(i+1, coordinates.shape[0]):
                rightNeuronLayer = coordinates[j]
                # print(rightNeuronLayer)
                rightDepth = rightNeuronLayer[0].y

                Y = np.array([(n.x, n.ID) for n in rightNeuronLayer])
                # print(X)
                # print(i, j)
                
                t1 = timer()
                links = np.empty((X.shape[0], 3))

                # print(X, Y)
                for x in X:
                    inputs = self.calculateLinks(x, Y)
                    cppnPheno.execution.update(inputs)

                # totalArraySize = X.size * Y.size
                # threadsperblock = 32
                # blockspergrid = (totalArraySize + (threadsperblock - 1)) // threadsperblock
                # print(totalArraySize, blockspergrid, threadsperblock)
                # self.calculateLinks[blockspergrid, threadsperblock](X, leftDepth, Y, rightDepth, cppn_exec, links)

                # self.calculateLinks(X, leftDepth, Y, rightDepth, cppn_exec, links)
                
                # for l_i, l in enumerate(leftLayerData):
                    # left_ID = leftNeuronLayer[l_i].ID
                    # links = self.calculateLinks(left_ID, l, leftDepth, X, rightDepth, cppn_exec)
                t2 = timer()
                print("time:", t2 - t1)

                    # print(links)
                    # print(values.shape)
                    # for v_i, v in enumerate(values):
                        # graph.add_edge(leftNeuronLayer[l_i].ID, rightNeuronLayer[v_i].ID, weight=v[0])
                        # print(leftNeuronLayer[l_i], rightNeuronLayer[v_i], v)
                        # if abs(v) >= 0.5:
                            # links.append(LinkGene(leftNeuronLayer[l_i], rightNeuronLayer[v_i], len(links), v))
                        # links.append((leftNeuronLayer[l_i].ID, rightNeuronLayer[v_i].ID, v[0]))
            #     # print(rightLayerData)
            #     for r_i, r in enumerate(rightLayerData):
            #         print(r)
            #         # positions = self.calculateLinks(rightLayerData, leftLayerData)
            #         positions = self.calculateLinks(np.array([r]), leftLayerData)
            #         # print("positions", positions)
            #         for neuron in zip(positions, np.repeat(leftNeuronLayer, len(rightNeuronLayer)), np.tile(rightNeuronLayer, len(leftNeuronLayer))):
            #             # t1 = timer()
            #             cppnInput = [neuron[0][0], leftDepth, neuron[0][1], rightDepth]
                        
            #             outputs = cppnPheno.update(cppnInput)
            #             output = outputs[0]

            #             # t2 = timer()
            #             # print("time:", t2 - t1)
            #             # if abs(output) >= 0.5:
            #             links.append(LinkGene(neuron[1], neuron[2], len(links), output))

        graph.add_weighted_edges_from(links)
        graph.remove_nodes_from(list(nx.isolates(graph)))

        # substrateGenome.links = links

        # graph.add_weighted_edges_from(links)

        # phenotype = Phenotype(graph, cppn.ID)

        return Phenotype(graph, cppn.ID)
