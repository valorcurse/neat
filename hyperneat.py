from typing import List, Set, Dict, Tuple, Optional

import numpy as np
from copy import deepcopy

from neat.neat import NEAT
from neat.utils import fastCopy
from neat.types import NeuronType
from neat.phenotypes import Phenotype, SubstrateCUDA
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
    # @cuda.jit
    @njit
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
            
            array[i, 0] = X[x][0]
            array[i, 1] = X[x][1]
            array[i, 2] = Y[y][0]
            array[i, 3] = Y[y][1]


    def createSubstrate(self, cppn: Genome) -> Phenotype:
            
        cppnPheno = cppn.createPhenotype()
        print("cppn outputs:", len(cppnPheno.out_nodes))
        # cppn_exec = cppnPheno.execution
        
        graph = nx.DiGraph()
        graph.add_nodes_from([(n.ID,  {"activation": n.activation}) for n in self.substrateNeurons if n.neuronType != NeuronType.HIDDEN])
        # graph.add_nodes_from([(n.ID,  {"activation": n.activation}) for n in self.substrateNeurons])

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
        # batch_size = 100000000
        for i in range(coordinates.shape[0] - 1):
            leftNeuronLayer = coordinates[i]
            leftDepth = leftNeuronLayer[0].y

            X = np.array([(n.x, n.y) for n in leftNeuronLayer])
            X_IDs = np.array([n.ID for n in leftNeuronLayer])
            X_data = {"data": X, "IDs": X_IDs}

            for j in range(i+1, coordinates.shape[0]):
                rightNeuronLayer = coordinates[j]
                rightDepth = rightNeuronLayer[0].y

                Y = np.array([(n.x, n.y) for n in rightNeuronLayer])
                Y_IDs = np.array([n.ID for n in rightNeuronLayer])
                Y_data = {"data": Y, "IDs": Y_IDs}

                links = np.empty((X.shape[0], 3))

                substrateCUDA = SubstrateCUDA(cppnPheno)
                outputs = substrateCUDA.update(X_data, Y_data)
                # links.append(outputs)
                graph.add_weighted_edges_from(outputs)

        # graph.add_weighted_edges_from(links)
        # graph.remove_nodes_from(list(nx.isolates(graph)))

        return Phenotype(graph, cppn.ID)
