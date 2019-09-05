from typing import List, Set, Dict, Tuple, Optional

import numpy as np
from copy import deepcopy

from neat.neat import NEAT
from neat.utils import fastCopy
from neat.types import NeuronType
from neat.phenotypes import Phenotype
from neat.population import PopulationUpdate, PopulationConfiguration
from neat.genes import Genome, LinkGene, NeuronGene, MutationRates, Phase, SpeciationType

from numba import jit, njit, int32, float32, cuda

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
            substrates.append(self.createSubstrate(c).createPhenotype())
            
            print("\rCreated substrates: %d/%d"%(i, len(cppns)), end='')

        print("")

        return substrates


    @staticmethod
    @njit
    def calculateLinks(X, Y):
        array = np.empty((X.shape[0]*Y.shape[0], 2))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                array[i, :] = (x, y)

        return array

        # return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    def createSubstrate(self, cppn):
        t1 = timer()
            
        cppnPheno = cppn.createPhenotype()

        nrOfInputs = self.n_inputs
        nrOfOutputs = self.n_outputs

        substrateGenome: Genome = Genome(cppn.ID, nrOfInputs, nrOfOutputs, neurons=fastCopy(self.substrateNeurons))

        layers = [list(g) for k, g in groupby(substrateGenome.neurons, lambda n: n.y)]

        coordinates = []
        for l in layers:
            singleLayer = np.array([n for n in l])
            coordinates.append(singleLayer)
        coordinates = np.array(coordinates)


        links = []
        for i in range(coordinates.shape[0] - 1):
            leftNeuronLayer = coordinates[i]

            leftLayerData = np.array([n.x for n in leftNeuronLayer])
            leftDepth = leftNeuronLayer[0].y

            for j in range(i+1, coordinates.shape[0]):
                rightNeuronLayer = coordinates[j]
                rightDepth = rightNeuronLayer[0].y

                rightLayerData = np.array([n.x for n in rightNeuronLayer])
                
                positions = self.calculateLinks(rightLayerData, leftLayerData)

                for neuron in zip(positions, np.repeat(leftNeuronLayer, len(rightNeuronLayer)), np.tile(rightNeuronLayer, len(leftNeuronLayer))):
                    cppnInput = [neuron[0][0], leftDepth, neuron[0][1], rightDepth]
                    
                    outputs = cppnPheno.update(cppnInput)
                    output = outputs[0]

                    # if abs(output) >= 0.5:
                    links.append(LinkGene(neuron[1], neuron[2], len(links), output))

        t2 = timer()
        print("time:", t2 - t1)
        
        substrateGenome.links = links

        return substrateGenome
