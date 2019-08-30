from typing import List, Set, Dict, Tuple, Optional

import numpy as np
from copy import deepcopy

from neat.neat import NEAT
from neat.utils import fastCopy, cartesian
from neat.types import NeuronType
from neat.phenotypes import Phenotype
from neat.population import PopulationUpdate, PopulationConfiguration
from neat.genes import Genome, LinkGene, NeuronGene, MutationRates, Phase, SpeciationType

from numba import jit, njit, int32, float32

from itertools import groupby, product

from timeit import default_timer as timer



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
        hiddenLayersWidth: int = population_configuration.n_inputs

        self.substrateNeurons: List[NeuronGene] = []
        for y in self.layers:
                
            if y == -1.0:
                for x in np.linspace(-1.0, 1.0, num=population_configuration.n_inputs):
                    self.substrateNeurons.append(NeuronGene(NeuronType.INPUT, -1, y, x))        

            elif y == 1.0:
                for x in np.linspace(-1.0, 1.0, num=population_configuration.n_outputs):
                    self.substrateNeurons.append(NeuronGene(NeuronType.OUTPUT, -1, y, x))
            
            else:
                for x in np.linspace(-1.0, 1.0, num=hiddenLayersWidth):
                    self.substrateNeurons.append(NeuronGene(NeuronType.HIDDEN, -1, y, x))


        population_configuration._data["n_inputs"] = 4
        population_configuration._data["n_outputs"] = 1
        self.neat = NEAT(population_configuration, mutation_rates)

    # def getCandidate(self):
    #     return self.neat.population.reproduce()

    # def updateCandidate(self, candidate, fitness, features) -> bool:
    #     return self.neat.population.updateArchive(candidate, fitness, features)

    def epoch(self, update_data: PopulationUpdate) -> List[Phenotype]:
        self.neat.population.updatePopulation(update_data)

        cppns = self.neat.population.reproduce()
        substrates = []
        for i, c in enumerate(cppns):
            # print(i)
            # t1 = timer()
            substrates.append(self.createSubstrate(c).createPhenotype())
            print("\r%d/%d"%(i, len(cppns)), end='')
            # t2 = timer()
            # print("time:", t2 - t1)


        return substrates


    # @njit(float32[:, :](float32[:, :]))
    # @jit(int32(float32[:]))
    @staticmethod
    def calculateLinks(x, y):
        return cartesian(x, y)

    def createSubstrate(self, cppn):
        cppnPheno = cppn.createPhenotype()

        nrOfInputs = self.n_inputs
        nrOfOutputs = self.n_outputs


        substrateGenome: Genome = Genome(cppn.ID, nrOfInputs, nrOfOutputs, neurons=fastCopy(self.substrateNeurons))

        layers = [list(g) for k, g in groupby(substrateGenome.neurons, lambda n: n.y)]

        # print(layers)
        coordinates = []
        for l in layers:
            singleLayer = np.array([n for n in l])
            coordinates.append(singleLayer)
        coordinates = np.array(coordinates)


        
        links = []
        for i in range(coordinates.shape[0] - 1):
            leftNeuronLayer = coordinates[i]

            leftLayerData = np.array([(n.x, n.y) for n in leftNeuronLayer])

            for j in range(i+1, coordinates.shape[0]):
                rightNeuronLayer = coordinates[j]

                rightLayerData = np.array([(n.x, n.y) for n in rightNeuronLayer])
                
                positions = self.calculateLinks(rightLayerData, leftLayerData)

                for neuron in zip(positions, leftNeuronLayer, rightNeuronLayer):
                    # print(neuron)
                    outputs = cppnPheno.update(neuron[0])
                    output = outputs[0]
                    
                    links.append(LinkGene(neuron[1], neuron[2], -1, output))

        # print(np.array(products))
 
        # links = self.calculateLinks(products)
        # print(links)
        # links = []
        # for l in range(len(layers)-1):
        #     layer = layers[l]
            
        #     for i in range(len(layer)):
        #         print(layer[0])
        #         neuron = layer[i]
                
        #         nextLayer = layers[l+1]

        #         for j in range(len(nextLayer)):
        #             otherNeuron = nextLayer[j]

        #             if (neuron.y == otherNeuron.y):
        #                 continue
                    
        #             # print(i, j)
        #             layer = np.where(self.layers == neuron.y)[0][0] - 1
                    
        #             coordsInput = [neuron.x, neuron.y, otherNeuron.x, otherNeuron.y]
        #             outputs = cppnPheno.update(coordsInput)

        #             output = cppnPheno.update(coordsInput)[layer]
        #             # if output >= 0.2 or output <= -0.2:
        #             links.append(LinkGene(neuron, otherNeuron, -1, output))

        substrateGenome.links = links

        return substrateGenome
