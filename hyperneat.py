from typing import List, Set, Dict, Tuple, Optional

import numpy as np
from copy import deepcopy

from neat.neat import NEAT
from neat.types import NeuronType
from neat.phenotypes import Phenotype
from neat.population import PopulationUpdate, PopulationConfiguration
from neat.genes import Genome, LinkGene, NeuronGene, MutationRates, Phase, SpeciationType


class HyperNEAT(NEAT):

    def __init__(self, population_configuration: PopulationConfiguration, mutation_rates: MutationRates=MutationRates()):
        
        # CPPNs take 4 inputs, gotta move this somewhere else
        nrOfLayers: int = 3
        self.layers = np.linspace(-1.0, 1.0, num=nrOfLayers)
        cppnInputs: int = 4
        cppnOutputs: int = nrOfLayers - 1

        self.neat = NEAT(population_configuration, mutation_rates)

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

    # def getCandidate(self):
    #     return self.neat.population.reproduce()

    # def updateCandidate(self, candidate, fitness, features) -> bool:
    #     return self.neat.population.updateArchive(candidate, fitness, features)

    def epoch(self, update_data: PopulationUpdate) -> List[Phenotype]:
        self.neat.population.updatePopulation(update_data)

        cppns = self.neat.population.reproduce()

        return [self.createSubstrate(c) for c in cppns]


    def createSubstrate(self, cppn):
        cppnPheno = cppn.createPhenotype()

        nrOfInputs = len([n for n in self.substrateNeurons if n.neuronType == NeuronType.INPUT])
        nrOfOutputs = len([n for n in self.substrateNeurons if n.neuronType == NeuronType.OUTPUT])

        substrateGenome: Genome = Genome(cppn.ID, nrOfInputs, nrOfOutputs, neurons=deepcopy(self.substrateNeurons))
        links = []
        for i in range(0, len(substrateGenome.neurons) - 1):
            neuron = substrateGenome.neurons[i]
            
            for j in range(i+1, len(substrateGenome.neurons)):
                otherNeuron = substrateGenome.neurons[j]

                if (neuron.y == otherNeuron.y):
                    continue
                
                layer = np.where(self.layers == neuron.y)[0][0] - 1
                
                coordsInput = [neuron.x, neuron.y, otherNeuron.x, otherNeuron.y]
                outputs = cppnPheno.update(coordsInput)

                output = cppnPheno.update(coordsInput)[layer]
                # if output >= 0.2 or output <= -0.2:
                links.append(LinkGene(neuron, otherNeuron, -1, output))

        substrateGenome.links = links

        return substrateGenome