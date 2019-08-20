from typing import List, Set, Dict, Tuple, Optional

import numpy as np
from copy import deepcopy

from neat.neat import NEAT
from neat.genes import NeuronType, Genome, LinkGene, NeuronGene, MutationRates, Phase, SpeciationType
from neat.mapelites import MapElites, MapElitesConfiguration

class HyperNEAT:

    def __init__(self, numberOfGenomes: int, numOfInputs: int, numOfOutputs: int,
            populationConfiguration: MapElitesConfiguration, mutationRates: MutationRates=MutationRates()):
        
        # CPPNs take 4 inputs, gotta move this somewhere else
        cppnInputs: int = 4
        cppnOutputs: int = 1
        self.neat = NEAT(numberOfGenomes, cppnInputs, cppnOutputs, populationConfiguration)

        # Substrate
        nrOfLayers: int = 3
        hiddenLayersWidth: int = numOfInputs

        self.substrateNeurons: List[NeuronGene] = []
        for y in np.linspace(-1.0, 1.0, num=nrOfLayers):
                
            if y == -1.0:
                for x in np.linspace(-1.0, 1.0, num=numOfInputs):
                    self.substrateNeurons.append(NeuronGene(NeuronType.INPUT, -1, y, x))        

            elif y == 1.0:
                for x in np.linspace(-1.0, 1.0, num=numOfOutputs):
                    self.substrateNeurons.append(NeuronGene(NeuronType.OUTPUT, -1, y, x))
            
            else:
                for x in np.linspace(-1.0, 1.0, num=hiddenLayersWidth):
                    self.substrateNeurons.append(NeuronGene(NeuronType.HIDDEN, -1, y, x))

    def getCandidate(self):
        return self.neat.population.reproduce()

    def updateCandidate(self, candidate, fitness, features) -> bool:
        return self.neat.population.updateArchive(candidate, fitness, features)

    def createSubstrate(self, cppn):
        cppn = self.neat.population.reproduce()

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
                
                coordsInput = [neuron.x, neuron.y, otherNeuron.x, otherNeuron.y]
                output = cppnPheno.update(coordsInput)[0]
                if output >= 0.2 or output <= -0.2:
                    links.append(LinkGene(neuron, otherNeuron, -1, output))

        substrateGenome.links = links

        return substrateGenome