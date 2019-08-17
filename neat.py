from typing import List, Set, Dict, Tuple, Optional

import random
from random import randint

import math
import numpy as np
from enum import Enum

import itertools
from copy import copy
from copy import deepcopy
from operator import attrgetter
from collections import OrderedDict

from prettytable import PrettyTable

import pickle

# import genes
from neat.genes import NeuronType, Genome, LinkGene, NeuronGene, innovations, MutationRates, Phase, SpeciationType
from neat.phenotypes import CNeuralNet
# from neat.population import PopulationConfiguration
# from neat.defaultPopulation import DefaultPopulation
from neat.mapelites import MapElites, MapElitesConfiguration

global innovations

class NEAT:

    def __init__(self, numberOfGenomes: int, numOfInputs: int, numOfOutputs: int,
        populationConfiguration: MapElitesConfiguration, mutationRates: MutationRates=MutationRates()) -> None:

        self.mutationRates: MutationRates = mutationRates
        self.phenotypes: List[CNeuralNet] = []
        

        self.generation: int = 0

        self.phase: Phase = Phase.COMPLEXIFYING
        self.speciationType: SpeciationType = SpeciationType.NOVELTY

        self.milestone: float = 0.01

        # CPPNs take 4 inputs, gotta move this somewhere else


        inputs = []
        cppnInputs = 4
        for n in range(cppnInputs):
            print("\rCreating inputs neurons (" + str(n + 1) + "/" + str(cppnInputs) + ")", end='')
            
            newInput = innovations.createNewNeuron(-1.0, NeuronType.INPUT, fromNeuron = None, toNeuron = None, neuronID = -n-1)
            inputs.append(newInput)

        inputs.append(innovations.createNewNeuron(-1.0, NeuronType.BIAS, fromNeuron = None, toNeuron = None, neuronID = -len(inputs)-1))
        print("")

        # outputs = [innovations.createNewNeuron(1.0, NeuronType.OUTPUT, fromNeuron = None, toNeuron = None, neuronID = -numOfInputs-n-1)]
        outputs = []
        cppnOutputs = 1
        for n in range(cppnOutputs):
            print("\rCreating output neurons (" + str(n + 1) + "/" + str(cppnOutputs) + ")", end='')
            newOutput = innovations.createNewNeuron(1.0, NeuronType.OUTPUT, fromNeuron = None, toNeuron = None, neuronID = -numOfInputs-n-1)
            outputs.append(newOutput)

        print("")

    
        links: List[LinkGene] = []

        # self.population = DefaultPopulation(numberOfGenomes, mutationRates)
        self.population = MapElites(numberOfGenomes, inputs, outputs, mutationRates, populationConfiguration)
        # self.population.initiate(inputs + outputs, links, cppnInputs, cppnOutputs)

        # Substrate

        nrOfLayers: int = 3
        hiddenLayersWidth: int = numOfInputs

        self.substrateNeurons: List[NeuronGene] = []
        for y in np.linspace(-1.0, 1.0, num=nrOfLayers):
                
            if y == -1.0:
                for x in np.linspace(-1.0, 1.0, num=numOfInputs):
                    self.substrateNeurons.append(NeuronGene(NeuronType.INPUT, -1, -1, y, x))        

            elif y == 1.0:
                for x in np.linspace(-1.0, 1.0, num=numOfOutputs):
                    self.substrateNeurons.append(NeuronGene(NeuronType.OUTPUT, -1, -1, y, x))
            
            else:
                for x in np.linspace(-1.0, 1.0, num=hiddenLayersWidth):
                    self.substrateNeurons.append(NeuronGene(NeuronType.HIDDEN, -1, -1, y, x))


        # mpc = self.calculateMPC()
        # mpc = 100
        # self.mpcThreshold: int = mpc + mutationRates.mpcMargin
        # self.lowestMPC: int = mpc
        # self.mpcStagnation: int = 0
        
        # print("mpc", mpc)
        # print("mpc threshold", self.mpcThreshold)

        # self.population.speciate()
        # self.epoch([0]*len(self.population.genomes))


    # def calculateMPC(self):
    #     allMutations = [[n for n in g.neurons] + [l for l in g.links] for g in self.genomes]
    #     nrOfMutations = len([item for sublist in allMutations for item in sublist])
    #     return (nrOfMutations / len(self.genomes))

    def epoch(self, fitnessScores: List[float], novelty: Optional[List[float]] = None) -> None:
        
        if novelty is not None:
            for index, genome in enumerate(self.population.genomes):
                genome.novelty = novelty[index]

        # Set fitness score to their respesctive genome
        for index, genome in enumerate(self.population.genomes):
            genome.fitness = fitnessScores[index]

        self.calculateSpawnAmount()
 
        self.population.reproduce()
        
        self.population.speciate()

        # mpc = self.calculateMPC()

        # if self.phase == Phase.PRUNING:
        #     if mpc < self.lowestMPC:
        #         self.lowestMPC = mpc
        #         self.mpcStagnation = 0
        #     else:
        #         self.mpcStagnation += 1

        #     if self.mpcStagnation >= 10:
        #         self.phase = Phase.COMPLEXIFYING
        #         self.mpcThreshold = mpc + self.mutationRates.mpcMargin

        # elif self.phase == Phase.COMPLEXIFYING:
        #     if mpc >= self.mpcThreshold:
        #         self.phase = Phase.PRUNING
        #         self.mpcStagnation = 0
        #         self.lowestMPC = mpc


        newPhenotypes = []
        for genome in self.population.genomes:
            newPhenotypes.append(genome.createPhenotype())

        self.generation += 1

        self.phenotypes = newPhenotypes


    def calculateSpawnAmount(self) -> None:
        # Remove stagnant species
        # if self.phase == Phase.COMPLEXIFYING:
        for s in self.population.species:
            s.becomeOlder(len(self.population.species) == 1)

            if s.stagnant and len(self.population.species) > 1:
                self.population.species.remove(s)
        
        for s in self.population.species:
            s.adjustFitnesses()
        
        allFitnesses = sum([m.adjustedFitness for spc in self.population.species for m in spc.members])
        for s in self.population.species:
            sumOfFitnesses: float = sum([m.adjustedFitness for m in s.members])

            portionOfFitness: float = 1.0 if allFitnesses == 0.0 and sumOfFitnesses == 0.0 else sumOfFitnesses/allFitnesses
            s.numToSpawn = int(self.population.populationSize * portionOfFitness)

    def getCandidate(self):
        print("########################### cppn ###########################")
        cppn = self.population.reproduce()

        print("Links: %d"%len(cppn.links))

        cppnPheno = cppn.createPhenotype()
        print("############################################################")

        substrateGenome: Genome = Genome(cppn.ID, self.substrateNeurons, [])
        links = []
        for i in range(0, len(substrateGenome.neurons) - 1):
            neuron = substrateGenome.neurons[i]
            
            for j in range(i+1, len(substrateGenome.neurons)):
                otherNeuron = substrateGenome.neurons[j]

                if (neuron.y == otherNeuron.y):
                    continue
                
                coordsInput = [neuron.x, neuron.y, otherNeuron.x, otherNeuron.y]
                output = cppnPheno.update(coordsInput)

                if output[0] >= 0.2:
                    # print(output)
                    links.append(LinkGene(neuron, otherNeuron, -1, output))

        
        # print("links: %d"%(len(links)))

        substrateGenome.links = links

        return [
            cppn,
            substrateGenome
        ]

    def updateCandidate(self, candidate, fitness, features) -> bool:
        return self.population.updateArchive(candidate, fitness, features)