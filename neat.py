from typing import List, Set, Dict, Tuple, Optional

import random
from random import randint

import math
import numpy as np
from enum import Enum

from copy import copy
from copy import deepcopy
import itertools
from operator import attrgetter

from prettytable import PrettyTable

# import genes
from neat.genes import NeuronType, Genome, LinkGene, NeuronGene, innovations, MutationRates, Phase, SpeciationType
from neat.phenotypes import CNeuralNet
# from neat.population import PopulationConfiguration
# from neat.defaultPopulation import DefaultPopulation
from neat.mapelites import MapElites, MapElitesConfiguration

global innovations

class NEAT:

    def __init__(self, numberOfGenomes: int, numOfInputs: int, numOfOutputs: int, populationConfiguration: MapElitesConfiguration,
        mutationRates: MutationRates=MutationRates(), fullyConnected: bool=False) -> None:

        self.mutationRates: MutationRates = mutationRates
        self.phenotypes: List[CNeuralNet] = []
        

        self.generation: int = 0

        self.phase: Phase = Phase.COMPLEXIFYING
        self.speciationType: SpeciationType = SpeciationType.NOVELTY

        self.milestone: float = 0.01

        inputs = []
        for n in range(numOfInputs):
            print("\rCreating inputs neurons (" + str(n + 1) + "/" + str(numOfInputs) + ")", end='')
            newInput = innovations.createNewNeuron(0.0, NeuronType.INPUT, fromNeuron = None, toNeuron = None, neuronID = -n-1)
            inputs.append(newInput)

        inputs.append(innovations.createNewNeuron(1.0, NeuronType.BIAS, fromNeuron = None, toNeuron = None, neuronID = -len(inputs)-1))
        print("")

        outputs = []
        for n in range(numOfOutputs):
            print("\rCreating output neurons (" + str(n + 1) + "/" + str(numOfOutputs) + ")", end='')
            newOutput = innovations.createNewNeuron(1.0, NeuronType.OUTPUT, fromNeuron = None, toNeuron = None, neuronID = -numOfInputs-n-1)
            outputs.append(newOutput)

        print("")

        links = []
        if fullyConnected:
            for input in inputs:
                for output in outputs:
                    new_link = innovations.createNewLink(input, output, True, 1.0)
                    links.append(new_link)

        inputs.extend(outputs)
        
        # self.population = DefaultPopulation(numberOfGenomes, mutationRates)
        self.population = MapElites(numberOfGenomes, inputs, outputs, mutationRates, populationConfiguration)
        self.population.initiate(inputs, links, numOfInputs, numOfOutputs)


        print("")


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

    def getCandidate(self) -> Genome:
        return self.population.reproduce()

    def updateCandidate(self, candidate, fitness, features) -> bool:
        return self.population.updateArchive(candidate, fitness, features)