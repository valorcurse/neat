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
from neat.genes import NeuronType
from neat.genes import CGenome
from neat.genes import SLinkGene
from neat.genes import SNeuronGene
from neat.genes import innovations
from neat.genes import MutationRates
from neat.genes import Phase
from neat.genes import SpeciationType

from neat.phenotypes import CNeuralNet

global innovations

class CSpecies:
    numGensAllowNoImprovement = 20

    def __init__(self, speciesID: int, leader: CGenome):
        self.ID: int = speciesID

        self.members: List[CGenome] = []
        self.addMember(leader)
        self.leader: CGenome = leader

        self.age: int = 0
        self.numToSpawn: int = 0


        self.youngAgeThreshold: int = 10
        self.youngAgeBonus: float = 1.5
        self.oldAgeThreshold: int = 50
        self.oldAgePenalty: float = 0.5

        self.highestFitness: float = 0.0
        self.generationsWithoutImprovement: int = 0

        self.milestone: float = leader.milestone

        self.stagnant: bool = False

    def __contains__(self, key: int) -> bool:
        return key in [m.ID for m in self.members]

    def isMember(self, genome: CGenome) -> bool:
        return (genome.ID in [m.ID for m in self.members])

    def addMember(self, genome: CGenome) -> None:
        self.members.append(genome)
        genome.species = self

    def best(self) -> CGenome:
        return max(self.members)


    def spawn(self) -> CGenome:
        return random.choice(self.members)

    def adjustFitnesses(self) -> None:
        # avgMilestone = np.average([m.milestone for m in self.members])
        
        # self.members = [m for m in self.members if m.milestone >= avgMilestone]

        for m in self.members:
            m.adjustedFitness = m.fitness / len(self.members)

            # if self.age <= self.youngAgeThreshold:
            #     m.adjustedFitness *= self.youngAgeBonus

            # if self.age >= self.oldAgeThreshold:
            #     m.adjustedFitness *= self.oldAgePenalty

    def becomeOlder(self, alone: bool) -> None:
        self.age += 1

        highestFitness = max([m.fitness for m in self.members])

        if alone:
            return

        # Check if species is stagnant
        if (highestFitness < self.highestFitness):
            self.generationsWithoutImprovement += 1
        else:
            self.generationsWithoutImprovement = 0
            self.highestFitness = highestFitness

        if (self.generationsWithoutImprovement >= self.numGensAllowNoImprovement):
            self.stagnant = True

class NEAT:

    def __init__(self, numberOfGenomes: int, numOfInputs: int, numOfOutputs: int, mutationRates: MutationRates=MutationRates(), fullyConnected: bool=False) -> None:

        self.genomes: List[CGenome] = []
        self.phenotypes: List[CNeuralNet] = []
        self.species: List[CSpecies] = []
        self.speciesNumber: int = 0

        self.populationSize: int = numberOfGenomes

        self.generation: int = 0

        self.currentGenomeID: int = 0

        self.phase: Phase = Phase.COMPLEXIFYING
        self.speciationType: SpeciationType = SpeciationType.NOVELTY

        self.mutationRates: MutationRates = mutationRates

        self.averageInterspeciesDistance: float = 0.0
        self.milestone: float = 0.01

        inputs = []
        for n in range(numOfInputs):
            print("\rCreating inputs neurons (" + str(n + 1) + "/" + str(numOfInputs) + ")", end='')
            newInput = innovations.createNewNeuron(0.0, NeuronType.INPUT, fromNeuron = None, toNeuron = None, neuronID = -n-1)
            inputs.append(newInput)

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
        newGenome = CGenome(self.currentGenomeID, inputs, links, numOfInputs, numOfOutputs)
        newGenome.parents = [newGenome]
        for i in range(self.populationSize):
            print("\rCreating genomes (" + str(i + 1) + "/" + str(self.populationSize) + ")", end='')
            self.genomes.append(deepcopy(newGenome))
            self.currentGenomeID += 1

        print("")


        # mpc = self.calculateMPC()
        # mpc = 100
        # self.mpcThreshold: int = mpc + mutationRates.mpcMargin
        # self.lowestMPC: int = mpc
        # self.mpcStagnation: int = 0
        
        # print("mpc", mpc)
        # print("mpc threshold", self.mpcThreshold)

        self.speciate()
        self.epoch([0]*len(self.genomes))


    # def calculateMPC(self):
    #     allMutations = [[n for n in g.neurons] + [l for l in g.links] for g in self.genomes]
    #     nrOfMutations = len([item for sublist in allMutations for item in sublist])
    #     return (nrOfMutations / len(self.genomes))

    def epoch(self, fitnessScores: List[float], novelty: Optional[List[float]] = None) -> None:
        
        if novelty is not None:
            for index, genome in enumerate(self.genomes):
                genome.novelty = novelty[index]

        # Set fitness score to their respesctive genome
        for index, genome in enumerate(self.genomes):
            genome.fitness = fitnessScores[index]

        self.calculateSpawnAmount()
 
        self.reproduce()
        
        self.speciate()

        if len(self.species) > 1:
            totalDistance: float = 0.0
            for s in self.species:
                randomSpecies: CSpecies = random.choice([r for r in self.species if r is not s])

                totalDistance += s.leader.calculateCompatibilityDistance(randomSpecies.leader)
            
            self.averageInterspeciesDistance = max(self.mutationRates.newSpeciesTolerance, (totalDistance/len(self.species))*0.5)

            print("averageInterspeciesDistance: " + str(self.averageInterspeciesDistance))

        # speciesCombinations = list(itertools.permutations(self.species))
        # print(speciesCombinations)
        # if len(speciesCombinations) > 1:
        #     totalDistance: float = 0.0
        #     for combination in speciesCombinations:
        #         totalDistance += combination[0].leader.calculateCompatibilityDistance(combination[1].leader)
        

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
        for genome in self.genomes:
            newPhenotypes.append(genome.createPhenotype())

        self.generation += 1

        self.phenotypes = newPhenotypes

    def speciate(self) -> None:


        # Find best leader for species from the new population
        unspeciated = list(range(0, len(self.genomes)))
        for s in self.species:
            compareMember = s.leader

            s.members = []

            candidates: List[Tuple[float, int]] = []
            for i in unspeciated:
                genome = self.genomes[i]

                distance = genome.calculateCompatibilityDistance(compareMember)

                if (distance < max(self.mutationRates.newSpeciesTolerance, self.averageInterspeciesDistance)):
                    candidates.append((distance, i))

            if len(candidates) == 0:
                self.species.remove(s)
                continue

            _, bestCandidate = min(candidates, key=lambda x: x[0])

            s.leader = self.genomes[bestCandidate]
            s.members.append(s.leader)
            unspeciated.remove(bestCandidate)

        # Distribute genomes to their closest species
        for i in unspeciated:
            genome = self.genomes[i]

            # closestDistance = self.mutationRates.newSpeciesTolerance
            closestDistance = max(self.mutationRates.newSpeciesTolerance, self.averageInterspeciesDistance)
            closestSpecies = None
            for s in self.species:
                distance = genome.calculateCompatibilityDistance(s.leader)
                # If genome falls within tolerance of species
                if (distance < closestDistance):
                    closestDistance = distance
                    closestSpecies = s

            if (closestSpecies is not None): # If found a compatible species
                # closestSpecies.members.append(genome)
                closestSpecies.addMember(genome)

            else: # Else create a new species
                chance: float = random.random()

                parentSpecies: Optional[CSpecies] = random.choice(genome.parents).species

                # if (chance >= 0.1) and parentSpecies is not None:
                #     parentSpecies.addMember(genome)
                # else:
                self.speciesNumber += 1
                self.species.append(CSpecies(self.speciesNumber, genome))


    def reproduce(self) -> None:
        newPop = []
        for s in self.species:
            # numToSpawn = s.numToSpawn

            # members = deepcopy(s.members)
            s.members.sort(reverse=True, key=lambda x: x.fitness)

            topPercent = int(math.ceil(0.01 * len(s.members)))
            print("topPercent: " + str(topPercent))
            # Grabbing the top 2 performing genomes
            for topMember in s.members[:topPercent]:
                newPop.append(topMember)
                # s.members.remove(topMember)
                s.numToSpawn -= 1

            # Only select members who got past the milestone
            # s.members = [m for m in s.members if m.milestone >= s.milestone]

            # s.members.sort(reverse=True, key=lambda x: x.milestone)
            # Only use the survival threshold fraction to use as parents for the next generation.
            cutoff = int(math.ceil(0.1 * len(s.members)))
            # Use at least two parents no matter what the threshold fraction result is.
            cutoff = max(cutoff, 2)
            s.members = s.members[:cutoff]

            # if (s.numToSpawn <= 0 or len(s.members) <= 0):
            #     continue

            for i in range(s.numToSpawn):
                baby: Optional[CGenome] = None

                # if (self.phase == Phase.PRUNING or random.random() > self.mutationRates.crossoverRate):
                if (random.random() > self.mutationRates.crossoverRate):
                    baby = deepcopy(random.choice(s.members))
                    baby.mutate(Phase.COMPLEXIFYING, self.mutationRates)
                else:
                    # g1 = random.choice(members)
                    # g2 = random.choice(members)
                    
                    # Tournament selection
                    randomMembers = [random.choice(s.members) for _ in range(5)]
                    g1 = sorted(randomMembers, key=lambda x: x.fitness)[0]
                    g2 = sorted(randomMembers, key=lambda x: x.fitness)[0]
                    
                    baby = self.crossover(g1, g2)

                self.currentGenomeID += 1
                baby.ID = self.currentGenomeID

                # baby.mutate(Phase.COMPLEXIFYING, self.mutationRates)
                # baby.mutate(self.phase, self.mutationRates)

                newPop.append(baby)

        self.genomes = newPop

    def calculateSpawnAmount(self) -> None:
        # Remove stagnant species
        # if self.phase == Phase.COMPLEXIFYING:
        for s in self.species:
            s.becomeOlder(len(self.species) == 1)

            if s.stagnant and len(self.species) > 1:
                self.species.remove(s)
        
        for s in self.species:
            s.adjustFitnesses()
        
        allFitnesses = sum([m.adjustedFitness for spc in self.species for m in spc.members])
        for s in self.species:
            sumOfFitnesses: float = sum([m.adjustedFitness for m in s.members])

            portionOfFitness: float = 1.0 if allFitnesses == 0.0 and sumOfFitnesses == 0.0 else sumOfFitnesses/allFitnesses
            s.numToSpawn = int(self.populationSize * portionOfFitness)

    def crossover(self, mum: CGenome, dad: CGenome) -> CGenome:
        
        best = None
        if (mum.fitness == dad.fitness):
            if (len(mum.links) == len(dad.links)):
                best = random.choice([mum, dad])
            else:
                best = mum if len(mum.links) < len(dad.links) else dad
        else:
            best = mum if mum.fitness > dad.fitness else dad

        # Copy input and output neurons
        babyNeurons = [deepcopy(n) for n in best.neurons
                       if (n.neuronType in [NeuronType.INPUT, NeuronType.OUTPUT])]

        combinedIndexes = list(set(
            [l.innovationID for l in mum.links] + [l.innovationID for l in dad.links]))
        combinedIndexes.sort()
        
        mumDict: Dict[int, SLinkGene] = {l.innovationID: l for l in mum.links}
        dadDict: Dict[int, SLinkGene] = {l.innovationID: l for l in dad.links}

        # print("-------------------------------------------------")
        babyLinks: List[SLinkGene] = []
        for i in combinedIndexes:
            mumLink: Optional[SLinkGene] = mumDict.get(i)
            dadLink: Optional[SLinkGene] = dadDict.get(i)
            
            if (mumLink is None):
                if (dadLink is not None and best == dad):
                    babyLinks.append(deepcopy(dadLink))

            elif (dadLink is None):
                if (mumLink is not None and best == mum):
                    babyLinks.append(deepcopy(mumLink))

            else:
                babyLinks.append(random.choice([mumLink, dadLink]))

        for link in babyLinks:

            if (link.fromNeuron.innovationID not in [n.innovationID for n in babyNeurons]):
                babyNeurons.append(deepcopy(link.fromNeuron))

            if (link.toNeuron.innovationID not in [n.innovationID for n in babyNeurons]):
                babyNeurons.append(deepcopy(link.toNeuron))

        babyNeurons.sort(key=lambda x: x.splitY, reverse=False)

        return CGenome(self.currentGenomeID, babyNeurons, babyLinks, best.inputs, best.outputs, [mum, dad])

