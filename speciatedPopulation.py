from typing import List, Optional, Tuple

from neat.genes import Genome
from neat.utils import fastCopy
from neat.neat import MutationRates
from neat.innovations import Innovations
from neat.population import Population, PopulationConfiguration

import math
import random

from neat.genes import Phase

class Species:
    numGensAllowNoImprovement = 20

    def __init__(self, speciesID: int, leader: Genome):
        self.ID: int = speciesID

        self.members: List[Genome] = []
        self.addMember(leader)
        self.leader: Genome = leader

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

    def isMember(self, genome: Genome) -> bool:
        return (genome.ID in [m.ID for m in self.members])

    def addMember(self, genome: Genome) -> None:
        self.members.append(genome)
        genome.species = self

    def best(self) -> Genome:
        return max(self.members)


    def spawn(self) -> Genome:
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

class SpeciesConfiguration(PopulationConfiguration):
    def __init__(self, population_size: int):
        self.population_size = population_size

class SpeciatedPopulation(Population):

    def __init__(self, innovations: Innovations, mutationRates: MutationRates, configuration: SpeciesConfiguration):
        super().__init__(innovations, mutationRates)

        self.population_size = configuration.population_size
        self.generation: int = 0
        self.species: List[Species] = []
    

    # def initiate(self, neurons: List[NeuronGene], links: List[LinkGene], numOfInputs: int, numOfOutputs: int, parents=[]):
        
    #     self.numOfInputs = numOfInputs
    #     self.numOfOutputs = numOfOutputs

    #     for i in range(self.population_size):
    #         genome = self.newGenome(neurons, links)
    #         genome.parents = [genome]

    def updateFitness(self, fitness: List[float]) -> None:
        pass

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
            s.numToSpawn = int(self.population_size * portionOfFitness)


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

                parentSpecies: Optional[Species] = random.choice(genome.parents).species

                # if (chance >= 0.1) and parentSpecies is not None:
                #     parentSpecies.addMember(genome)
                # else:
                self.speciesNumber += 1
                self.species.append(Species(self.speciesNumber, genome))

        # Calculate average interspecies distance
        if len(self.species) > 1:
            totalDistance: float = 0.0
            for s in self.species:
                randomSpecies: Species = random.choice([r for r in self.species if r is not s])

                totalDistance += s.leader.calculateCompatibilityDistance(randomSpecies.leader)
            
            self.averageInterspeciesDistance = max(self.mutationRates.newSpeciesTolerance, (totalDistance/len(self.species)))

            print("averageInterspeciesDistance: " + str(self.averageInterspeciesDistance))

    def newGeneration(self) -> None:

        newPop = []
        for s in self.species:
            s.members.sort(reverse=True, key=lambda x: x.fitness)

            topPercent = int(math.ceil(0.01 * len(s.members)))
            print("topPercent: " + str(topPercent))
            # Grabbing the top 2 performing genomes
            for topMember in s.members[:topPercent]:
                newPop.append(topMember)
                # s.members.remove(topMember)
                s.numToSpawn -= 1

            # Only use the survival threshold fraction to use as parents for the next generation.
            cutoff = int(math.ceil(0.1 * len(s.members)))
            # Use at least two parents no matter what the threshold fraction result is.
            cutoff = max(cutoff, 2)
            s.members = s.members[:cutoff]

            for i in range(s.numToSpawn):
                baby: Genome = None

                if (random.random() > self.mutationRates.crossoverRate):
                    member = random.choice(s.members)
                    baby = fastCopy(member)
                    baby.mutate(self.mutationRates)
                else:
                    # Tournament selection
                    randomMembers = lambda: [random.choice(s.members) for _ in range(5)]
                    g1 = sorted(randomMembers(), key=lambda x: x.fitness)[0]
                    g2 = sorted(randomMembers(), key=lambda x: x.fitness)[0]
                    
                    baby = self.crossover(g1, g2)

                self.currentGenomeID += 1
                baby.ID = self.currentGenomeID

                newPop.append(baby)

        self.genomes = newPop

    def reproduce(self) -> List[Genome]:
        self.generation += 1

        self.calculateSpawnAmount()
        self.newGeneration()
        self.speciate()

        return self.genomes