from typing import List, Tuple

import neat.genes as genes
from neat.genes import Genome, NeuronGene, LinkGene
from neat.innovations import Innovations
from neat.utils import fastCopy, debug_print
from neat.population import Population, PopulationConfiguration, PopulationUpdate

import math
import random
import numpy as np
from copy import deepcopy, copy
from icontract import require


class Species:
    numGensAllowNoImprovement = 25

    def __init__(self, speciesID: int, leader: genes.Genome):
        self.ID: int = speciesID

        self.members: List[genes.Genome] = []
        self.addMember(leader)
        self.leader: genes.Genome = leader

        self.age: int = 0
        self.numToSpawn: int = 0


        self.youngAgeThreshold: int = 10
        self.youngAgeBonus: float = 1.5
        self.oldAgeThreshold: int = 50
        self.oldAgePenalty: float = 0.5

        self.highestFitness: float = 0.0
        self.generationsWithoutImprovement: int = 0

        # self.milestone: float = leader.milestone

        self.stagnant: bool = False

    def __contains__(self, key: int) -> bool:
        return key in [m.ID for m in self.members]

    def isMember(self, genome: genes.Genome) -> bool:
        return (genome.ID in [m.ID for m in self.members])

    def addMember(self, genome: genes.Genome) -> None:
        self.members.append(genome)
        genome.species = self

    def best(self) -> genes.Genome:
        return max(self.members)


    def spawn(self) -> genes.Genome:
        return random.choice(self.members)

    def adjustFitnesses(self) -> None:
        for m in self.members:
            m.adjustedFitness = m.fitness / len(self.members)


    def becomeOlder(self) -> None:
        self.age += 1

        highestFitness = max([m.fitness for m in self.members])

        # Check if species is stagnant
        if (highestFitness < self.highestFitness):
            self.generationsWithoutImprovement += 1
        else:
            self.generationsWithoutImprovement = 0
            self.highestFitness = highestFitness

        if (self.generationsWithoutImprovement >= self.numGensAllowNoImprovement):
            self.stagnant = True

class SpeciesConfiguration(PopulationConfiguration):
    def __init__(self, population_size: int, n_inputs: int, n_outputs: int):
        self._data = {
            "population_size": population_size,
            "n_inputs": n_inputs, 
            "n_outputs": n_outputs
        }


class SpeciesUpdate(PopulationUpdate):
    def __init__(self, fitness: np.ndarray):
        super().__init__()

        self._data = {
            "fitness": fitness
        }

class SpeciatedPopulation(Population):

    def __init__(self, configuration: SpeciesConfiguration, innovations: Innovations, mutationRates: genes.MutationRates):
        super().__init__(configuration, innovations, mutationRates)

        self.population_size = configuration.population_size
        self.generation: int = 0
        
        self.species: List[Species] = []
        self.speciesNumber: int = 0
        self.averageInterspeciesDistance: float = 0.0

        self.numOfInputs = configuration.n_inputs
        self.numOfOutputs = configuration.n_outputs
        
        baseGenome = self.newGenome()
        firstSpecies = Species(self.speciesNumber, baseGenome)
        self.species.append(firstSpecies)
        
        for i in range(self.population_size - 1):
            newGenome = fastCopy(baseGenome)
            newGenome.ID = self.currentGenomeID
            firstSpecies.addMember(newGenome)
            self.genomes.append(newGenome)
            self.currentGenomeID += 1

        # Temp
        # for g in self.genomes:
        #     for _ in range(50):
        #         g.mutate(mutationRates)

        # self.speciate()

    def __len__(self):
        return len(self.genomes)

    def population_and_fitnesses(self):
        return [{"genome": g, "fitness": g.fitness} for g in self.genomes]

    @require(lambda update: isinstance(update, SpeciesUpdate))
    def updatePopulation(self, update: PopulationUpdate) -> None:
        # Set fitness score to their respective genome
        for index, genome in enumerate(self.genomes):
            genome.fitness = update.fitness[index]

    def calculateSpawnAmount(self) -> None:
        # Age species and remove stagnant species
        for s in self.species:
            if len(self.species) > 1:
                s.becomeOlder()

                if s.stagnant and len(self.species) > 1:
                    self.species.remove(s)
        
        for s in self.species:
            s.adjustFitnesses()

        # multiobjectivePopulation.py:86
        pop_available = self.population_size - self.population_size/5
        half_pop = int(pop_available / 2)
        shared_pop = int(half_pop / len(self.species))

        debug_print("half_pop: {} | shared_pop: {}".format(half_pop, shared_pop))

        species_fitnesses = [np.array([m.adjustedFitness for m in s.members]) for s in self.species]
        sum_of_fitnesses = [0.0]*len(species_fitnesses)
        total_fitnesses = 0.0

        # This translates the fitnesses so the lowest fitness starts at 0
        # It's mostly to prevent there being negative fitnesses
        for i, f in enumerate(species_fitnesses):
            min = np.min(f)

            # Set a minimum of 1 fitness for mathematical reasons
            sum_of_fitnesses[i] = np.sum(f - min) + 1
            total_fitnesses += sum_of_fitnesses[i]

        for i, s in enumerate(self.species):
            spawn_portion = sum_of_fitnesses[i]/total_fitnesses
            s.numToSpawn = shared_pop + int(spawn_portion * half_pop)

            debug_print("Species {} - Spawn: {}".format(s.ID, s.numToSpawn))

    def speciate(self, genome) -> None:

        # if random.random() > 0.01:
        if random.random() > 1.0:
            random.choice(genome.parents).species.addMember(genome)
            return

        species = None
        for s in self.species:
            distance = genome.calculateCompatibilityDistance(s.leader)

            if distance <= self.mutationRates.newSpeciesTolerance:
                species = s
                break

        if species is None:
            self.speciesNumber += 1
            self.species.append(Species(self.speciesNumber, genome))
        else:
            species.addMember(genome)

    def tournament_selection(self, genomes):
        return sorted(genomes, key=lambda x: x.fitness)[0]

    def newGeneration(self) -> None:

        # for s in self.species:
        #     s.leader = random.choice(s.members)
            # s.members = []

        newPop = []
        for s in self.species:
            # s.members.sort(reverse=True, key=lambda x: x.fitness)

            newPop.extend(s.members)

            reproduction_members = copy(s.members)

            # topPercent = int(math.ceil(0.01 * len(s.members)))
            # Grabbing the top 2 performing genomes
            # for topMember in s.members[:topPercent]:
            #     newPop.append(topMember)
            #     s.numToSpawn -= 1

            # Only use the survival threshold fraction to use as parents for the next generation.
            # cutoff = int(math.ceil(0.1 * len(s.members)))
            # Use at least two parents no matter what the threshold fraction result is.
            # cutoff = max(cutoff, 2)
            # s.members = s.members[:cutoff]


            # Generate new baby genome
            to_spawn = max(0, s.numToSpawn - len(s.members))
            debug_print("Spawning {} for species {}".format(to_spawn, s.ID))
            for i in range(to_spawn):
                baby: genes.Genome = None

                if random.random() > self.mutationRates.crossoverRate:
                    member = random.choice(reproduction_members)
                    baby = deepcopy(member)
                    baby.mutate(self.mutationRates)
                else:

                    # Tournament selection

                    randomMembers = lambda: [random.choice(reproduction_members) for _ in range(5)]
                    g1 = self.tournament_selection(randomMembers())
                    g2 = self.tournament_selection(randomMembers())

                    # g1 = sorted(randomMembers(), key=lambda x: x.fitness)[0]
                    # g2 = sorted(randomMembers(), key=lambda x: x.fitness)[0]

                    baby = self.crossover(g1, g2)

                self.currentGenomeID += 1
                baby.ID = self.currentGenomeID

                # Find species for new baby
                # self.speciate(baby)
                s.addMember(baby)

                newPop.append(baby)

        #  Remove empty species
        for s in [s for s in self.species if len(s.members) == 0]:
            self.species.remove(s)

        self.genomes = newPop

    def reproduce(self) -> List[genes.Genome]:
        for s in self.species:
            s.leader = min([(m, m.calculateCompatibilityDistance(s.leader)) for m in s.members], key=lambda m: m[1])[0]
            s.members = []
            # for m in s.members:
            #     self.speciate(m)


        for g in self.genomes:
            self.speciate(g)

        self.generation += 1

        self.calculateSpawnAmount()
        self.newGeneration()

        return self.genomes