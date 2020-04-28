from typing import List

import neat.genes as genes
from neat.genes import Genome
from neat.innovations import Innovations
from neat.utils import fastCopy
from neat.populations.population import Population, PopulationConfiguration, PopulationUpdate

import random
import numpy as np
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


        self.num_of_elites = 2

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
            # m.adjustedFitness = m.data["rank"] / len(self.members)
            m.adjustedFitness = m.fitness / len(self.members)


    def becomeOlder(self, allowStagnation) -> None:
        self.age += 1

        highestFitness = max([m.fitness for m in self.members])

        if not allowStagnation:
            return

        # Check if species is stagnant
        if (highestFitness <= self.highestFitness):
            self.generationsWithoutImprovement += 1
        else:
            self.generationsWithoutImprovement = 0
            self.highestFitness = highestFitness

        if (self.generationsWithoutImprovement >= self.numGensAllowNoImprovement):
            self.stagnant = True

class SpeciesConfiguration(PopulationConfiguration):
    def __init__(self, population_size: int, n_inputs: int, n_outputs: int):
        super().__init__(population_size, n_inputs, n_outputs)

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
            # newGenome = deepcopy(baseGenome)
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
        best_leaders = []

        # Best species
        best_species = None
        best_fitness = -1000.0
        for s in self.species:
            # max_rank = np.max([m.data["rank"] for m in s.members])

            max_fitness = np.max([m.fitness for m in s.members])
            # best_leaders.append(max_rank)

            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_species = s

        # best_leaders = set(best_leaders)

        # Age species and remove stagnant species
        for s in self.species:
            # if s != best_species:
            s.becomeOlder(s != best_species)

            if s.stagnant and len(self.species) > 1:
                self.species.remove(s)
        
        for s in self.species:
            s.adjustFitnesses()

        # multiobjectivePopulation.py:86
        # pop_available = self.population_size - self.population_size/5
        pop_available = self.population_size
        half_pop = int(pop_available / 2)
        shared_pop = int(half_pop / len(self.species))

        # debug_print("half_pop: {} | shared_pop: {}".format(half_pop, shared_pop))

        # species_fitnesses = [m.adjustedFitness for s in self.species for m in s.members]
        species_fitnesses = [np.array([m.adjustedFitness for m in s.members]) for s in self.species]

        sum_of_fitnesses = [0.0]*len(species_fitnesses)
        total_fitnesses = 0.0

        # This translates the fitnesses so the lowest fitness starts at 0
        # It's mostly to prevent there being negative fitnesses
        min_fitness = 1000.0
        for fitnesses in species_fitnesses:
            min_fitness = min(np.min(fitnesses), min_fitness)
        # for i, f in enumerate(species_fitnesses):
        #     min_fitness = min(np.min(f), min_fitness)

        for i, f in enumerate(species_fitnesses):
            # Set a minimum of 1 fitness for mathematical reasons
            sum_of_fitnesses[i] = np.sum(f - min_fitness) + 1
            total_fitnesses += sum_of_fitnesses[i]

        for i, s in enumerate(self.species):
            spawn_portion = sum_of_fitnesses[i]/total_fitnesses
            s.numToSpawn = shared_pop + int(spawn_portion * half_pop)

            # debug_print("Species {} - Spawn: {}".format(s.ID, s.numToSpawn))

    def speciateAll(self):
        # Wipe species clean for redistrubition of genomes
        for s in self.species:
            s.members = []

        # Distribute genomes to their most fitting species
        for g in self.genomes:
            self.speciate(g)

        # Remove species that have no more members
        self.purgeSpecies()

        # Determine new species leader from the new members
        for s in self.species:
            distances = [m.calculateCompatibilityDistance(s.leader) for m in s.members]
            new_leader_id = np.argmin(distances)
            new_leader = s.members[new_leader_id]

            # debug_print("Species {} - Distance between leaders: {}".format(s.ID, distances[new_leader_id]))

            s.leader = new_leader

        genomes_in_species = np.sum([len(s.members) for s in self.species])
        assert genomes_in_species == len(self.genomes), "There are genomes that are not part of a species. {}/{} are accounted for.".format(genomes_in_species, len(self.genomes))

    def speciate(self, genome) -> None:

        distances = [genome.calculateCompatibilityDistance(s.leader) for s in self.species]
        closest_species = np.argmin(distances)

        if random.random() > 0.01:
            self.species[closest_species].addMember(genome)
            return

        # If it doesn't fit anywhere, create new species
        if distances[closest_species] >= self.mutationRates.newSpeciesTolerance:
            self.speciesNumber += 1
            self.species.append(Species(self.speciesNumber, genome))
        else:
            self.species[closest_species].addMember(genome)

    # Remove species without members
    def purgeSpecies(self):
        for s in [s for s in self.species if len(s.members) == 0]:
            self.species.remove(s)

    def tournament_selection(self, genomes):
        return sorted(genomes, key=lambda x: x.fitness)[0]

    def newGeneration(self) -> List[Genome]:

        for s in self.species:
            new_genomes = []

            s.members.sort(reverse=True, key=lambda x: x.fitness)

            print(s.best())

            # Grabbing the top 2 performing genomes
            for topMember in s.members[:s.num_of_elites]:
                new_genomes.append(topMember)

            # cutoff = max(2, int(len(s.members)*0.2))
            # reproduction_members = s.members[:cutoff]

            reproduction_members = s.members

            # s.members = []
            # max_1 = max([m.fitness for m in reproduction_members])
            # Generate new baby genome
            to_spawn = max(0, s.numToSpawn - len(new_genomes))
            # debug_print("Spawning {} for species {}".format(to_spawn, s.ID))

            for i in range(to_spawn):
                baby: Genome = None

                if random.random() > self.mutationRates.crossoverRate:
                    member = random.choice(reproduction_members)
                    # baby = deepcopy(member)
                    baby = fastCopy(member)
                    baby.mutate(self.mutationRates)
                else:

                    # Tournament selection
                    randomMembers = lambda: [random.choice(reproduction_members) for _ in range(2)]
                    g1 = self.tournament_selection(randomMembers())
                    g2 = self.tournament_selection(randomMembers())

                    baby = self.crossover(g1, g2)

                self.currentGenomeID += 1
                baby.ID = self.currentGenomeID

                new_genomes.append(baby)
                # s.addMember(baby)

            s.members = []
            for g in new_genomes:
                s.addMember(g)

        return [g for s in self.species for g in s.members]


    def reproduce(self) -> None:
        self.generation += 1

        self.speciateAll()

        self.calculateSpawnAmount()

        self.genomes = self.newGeneration()