from typing import List, Tuple
from icontract import require

import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import random
import math

from copy import copy, deepcopy

# import neat.genes as genes
from neat.utils import debug_print
from neat.aurora import Aurora
from neat.innovations import Innovations
from neat.population import PopulationUpdate
from neat.noveltySearch import NoveltySearch
from neat.population import PopulationConfiguration
from neat.genes import NeuronGene, LinkGene, MutationRates, Genome
from neat.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration, SpeciesUpdate

class MOConfiguration(PopulationConfiguration):
    def __init__(self, population_size: int, n_inputs: int, n_outputs: int, behavior_dimensions: int = 0):
        super().__init__(population_size, n_inputs, n_outputs)

        self._data["behavior_dimensions"] = behavior_dimensions

class MOUpdate(SpeciesUpdate):
    def __init__(self, fitness: np.ndarray, features):
        super().__init__(fitness)

        self._data["features"] = features

class MOPopulation(SpeciatedPopulation):

    def __init__(self, configuration: SpeciesConfiguration, innovations: Innovations, mutationRates: MutationRates):
        super().__init__(configuration, innovations, mutationRates)

        # encoding_dim = 8
        # self.behavior_dimensions = configuration.behavior_dimensions

        # Temp way to trigger aurora refinement
        self.epochs = 0
        # self.aurora = Aurora(encoding_dim, self.behavior_dimensions)
        #
        # self.novelty_search = NoveltySearch(encoding_dim)

        # self.use_local_competition = False

    def newGenome(self, neurons: List[NeuronGene] = [], links: List[LinkGene] = [], parents=[]):

        genome = super().newGenome(neurons, links, parents)
        genome.data['rank'] = 0
        genome.data['crowding_distance'] = 0

        return genome

    @require(lambda update: isinstance(update, MOUpdate))
    def updatePopulation(self, update: PopulationUpdate) -> None:

        # features = self.aurora.characterize(update.behaviors)

        super().updatePopulation(update)

        # fitnesses = np.array([g.fitness for g in self.genomes])

        # novelties = None
        # if self.use_local_competition:
        #     novelties, fitnesses = self.novelty_search.calculate_local_competition(features, fitnesses)
        # else:
        #     novelties = self.novelty_search.calculate_novelty(features, fitnesses)
        #
        # for genome, novelty in zip(self.genomes, novelties):
        #     genome.novelty = novelty

        # Fitness and novelty are made negative, because the non dominated sorting
        # is a minimalization algorithm
        # points = list(zip(-fitnesses, -novelties))

        inverse_fitness = -update.fitness
        inverse_features = [-f for f in update.features]

        if len(inverse_features) == 0:
            # Padding zeroes
            z = np.zeros((len(inverse_fitness)))
            points = list(zip(*[inverse_fitness, z]))
        else:
            combined = inverse_fitness + inverse_features
            points = list(zip(*combined))

        points = np.array(points)

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=points)

        for i in range(len(ndf)):
            front_genomes = [self.genomes[j] for j in ndf[i]]
            front_points = [points[j] for j in ndf[i]]
            front_ranks = [ndr[j] for j in ndf[i]]

            crowding_distances = pg.crowding_distance(front_points) if len(front_points) > 1 else np.zeros((len(front_points)))
            for g, d, r in zip(front_genomes, crowding_distances, front_ranks):
                g.data['crowding_distance'] = d
                g.data['rank'] = r

        # super().updatePopulation(update)

        self.epochs += 1

    def tournament_selection(self, genomes):
        return sorted(genomes, key=lambda x: (x.data['rank'], -x.data['crowding_distance']))[0]

    def refine_behaviors(self, evaluate):
        plt.figure(1)
        plt.clf()

        phenotypes = self.create_phenotypes()
        print("Refining AURORA...")
        self.aurora.refine(phenotypes, evaluate)
        print("Done.")

        fitnesses, states = evaluate(phenotypes)

        print("Re-characterizing archived genomes...")
        features = self.aurora.characterize(states)
        print("Done.")

        features = np.array([x for _, x in sorted(zip(fitnesses, features), key=lambda a: a[0])])
        fitnesses = sorted(fitnesses)

        self.novelty_search.reset()
        if self.use_local_competition:
            self.novelty_search.calculate_local_competition(features, fitnesses)
        else:
            self.novelty_search.calculate_novelty(features, fitnesses)

    def newGeneration(self) -> List[Genome]:

        new_genomes = []
        for s in self.species:
            reproduction_members = copy(s.members)

            # Generate new baby genome
            to_spawn = max(0, s.numToSpawn - len(s.members))
            debug_print("Spawning {} for species {}".format(to_spawn, s.ID))
            for i in range(to_spawn):
                baby: Genome = None

                if random.random() > self.mutationRates.crossoverRate:
                    member = random.choice(reproduction_members)
                    baby = deepcopy(member)
                    baby.mutate(self.mutationRates)
                else:

                    # Tournament selection
                    randomMembers = lambda: [random.choice(reproduction_members) for _ in range(2)]
                    g1 = self.tournament_selection(randomMembers())
                    g2 = self.tournament_selection(randomMembers())

                    baby = self.crossover(g1, g2)

                self.currentGenomeID += 1
                baby.ID = self.currentGenomeID

                s.addMember(baby)

            new_genomes += s.members

        return new_genomes

    def reproduce(self) -> None:
        self.generation += 1


        for s in self.species:
            nd_genomes = sorted(s.members, key=lambda x: (x.data['rank'], -x.data['crowding_distance']))
            space_to_fill = int(len(s.members) / 10)
            s.members = nd_genomes[:space_to_fill]

        self.genomes = [m for s in self.species for m in s.members]
        # self.genomes = nd_genomes

        # for s in self.species:
        #     reproduction_members = [g for g in nd_genomes if g in s.members]
        #     s.members = reproduction_members

        self.speciateAll()

        self.calculateSpawnAmount()

        self.genomes = self.newGeneration()