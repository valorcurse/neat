from typing import List, Tuple
from icontract import require

import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import random
import math

from copy import deepcopy

# import neat.genes as genes
from neat.utils import debug_print
from neat.aurora import Aurora
from neat.evaluation import Evaluation
from neat.innovations import Innovations
from neat.population import PopulationUpdate
from neat.noveltySearch import NoveltySearch
from neat.genes import NeuronGene, LinkGene, MutationRates
from neat.population import PopulationConfiguration
from neat.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration, SpeciesUpdate

class MOConfiguration(PopulationConfiguration):
    def __init__(self, population_size: int, n_inputs: int, n_outputs: int, behavior_dimensions: int, obj_ranges: List[Tuple[float, float]]):
        self._data = {
            "population_size": population_size,
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "behavior_dimensions": behavior_dimensions,
            "objective_ranges": obj_ranges
        }

class MOUpdate(SpeciesUpdate):
    def __init__(self, behaviors, fitness: np.ndarray):
        super().__init__(fitness)

        self._data["behaviors"] = behaviors

class MOPopulation(SpeciatedPopulation):

    def __init__(self, configuration: SpeciesConfiguration, innovations: Innovations, mutationRates: MutationRates):
        super().__init__(configuration, innovations, mutationRates)

        encoding_dim = 8
        self.behavior_dimensions = configuration.behavior_dimensions

        # Temp way to trigger aurora refinement
        self.epochs = 0
        self.aurora = Aurora(encoding_dim, self.behavior_dimensions)

        self.novelty_search = NoveltySearch(encoding_dim)

        self.use_local_competition = False

    def newGenome(self, neurons: List[NeuronGene] = [], links: List[LinkGene] = [], parents=[]):

        genome = super().newGenome(neurons, links, parents)
        genome.data['rank'] = 0
        genome.data['crowding_distance'] = 0

        return genome

    @require(lambda update: isinstance(update, MOUpdate))
    def updatePopulation(self, update: PopulationUpdate) -> None:

        features = self.aurora.characterize(update.behaviors)

        fitnesses = update.fitness
        novelties = []
        if self.use_local_competition:
            novelties, fitnesses = self.novelty_search.calculate_local_competition(features, fitnesses)
        else:
            novelties = self.novelty_search.calculate_novelty(features, fitnesses)


        for genome, novelty in zip(self.genomes, novelties):
            genome.novelty = novelty

        # Fitness and novelty are made negative, because the non dominated sorting
        # is a minimalization algorithm
        points = list(zip(-fitnesses, -novelties))

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=points)

        print("Non-dominated genomes:")
        print([(fitnesses[i], novelties[i]) for i in ndf[0]])

        nd_genomes = []
        space_to_fill = int(self.population_size/2)
        space_left = space_to_fill
        i = 0
        while (space_left > 0):
            front_genomes = [self.genomes[j] for j in ndf[i]]
            front_points = [points[j] for j in ndf[i]]
            front_ranks = [ndr[j] for j in ndf[i]]


            crowding_distances = pg.crowding_distance(front_points) if len(front_points) > 1 else np.zeros((len(front_points)))
            for g, d, r in zip(front_genomes, crowding_distances, front_ranks):
                g.data['crowding_distance'] = d
                g.data['rank'] = r

            genomes_to_take = min(len(front_genomes), space_left)

            nd_genomes.extend(front_genomes[:genomes_to_take])

            space_left = max(0, space_left - len(front_genomes[:genomes_to_take]))

            i += 1

        # Make sure the correct number of genomes were copied over
        assert len(nd_genomes) == space_to_fill, \
               "Only {}/{} genomes were copied.".format(len(nd_genomes), space_to_fill)

        self.genomes = nd_genomes
        for s in self.species:
            s.members = []

        for g in self.genomes:
            self.speciate(g)

        for s in [s for s in self.species if len(s.members) == 0]:
            self.species.remove(s)

        nd_indexes = [a for l in ndf[:i] for a in l]
        update.fitness = np.array([update.fitness[i] for i in nd_indexes], dtype=np.float32)
        update.behaviors = np.array([update.behaviors[i] for i in nd_indexes], dtype=np.float32)

        super().updatePopulation(update)

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