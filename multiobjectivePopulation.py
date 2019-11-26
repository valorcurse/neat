from typing import List, Tuple
from icontract import require

import numpy as np
import scipy as sp
import pygmo as pg
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns


import neat.genes as genes
from neat.aurora import Aurora
from neat.visualize import Visualize
from neat.evaluation import Evaluation
from neat.innovations import Innovations
from neat.population import PopulationUpdate
from neat.noveltySearch import NoveltySearch
from neat.population import PopulationConfiguration
from neat.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration, SpeciesUpdate

class MOConfiguration(PopulationConfiguration):
    def __init__(self, population_size: int, n_inputs: int, n_outputs: int):
        self._data = {
            "population_size": population_size,
            "n_inputs": n_inputs,
            "n_outputs": n_outputs
        }

class MOUpdate(SpeciesUpdate):
    def __init__(self, behaviors, fitness: np.ndarray):
        super().__init__(fitness)

        self._data["behaviors"] = behaviors

class MOPopulation(SpeciatedPopulation):

    def __init__(self, configuration: SpeciesConfiguration, innovations: Innovations, mutationRates: genes.MutationRates):
        super().__init__(configuration, innovations, mutationRates)
        encoding_dim = 2
        behavior_dimensions = 14
        behavior_steps = 250
        behavior_matrix_size = behavior_dimensions * behavior_steps

        # Temp way to trigger aurora refinement
        self.epochs = 0
        self.aurora = Aurora(encoding_dim, behavior_matrix_size)

        self.novelty_search = NoveltySearch(encoding_dim)

        self.use_local_competition = False

        plt.figure(1)
        plt.ion()
        plt.draw()
        plt.pause(0.1)
        plt.show()

        plt.figure(2)
        plt.ion()
        plt.draw()
        plt.pause(0.1)
        plt.show()

        # self.fitness_novelty_vis = Visualize().fig.add_subplot(1, 2, 2)
        # self.fitness_novelty_vis.set_title("Fitness/Novelty")

        # self.fig, self.ax = plt.subplots()
        # self.ax.set(xlabel="Exam score-1", ylabel="Exam score-2")
        # self.ax.legend()
        # plt.axis([-1, 1, -1, 1])
        # plt.xlabel('Novelty')
        # plt.ylabel('Fitness')
        # # plt.tight_layout()
        # plt.ion()
        # plt.show()

    @require(lambda update: isinstance(update, MOUpdate))
    def updatePopulation(self, update: PopulationUpdate) -> None:

        features = self.aurora.characterize(update.behaviors)

        fitnesses = update.fitness
        novelties = []
        if self.use_local_competition:
            novelties, fitnesses = self.novelty_search.calculate_local_competition(features, fitnesses)
        else:
            novelties = self.novelty_search.calculate_novelty(features, fitnesses)

        # self.fitness_novelty_vis.scatter(novelties, fitnesses)
        # plt.figure(1)
        # plt.scatter(features.T[0], features.T[1])

        plt.figure(2)
        plt.clf()
        plt.scatter(novelties, fitnesses)
        plt.draw()
        plt.pause(0.01)

        # Fitness and novelty are made negative, because the non dominated sorting
        # is a minimalization algorithm
        points = list(zip(-fitnesses, -novelties))

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=points)

        print("Non-dominated genomes:")
        print([(fitnesses[i], novelties[i]) for i in ndf[0]])

        nd_genomes = []
        space_to_fill = int(self.population_size/5)
        space_left = space_to_fill
        i = 0
        while(space_left > 0):
            non_dominated = [self.genomes[j] for j in ndf[i]]

            genomes_to_take = min(len(non_dominated), space_left)

            nd_genomes.extend(non_dominated[:genomes_to_take])

            space_left = max(0, space_left - len(non_dominated[:genomes_to_take]))

            i += 1

        # Make sure the correct number of genomes were copied over
        assert len(nd_genomes) == space_to_fill, \
               "Only {}/{} genomes were copied.".format(len(nd_genomes), space_to_fill)



        self.genomes = nd_genomes
        self.speciate()

        # nd_indexes = np.array(ndf[:i]).flatten()
        nd_indexes = [a for l in ndf[:i] for a in l]
        update.fitness = np.array([update.fitness[i] for i in nd_indexes], dtype=np.float32)
        update.behaviors = np.array([update.behaviors[i] for i in nd_indexes], dtype=np.float32)

        # novelties = np.array([novelties[i] for i in nd_indexes], dtype=np.float32)
        # update.fitness = novelties

        super().updatePopulation(update)

        self.epochs += 1


    def refine_behaviors(self, eval_env: Evaluation):
        plt.figure(1)
        plt.clf()

        phenotypes = self.create_phenotypes()
        print("Refining AURORA...")
        self.aurora.refine(phenotypes, eval_env)
        print("Done.")

        fitnesses, states = eval_env.evaluate(phenotypes)
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