from typing import List, Tuple
from icontract import require

import numpy as np
import scipy as sp
import pygmo as pg
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
import random
from copy import deepcopy


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

    def __init__(self, configuration: SpeciesConfiguration, innovations: Innovations, mutationRates: genes.MutationRates):
        super().__init__(configuration, innovations, mutationRates)

        encoding_dim = 2
        self.behavior_dimensions = configuration.behavior_dimensions

        # Temp way to trigger aurora refinement
        self.epochs = 0
        self.aurora = Aurora(encoding_dim, self.behavior_dimensions)

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
        space_to_fill = int(self.population_size/2)
        space_left = space_to_fill
        i = 0
        while(space_left > 0):
            front_genomes = [self.genomes[j] for j in ndf[i]]
            front_points = [points[j] for j in ndf[i]]
            front_ranks = [ndr[j] for j in ndf[i]]


            crowding_distances = pg.crowding_distance(front_points) if len(front_points) > 1 else np.zeros((len(front_points)))
            for g, d, r in zip(front_genomes, crowding_distances, front_ranks):
                g.crowding_distance = d
                g.rank = r

            genomes_to_take = min(len(front_genomes), space_left)

            nd_genomes.extend(front_genomes[:genomes_to_take])

            space_left = max(0, space_left - len(front_genomes[:genomes_to_take]))

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

    def newGeneration(self) -> None:

        newPop = []
        for s in self.species:
            # s.members.sort(reverse=True, key=lambda x: x.fitness)

            newPop.extend(s.members)

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

            for i in range(s.numToSpawn - len(s.members)):
                baby: genes.Genome = None

                if random.random() > self.mutationRates.crossoverRate:
                    member = random.choice(s.members)
                    baby = deepcopy(member)
                    baby.mutate(self.mutationRates)
                else:
                    # Tournament selection
                    randomMembers = lambda: [random.choice(s.members) for _ in range(2)]
                    g1 = sorted(randomMembers(), key=lambda x: (x.rank, -x.crowding_distance))[0]
                    g2 = sorted(randomMembers(), key=lambda x: (x.rank, -x.crowding_distance))[0]

                    baby = self.crossover(g1, g2)

                self.currentGenomeID += 1
                baby.ID = self.currentGenomeID

                newPop.append(baby)

        self.genomes = newPop

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