from typing import List, Tuple
from icontract import require

import numpy as np
import scipy as sp
import pygmo as pg
from scipy.spatial import cKDTree


import neat.genes as genes
from neat.aurora import Aurora
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
        encoding_dim = 8
        behavior_dimensions = 7
        behavior_steps = 60
        behavior_matrix_size = behavior_dimensions * behavior_steps

        # Temp way to trigger aurora refinement
        self.epochs = 0
        self.aurora = Aurora(encoding_dim, behavior_matrix_size)

        self.novelty_map = NoveltySearch(encoding_dim)

    @require(lambda update: isinstance(update, MOUpdate))
    def updatePopulation(self, update: PopulationUpdate) -> None:

        features = self.aurora.characterize(update.behaviors)
        # novelties = self.novelty_map.calculate_novelty(features)
        novelties, local_fitnesses = self.novelty_map.calculate_local_competition(features, update.fitness)
        points = list(zip(-local_fitnesses, -novelties))

        update.behaviors = features
        update.fitness = local_fitnesses

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=points)
        pg.plotting.plot_non_dominated_fronts(points)

        nd_genomes = []
        space_to_fill = int(self.population_size/2)
        space_left = space_to_fill
        i = 0
        while(space_left > 0):
            non_dominated = [self.genomes[j] for j in ndf[i]]
            genomes_to_take = min(len(non_dominated), space_left)
            nd_genomes.extend(non_dominated[:genomes_to_take])

            space_left = max(0, space_left - len(nd_genomes))

            i += 1

        self.genomes = nd_genomes
        self.speciate()

        # nd_indexes = np.array(ndf[:i]).flatten()
        nd_indexes = [a for l in ndf[:i] for a in l]
        update.fitness = np.array([update.fitness[i] for i in nd_indexes], dtype=np.float32)
        update.behaviors = np.array([update.behaviors[i] for i in nd_indexes], dtype=np.float32)

        super().updatePopulation(update)

        self.epochs += 1


    def refine_behaviors(self, eval_env: Evaluation):
        phenotypes = self.create_phenotypes()
        print("Refining AURORA...")
        self.aurora.refine(phenotypes, eval_env)
        print("Done.")

        fitnesses, states = eval_env.evaluate(phenotypes)
        print("Re-characterizing archived genomes...")
        features = self.aurora.characterize(states)
        print("Done.")

        self.novelty_map.reset()
        # self.novelty_map.calculate_novelty(features)
        self.novelty_map.calculate_local_competition(features, fitnesses)