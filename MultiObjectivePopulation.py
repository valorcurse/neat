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
        self.max_epochs = 50
        self.aurora = Aurora(encoding_dim, behavior_matrix_size)

        self.novelty_map = NoveltySearch(encoding_dim)

    @require(lambda update: isinstance(update, MOUpdate))
    def updatePopulation(self, update: PopulationUpdate) -> None:

        features = self.aurora.characterize(update.behaviors)
        novelties = self.novelty_map.calculate_novelty(features)
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=zip(update.fitness, novelties))

        # ndf = np.array(ndf).flatten()[::-1]

        position_fitnesses = [-f for f in np.array(np.hstack(ndf), dtype=np.int32)]

        # print(position_fitnesses)

        # for g, f in zip(self.genomes, position_fitnesses):
        #     g.fitness = f

        # update.fitness = -dc.astype(np.int32)

        self.genomes = [self.genomes[i] for i in ndf[0]]

        super().updatePopulation(update)

        self.epochs += 1

    def refine_behaviors(self, eval_env: Evaluation):
        # sorted_archive = sorted(self.population_and_fitnesses(), key=lambda a: a['fitness'])
        # ten_percent = max(1, int(len(sorted_archive) * 0.1))
        # self.genomes = [a['genome'] for a in sorted_archive[:ten_percent]]

        phenotypes = self.create_phenotypes()
        print("Refining AURORA...")
        self.aurora.refine(phenotypes, eval_env)
        print("Done.")

        _, states = eval_env.evaluate(phenotypes)
        print("Re-characterizing archived genomes...")
        features = self.aurora.characterize(states)
        print("Done.")

        self.novelty_map.reset()
        self.novelty_map.calculate_novelty(features)