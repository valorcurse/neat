from icontract import require

import numpy as np
from scipy.spatial import cKDTree


import neat.genes as genes
from neat.innovations import Innovations
from neat.populations.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration, SpeciesUpdate
from neat.populations.population import PopulationUpdate

class NoveltyUpdate(SpeciesUpdate):
    def __init__(self, behaviors, fitness: np.ndarray):
        super().__init__(fitness)

        self._data["behaviors"] = behaviors

class NoveltyPopulation(SpeciatedPopulation):

    def __init__(self, configuration: SpeciesConfiguration, innovations: Innovations, mutationRates: genes.MutationRates):
        super().__init__(configuration, innovations, mutationRates)
        self.novelty_map = []
        self.tree = cKDTree([])

    @require(lambda update: isinstance(update, NoveltyUpdate))
    def updatePopulation(self, update: PopulationUpdate) -> None:

        self.novelty_map += update.behaviors

        self.kdtree = cKDTree(self.novelty_map)

        super().updatePopulation(update)