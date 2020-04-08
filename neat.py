from typing import List, Any, Tuple

import numpy as np

import neat.innovations
from neat.utils import chunks
from neat.phenotypes import Phenotype
from neat.genes import MutationRates, Phase, SpeciationType

from neat.mapElites import MapElites, MapElitesConfiguration, MapElitesUpdate
from neat.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration, SpeciesUpdate
from neat.multiobjectivePopulation import MOConfiguration, MOPopulation, MOUpdate
from neat.population import PopulationConfiguration, Population

class NEAT:

    def __init__(self, eval_env, population_configuration: PopulationConfiguration, mutation_rates: MutationRates=MutationRates()) -> None:

        self.population_configuration = population_configuration
        self.innovations = neat.innovations.Innovations()

        # self.objective_ranges = population_configuration.objective_ranges

        self.mutation_rates: MutationRates = mutation_rates
        self.phenotypes: List[Phenotype] = []

        self.phase: Phase = Phase.COMPLEXIFYING
        self.speciationType: SpeciationType = SpeciationType.NOVELTY

        self.epochs = -1
        self.refinement_epochs = [0, 50, 100, 350, 750, 1550]
        # self.refinement_epochs = [0, 10, 25, 50, 100, 150, 350, 750, 1550]

        # If using MapElites
        if isinstance(self.population_configuration, MapElitesConfiguration):
            self.population: Population = MapElites(population_configuration, self.innovations, mutation_rates)

        # If using regular speciated population
        elif isinstance(self.population_configuration, SpeciesConfiguration):
            self.population = SpeciatedPopulation(population_configuration, self.innovations, mutation_rates)

        # If using regular speciated population
        elif isinstance(self.population_configuration, MOConfiguration):
            self.population = MOPopulation(population_configuration, self.innovations, mutation_rates)

        # Passed function that runs the evaluation environment
        self.eval_env = eval_env



    def evaluate(self):
        pass

    def _epoch(self):
        pass

        # if self.epochs in self.refinement_epochs:
        #     self.population.refine_behaviors(self.evaluate_vectorized)
        #
        # self.population.reproduce()
        #
        # self.phenotypes = self.population.create_phenotypes()
        # fitnesses, states = self.evaluate_vectorized(self.phenotypes)
        #
        # if isinstance(self.population_configuration, MapElitesConfiguration):
        #     self.population.updatePopulation(MapElitesUpdate(fitnesses, states))
        #
        # elif isinstance(self.population_configuration, SpeciesConfiguration):
        #     self.population.updatePopulation(SpeciesUpdate(fitnesses))
        #
        # elif isinstance(self.population_configuration, MOConfiguration):
        #     self.population.updatePopulation(MOUpdate(fitnesses, states))

    def epoch(self):

        while True:
            self.epochs += 1

            self._epoch()

            yield
