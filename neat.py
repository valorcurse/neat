from typing import List

import neat.innovations
from neat.phenotypes import Phenotype
from neat.genes import MutationRates, Phase, SpeciationType

from neat.populations.population import PopulationConfiguration, Population
from neat.populations.weightagnosticPopulation import WeightAgnosticConfiguration, WeightAgnosticPopulation
from neat.populations.mapElites import MapElites, MapElitesConfiguration
from neat.populations.multiobjectivePopulation import MOConfiguration, MOPopulation
from neat.populations.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration


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
        # self.refinement_epochs = [50, 100, 350, 750, 1550]
        # self.refinement_epochs = [0, 10, 25, 50, 100, 150, 350, 750, 1550]

        # If using MapElites
        if isinstance(self.population_configuration, MapElitesConfiguration):
            self.population: Population = MapElites(population_configuration, self.innovations, mutation_rates)

        # If using regular speciated population
        elif isinstance(self.population_configuration, SpeciesConfiguration):
            self.population = SpeciatedPopulation(population_configuration, self.innovations, mutation_rates)

        # If using multi-objective speciated population
        elif isinstance(self.population_configuration, MOConfiguration):
            self.population = MOPopulation(population_configuration, self.innovations, mutation_rates)

        # If using weight-agnostic speciated population
        elif isinstance(self.population_configuration, WeightAgnosticConfiguration):
            self.population = WeightAgnosticPopulation(population_configuration, self.innovations, mutation_rates)

        # Passed function that runs the evaluation environment
        self.eval_env = eval_env



    def evaluate(self):
        pass

    def _epoch(self):
        pass

    def epoch(self):

        while True:
            self.epochs += 1

            self._epoch()

            yield
