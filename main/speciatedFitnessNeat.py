from neat.neat import NEAT
from neat.utils import chunks
from neat.genes import MutationRates
from neat.aurora import Aurora
from neat.noveltySearch import NoveltySearch
from neat.evaluation import FitnessEvaluation
from neat.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration, SpeciesUpdate
# from neat.multiobjectivePopulation import MOPopulation, MOConfiguration, MOUpdate

import numpy as np

class SpeciatedFitnessNeat(NEAT):

    def __init__(self, eval_env, population_configuration: SpeciesConfiguration) -> None:
        super().__init__(eval_env, population_configuration)

        mutation_rates = MutationRates()
        self.population = SpeciatedPopulation(population_configuration, self.innovations, mutation_rates)

        self.evaluation_env: FitnessEvaluation = eval_env

    def _epoch(self):
        # old_pop = self.population.genomes
        # old_max = max([{"fitness": g.fitness, "genome": g} for g in self.population.genomes],
        #                key=lambda e: e["fitness"])

        self.population.reproduce()

        self.phenotypes = self.population.create_phenotypes()
        results = self.evaluation_env.evaluate(self.phenotypes)

        # old_pop = self.population.genomes
        # self.old_max = max([{"fitness": g.fitness, "genome": g} for g in self.population.genomes],
        #               key=lambda e: e["fitness"])

        # objectives = np.array([results])

        update = SpeciesUpdate(results)
        self.population.updatePopulation(update)