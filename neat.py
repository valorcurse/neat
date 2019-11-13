from typing import List, Any

import numpy as np

import neat.innovations
from neat.utils import chunks
from neat.phenotypes import Phenotype
from neat.evaluation import Evaluation
from neat.genes import MutationRates, Phase, SpeciationType

from neat.mapElites import MapElites, MapElitesConfiguration
from neat.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration
from neat.multiobjectivePopulation import MOConfiguration, MOPopulation, MOUpdate
from neat.population import PopulationConfiguration, Population

class NEAT:

    def __init__(self, eval_env: Evaluation, population_configuration: PopulationConfiguration, mutation_rates: MutationRates=MutationRates()) -> None:

        self.population_configuration = population_configuration
        self.innovations = neat.innovations.Innovations()

        self.mutation_rates: MutationRates = mutation_rates
        self.phenotypes: List[Phenotype] = []

        self.phase: Phase = Phase.COMPLEXIFYING
        self.speciationType: SpeciationType = SpeciationType.NOVELTY

        self.milestone: float = 0.01
        self.epochs = -1
        # self.refinement_epoch = 50
        self.refinement_epochs = [0, 50, 150, 350, 750, 1550]

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

    def evaluate_vectorized(self, phenotypes):
        all_states = []
        fitnesses = []
        for chunk in chunks(phenotypes, self.eval_env.num_of_envs):
            fitness, states = self.eval_env.evaluate(chunk)
            all_states.extend(states)
            fitnesses.extend(fitness)

        return (np.array(all_states), np.array(fitnesses))


    def epoch(self):

        while True:
            self.epochs += 1

            if self.epochs in self.refinement_epochs:
                self.population.refine_behaviors(self.eval_env)

            phenotypes = self.population.create_phenotypes()
            states, fitnesses = self.evaluate_vectorized(phenotypes)
            print("Fitnesses: {}".format(fitnesses))

            self.population.updatePopulation(MOUpdate(states, fitnesses))
            self.population.reproduce()

            yield
