from typing import List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, eval_env, population_configuration: PopulationConfiguration, mutation_rates: MutationRates=MutationRates()) -> None:

        self.population_configuration = population_configuration
        self.innovations = neat.innovations.Innovations()

        self.objective_ranges = population_configuration.objective_ranges

        self.mutation_rates: MutationRates = mutation_rates
        self.phenotypes: List[Phenotype] = []

        self.phase: Phase = Phase.COMPLEXIFYING
        self.speciationType: SpeciationType = SpeciationType.NOVELTY

        self.milestone: float = 0.01
        self.epochs = -1
        # self.refinement_epoch = 50
        self.refinement_epochs = [25, 50, 100, 350, 750, 1550]
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

    def evaluate_vectorized(self, phenotypes) -> Tuple[np.ndarray, np.ndarray]:
        all_features = np.empty((0, self.population_configuration.behavior_dimensions))
        all_fitnesses = np.empty((0, ))
        for chunk in chunks(phenotypes, self.eval_env.num_of_envs):
            fitnesses, features = self.eval_env.evaluate(chunk)

            assert fitnesses.shape == (len(chunk), ), \
                "Chunk of fitnesses have shape {} instead of {}.".format(fitnesses.shape, (len(chunk), ))
            assert features.shape == (len(chunk), self.population_configuration.behavior_dimensions), \
                "Chunk of states have shape {} instead of {}.".format(features.shape, (len(chunk), self.population_configuration.behavior_dimensions))

            all_features = np.vstack((all_features, features))
            all_fitnesses = np.hstack((all_fitnesses, fitnesses))

        assert all_fitnesses.shape[0] == len(phenotypes), "Combined fitnesses have size {} instead of {}.".format(len(all_fitnesses), len(phenotypes))
        assert all_features.shape[0] == len(phenotypes), "Combined states have size {} instead of {}.".format(len(all_features), len(phenotypes))

        # Normalize objectives
        # fitness_range = self.objective_ranges[0]
        # all_fitnesses = (all_fitnesses - fitness_range[0]) / (fitness_range[1] - fitness_range[0])

        for phenotype, fitness in zip(phenotypes, all_fitnesses):
            phenotype.fitness = fitness

        return (np.array(all_fitnesses), np.array(all_features))


    def epoch(self):

        while True:
            self.epochs += 1

            if self.epochs in self.refinement_epochs:
                # self.population.refine_behaviors(self.eval_env)
                self.population.refine_behaviors(self.evaluate_vectorized)

            self.phenotypes = self.population.create_phenotypes()
            fitnesses, states = self.evaluate_vectorized(self.phenotypes)
            print("Fitnesses: {}".format(fitnesses))

            self.population.updatePopulation(MOUpdate(states, fitnesses))
            self.population.reproduce()

            # plt.draw()
            # plt.pause(0.1)

            yield
