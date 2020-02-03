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

        self.milestone: float = 0.01
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

        # This is just for debugging
        for phenotype, fitness in zip(phenotypes, all_fitnesses):
            phenotype.fitness = fitness

        return (np.array(all_fitnesses), np.array(all_features))


    def epoch(self):

        while True:
            self.epochs += 1

            if self.epochs in self.refinement_epochs:
                self.population.refine_behaviors(self.evaluate_vectorized)

            self.population.reproduce()

            self.phenotypes = self.population.create_phenotypes()
            fitnesses, states = self.evaluate_vectorized(self.phenotypes)

            if isinstance(self.population_configuration, MapElitesConfiguration):
                self.population.updatePopulation(MapElitesUpdate(fitnesses, states))

            elif isinstance(self.population_configuration, SpeciesConfiguration):
                self.population.updatePopulation(SpeciesUpdate(fitnesses))

            elif isinstance(self.population_configuration, MOConfiguration):
                self.population.updatePopulation(MOUpdate(fitnesses, states))

            yield
