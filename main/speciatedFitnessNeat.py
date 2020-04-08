from neat.neat import NEAT
from neat.utils import chunks
from neat.genes import MutationRates
from neat.aurora import Aurora
from neat.noveltySearch import NoveltySearch
from neat.evaluation import FitnessEvaluation
# from neat.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration, SpeciesUpdate
from neat.multiobjectivePopulation import MOPopulation, MOConfiguration, MOUpdate

import numpy as np

class SpeciatedFitnessNeat(NEAT):

    def __init__(self, eval_env, population_configuration: MOConfiguration) -> None:
        super().__init__(eval_env, population_configuration)

        mutation_rates = MutationRates()
        self.population = MOPopulation(population_configuration, self.innovations, mutation_rates)

        self.evaluation_env: FitnessEvaluation = eval_env

    def _epoch(self):

        self.population.reproduce()

        self.phenotypes = self.population.create_phenotypes()
        results = self.evaluation_env.evaluate(self.phenotypes)

        update = MOUpdate(results[0], results[1:])
        self.population.updatePopulation(update)

    # def evaluate(self):
    #     all_fitnesses = np.empty((0,))
    #     for chunk in chunks(phenotypes, self.evaluation_env.num_of_envs):
    #         fitnesses, features = self.eval_env.evaluate(chunk)
    #
    #         assert fitnesses.shape == (len(chunk),), \
    #             "Chunk of fitnesses have shape {} instead of {}.".format(fitnesses.shape, (len(chunk),))
    #         assert features.shape == (len(chunk), self.population_configuration.behavior_dimensions), \
    #             "Chunk of states have shape {} instead of {}.".format(features.shape, (
    #             len(chunk), self.population_configuration.behavior_dimensions))
    #
    #         all_features = np.vstack((all_features, features))
    #         all_fitnesses = np.hstack((all_fitnesses, fitnesses))
    #
    #     assert all_fitnesses.shape[0] == len(phenotypes), "Combined fitnesses have size {} instead of {}.".format(
    #         len(all_fitnesses), len(phenotypes))
    #     assert all_features.shape[0] == len(phenotypes), "Combined states have size {} instead of {}.".format(
    #         len(all_features), len(phenotypes))
    #
    #     # This is just for debugging
    #     for phenotype, fitness in zip(phenotypes, all_fitnesses):
    #         phenotype.fitness = fitness
    #
    #     return (np.array(all_fitnesses), np.array(all_features))


    def evaluate(self):
        pass
        # self.evaluation_env.evaluate(self.phenotypes)
        # all_fitnesses = np.empty((0,))
        # for chunk in chunks(phenotypes, self.evaluation_env.num_of_envs):
        #     fitnesses, features = self.eval_env.evaluate(chunk)
        #
        #     assert fitnesses.shape == (len(chunk),), \
        #         "Chunk of fitnesses have shape {} instead of {}.".format(fitnesses.shape, (len(chunk),))
        #     assert features.shape == (len(chunk), self.population_configuration.behavior_dimensions), \
        #         "Chunk of states have shape {} instead of {}.".format(features.shape, (
        #         len(chunk), self.population_configuration.behavior_dimensions))
        #
        #     all_features = np.vstack((all_features, features))
        #     all_fitnesses = np.hstack((all_fitnesses, fitnesses))
        #
        # assert all_fitnesses.shape[0] == len(phenotypes), "Combined fitnesses have size {} instead of {}.".format(
        #     len(all_fitnesses), len(phenotypes))
        # assert all_features.shape[0] == len(phenotypes), "Combined states have size {} instead of {}.".format(
        #     len(all_features), len(phenotypes))
        #
        # # This is just for debugging
        # for phenotype, fitness in zip(phenotypes, all_fitnesses):
        #     phenotype.fitness = fitness
        #
        # return (np.array(all_fitnesses), np.array(all_features))