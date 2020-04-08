from neat.neat import NEAT
from neat.utils import chunks
from neat.genes import MutationRates
from neat.aurora import Aurora
from neat.noveltySearch import NoveltySearch
from neat.evaluation import FitnessEvaluation
from neat.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration, SpeciesUpdate

import numpy as np

class SpeciatedFitnessNeat(NEAT):

    def __init__(self, eval_env, population_configuration: SpeciesConfiguration) -> None:
        mutation_rates = MutationRates()
        self.population = SpeciatedPopulation(population_configuration, self.innovations, mutation_rates)

        self.evaluation_env: FitnessEvaluation = eval_env

    def _epoch(self):

        results = self.evaluation_env.evaluate(self.phenotypes)

        # Get this from somewhere
        encoding_dim = 8
        self.behavior_dimensions = configuration.behavior_dimensions

        self.aurora = Aurora(encoding_dim, self.behavior_dimensions)

        self.use_local_competition = False
        self.novelty_search = NoveltySearch(encoding_dim)

        features = self.aurora.characterize(update.behaviors)

        novelties = None
        if self.use_local_competition:
            novelties, fitnesses = self.novelty_search.calculate_local_competition(features, fitnesses)
        else:
            novelties = self.novelty_search.calculate_novelty(features, fitnesses)

        for genome, novelty in zip(self.genomes, novelties):
            genome.novelty = novelty

        self.population.updatePopulation(SpeciesUpdate(fitnesses))

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
        all_fitnesses = np.empty((0,))
        for chunk in chunks(phenotypes, self.evaluation_env.num_of_envs):
            fitnesses, features = self.eval_env.evaluate(chunk)

            assert fitnesses.shape == (len(chunk),), \
                "Chunk of fitnesses have shape {} instead of {}.".format(fitnesses.shape, (len(chunk),))
            assert features.shape == (len(chunk), self.population_configuration.behavior_dimensions), \
                "Chunk of states have shape {} instead of {}.".format(features.shape, (
                len(chunk), self.population_configuration.behavior_dimensions))

            all_features = np.vstack((all_features, features))
            all_fitnesses = np.hstack((all_fitnesses, fitnesses))

        assert all_fitnesses.shape[0] == len(phenotypes), "Combined fitnesses have size {} instead of {}.".format(
            len(all_fitnesses), len(phenotypes))
        assert all_features.shape[0] == len(phenotypes), "Combined states have size {} instead of {}.".format(
            len(all_features), len(phenotypes))

        # This is just for debugging
        for phenotype, fitness in zip(phenotypes, all_fitnesses):
            phenotype.fitness = fitness

        return (np.array(all_fitnesses), np.array(all_features))