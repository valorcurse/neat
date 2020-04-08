from typing import List, Tuple, Dict

import numpy as np

from neat.utils import chunks
from neat.phenotypes import Phenotype

class Evaluation:

    def __init__(self):
        self.num_of_envs = 0

    def evaluate_one_to_one(self, phenotypes) -> Dict:
        # all_features = np.empty((0, self.population_configuration.behavior_dimensions))
        # all_fitnesses = np.empty((0, ))

        all_results = []
        for chunk in chunks(phenotypes, self.num_of_envs):
            # fitnesses, features = self.evaluate(chunk)
            results = self.run_environment(chunk)

            all_results.append(results)

            # assert fitnesses.shape == (len(chunk), ), \
            #     "Chunk of fitnesses have shape {} instead of {}.".format(fitnesses.shape, (len(chunk), ))
            # assert features.shape == (len(chunk), self.population_configuration.behavior_dimensions), \
            #     "Chunk of states have shape {} instead of {}.".format(features.shape, (len(chunk), self.population_configuration.behavior_dimensions))

            # all_features = np.vstack((all_features, features))
            # all_fitnesses = np.hstack((all_fitnesses, fitnesses))

        # assert all_fitnesses.shape[0] == len(phenotypes), "Combined fitnesses have size {} instead of {}.".format(len(all_fitnesses), len(phenotypes))
        # assert all_features.shape[0] == len(phenotypes), "Combined states have size {} instead of {}.".format(len(all_features), len(phenotypes))

        # This is just for debugging
        # for phenotype, fitness in zip(phenotypes, all_fitnesses):
        #     phenotype.fitness = fitness

        # return (np.array(all_fitnesses), np.array(all_features))
        return all_results

    def evaluate_one_to_all(self, phenotypes) -> List[np.array]:
        return self.run_environment(phenotypes)

    ''' To be overloaded '''
    def run_environment(self, phenotypes: List[Phenotype]) -> List[np.array]:
        pass

    ''' To be overloaded '''
    def evaluate(self, phenotypes: List[Phenotype]) -> List[np.array]:
        pass

class FitnessEvaluation(Evaluation):

    def __init__(self):
        self.num_of_envs = 0

    def evaluate(self, phenotypes: List[Phenotype]) -> List[np.ndarray]:
        pass