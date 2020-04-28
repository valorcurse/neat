from typing import List

from neat.neat import NEAT
from neat.genes import MutationRates
from neat.evaluation import FitnessEvaluation
from neat.populations.weightagnosticPopulation import WeightAgnosticPopulation, WeightAgnosticConfiguration
from neat.populations.multiobjectivePopulation import MOPopulation
from neat.aurora import Aurora
from neat.phenotypes import Phenotype
from neat.noveltySearch import NoveltySearch
from neat.innovations import Innovations

import numpy as np
from copy import deepcopy

class WeightAgnosticMONEAT(NEAT):

    def __init__(self, eval_env, population_configuration: WeightAgnosticConfiguration, encoding_dim, inputs) -> None:
        super().__init__(eval_env, population_configuration)

        self.weights = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

        mutation_rates = MutationRates()
        self.population = WeightAgnosticMOPopulation(population_configuration, self.innovations, mutation_rates, self.weights)

        self.evaluation_env: FitnessEvaluation = eval_env

        self.aurora = Aurora(encoding_dim, inputs)

        self.novelty_search = NoveltySearch(encoding_dim)



    def _epoch(self):

        self.population.reproduce()

        self.phenotypes = self.population.create_phenotypes()
        results = self.evaluation_env.evaluate(self.phenotypes)

        fitness = results[0]

        n = len(self.weights)
        grouped_fitness = np.array([np.max(fitness[i:i + n]) for i in range(0, len(fitness), n)])

        for genome, f in zip(self.population.genomes, grouped_fitness):
            genome.fitness = f

        # features = np.argmax(results[1], axis=2)
        # features = np.array([np.mean(features[i:i + n].T, axis=1) for i in range(0, len(features), n)])
        features = np.max(np.argmax(np.array([results[1][i:i + n] for i in range(0, len(results[1]), n)]), axis=3), axis=1)
        features = self.aurora.characterize(features)

        # super().updatePopulation(update)

        # fitnesses = np.array([g.fitness for g in self.genomes])

        # novelties = None
        # if self.use_local_competition:
        #     novelties, fitnesses = self.novelty_search.calculate_local_competition(features, fitnesses)
        # else:
        novelties = self.novelty_search.calculate_novelty(features, fitness)
        #
        # for genome, novelty in zip(self.genomes, novelties):
        #     genome.novelty = novelty


        # update = SpeciesUpdate(results)
        # update = np.hstack((grouped_fitness.reshape(grouped_fitness.shape[0], -1), features))
        update = np.array(list(zip(grouped_fitness, novelties)))
        self.population.updatePopulation(update)

class WeightAgnosticMOPopulation(MOPopulation):

    def __init__(self, configuration: WeightAgnosticConfiguration, innovations: Innovations,
                 mutation_rates: MutationRates, weights: List[np.float]):
        mutation_rates.chanceToMutateWeight = 0.0
        # mutation_rates.chanceToMutateActivation = 0.03

        self.weights = weights

        super().__init__(configuration, innovations, mutation_rates)

        # for g in self.genomes:
        #     for _ in range(100):
        #         g.mutate(mutation_rates)

    # @require(lambda update: isinstance(update, SpeciesUpdate))
    # def updatePopulation(self, update: np.array) -> None:
    #     fitness = update[0]
        # n = len(self.weights)
        #
        # grouped_fitnesses = [np.max(fitnesses[i:i + n]) for i in range(0, len(fitnesses), n)]

        # for fitness, genome in zip(fitness, self.genomes):
        #     genome.fitness = fitness


        # for s in self.species:
        #     print(s.best())
        # [fitnesses[i:i + n] for i in range(0, len(fitnesses), n)]

        # Set fitness score to their respective genome
        # for index, genome in enumerate(self.genomes):
        #     genome.fitness = update.fitness[index]

    def create_phenotypes(self) -> List[Phenotype]:

        new_genomes = []
        for genome in self.genomes:
            for weight in self.weights:
                genome.createPhenotype()
                new_genome = deepcopy(genome)
                for link in new_genome.links:
                    link.weight = weight

                new_genomes.append(new_genome)

        return [g.createPhenotype() for g in new_genomes]




    # def refine_behaviors(self, evaluate):
    #
    #     n = len(self.weights)
    #     def evaluate_overload(phenotypes):
    #         _, behaviors = evaluate(phenotypes)
    #
    #         mean_behaviours = np.array([np.mean(behaviors[i:i + n].T, axis=1) for i in range(0, len(behaviors), n)])
    #
    #         return _, mean_behaviours

        # super(WeightAgnosticPopulation, self).refine_behaviors(evaluate_overload)