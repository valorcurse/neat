from typing import List

from copy import deepcopy
import numpy as np

from neat.innovations import Innovations
from neat.genes import MutationRates
from neat.phenotypes import Phenotype
from neat.populations.population import PopulationConfiguration, PopulationUpdate
# from neat.populations.multiobjectivePopulation import MOPopulation
from neat.populations.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration

# class WeightAgnosticConfiguration(PopulationConfiguration):
#     def __init__(self, population_size: int, n_inputs: int, n_outputs: int):
#         super().__init__(population_size, n_inputs, n_outputs)

class WeightAgnosticConfiguration(SpeciesConfiguration):
    def __init__(self, population_size: int, n_inputs: int, n_outputs: int):
        super().__init__(population_size, n_inputs, n_outputs)

        # self._data["behavior_dimensions"] = behavior_dimensions

class WeightAgnosticPopulation(SpeciatedPopulation):

    def __init__(self, configuration: WeightAgnosticConfiguration, innovations: Innovations, mutation_rates: MutationRates):
        mutation_rates.chanceToMutateWeight = 0.0
        # mutation_rates.chanceToMutateActivation = 0.03

        self.weights = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

        super().__init__(configuration, innovations, mutation_rates)

        # for g in self.genomes:
        #     for _ in range(100):
        #         g.mutate(mutation_rates)

    # @require(lambda update: isinstance(update, SpeciesUpdate))
    def updatePopulation(self, update: PopulationUpdate) -> None:
        fitnesses = update.fitness
        n = len(self.weights)

        grouped_fitnesses = [np.max(fitnesses[i:i + n]) for i in range(0, len(fitnesses), n)]

        for fitness, genome in zip(grouped_fitnesses, self.genomes):
            genome.fitness = fitness


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

        return [g.createPhenotype() for  g in new_genomes]




    def refine_behaviors(self, evaluate):

        n = len(self.weights)
        def evaluate_overload(phenotypes):
            _, behaviors = evaluate(phenotypes)

            mean_behaviours = np.array([np.mean(behaviors[i:i + n].T, axis=1) for i in range(0, len(behaviors), n)])

            return _, mean_behaviours

        super(WeightAgnosticPopulation, self).refine_behaviors(evaluate_overload)