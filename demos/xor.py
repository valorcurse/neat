from typing import Tuple, List

import numpy as np
from prettytable import PrettyTable

from neat.main.speciatedFitnessNeat import SpeciatedFitnessNeat
from neat.neatTypes import NeuronType
from neat.evaluation import FitnessEvaluation
from neat.phenotypes import Phenotype, ParallelCUDA
from neat.multiobjectivePopulation import MOConfiguration


if __name__ == '__main__':
    pop_size = 100
    inputs = 2
    outputs = 1


    class TestOrganism(FitnessEvaluation):

        def __init__(self):
            self.xor_inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
            self.xor_outputs = np.array([0.0, 1.0, 1.0, 0.0])

            self.feedforward = ParallelCUDA(self.xor_inputs)


        def run_environment(self, phenotypes: List[Phenotype]) -> List[np.array]:
            output = self.feedforward.update(phenotypes)

            fitnesses = np.full(len(phenotypes), 4.0)
            error = np.sum((output.reshape(-1, output.shape[1]) - self.xor_outputs) ** 2, axis=1)
            fitnesses -= error

            return [np.array(fitnesses[:len(phenotypes)])]

        def evaluate(self, phenotypes: List[Phenotype]) -> List[np.array]:
            return self.evaluate_one_to_all(phenotypes)



    pop_config = MOConfiguration(pop_size, inputs, outputs)
    neat = SpeciatedFitnessNeat(TestOrganism(), pop_config)

    highest_fitness = -1000.0
    for _ in neat.epoch():
        print("Epoch {}/{}".format(neat.epochs, 150))

        most_fit = max([{"fitness": g.fitness, "genome": g} for g in neat.population.genomes], key=lambda e: e["fitness"])

        if most_fit["fitness"] > highest_fitness:
            print("New highescore: {:1.2f}".format(most_fit["fitness"]))
            highest_fitness = most_fit["fitness"]


        print("Nr of innovations: {}".format(len(neat.innovations.listOfInnovations)))
        table = PrettyTable(
            ["ID", "age", "members", "max fitness", "avg. distance", "stag", "neurons", "links", "avg. weight",
             "max. compat.", "to spawn"])
        for s in neat.population.species:
            table.add_row([
                # Species ID
                s.ID,
                # Age
                s.age,
                # Nr. of members
                len(s.members),
                # Max fitness
                "{:1.4f}".format(max([m.fitness for m in s.members])),
                # Average distance
                "{:1.4f}".format(max([m.distance for m in s.members])),
                # Stagnation
                s.generationsWithoutImprovement,
                # Neurons
                # "{:1.2f}".format(np.mean([len([n for n in p.graph.nodes.data() if n[1]['type'] == NeuronType.HIDDEN]) for p in neat.phenotypes])),
                "{:1.2f}".format(np.mean(
                    [len([n for n in m.createPhenotype().graph.nodes.data() if n[1]['type'] == NeuronType.HIDDEN]) for m
                     in
                     s.members])),
                # Links
                # "{:1.2f}".format(np.mean([len(p.graph.edges) for p in neat.phenotypes])),
                "{:1.2f}".format(np.mean([len(m.createPhenotype().graph.edges) for m in s.members])),
                # Avg. weight
                "{:1.2f}".format(np.mean([l.weight for m in s.members for l in m.links])),
                # Max. compatiblity
                "{:1.2}".format(np.max([m.calculateCompatibilityDistance(s.leader) for m in s.members])),
                # Nr. of members to spawn
                s.numToSpawn])

        print(table)

        if most_fit["fitness"] > 3.9:
            print("Done!")
            break
