from typing import List

import numpy as np

from neat.neat import NEAT
from neat.neatTypes import NeuronType
from neat.phenotypes import Phenotype, SubstrateCUDA
from neat.population import PopulationUpdate, PopulationConfiguration
from neat.genes import Genome, NeuronGene, MutationRates

import networkx as nx
from visualize import Visualize

from itertools import groupby


class HyperNEAT(NEAT):

    def __init__(self, population_configuration: PopulationConfiguration, mutation_rates: MutationRates=MutationRates()):
        
        # CPPNs take 4 inputs, gotta move this somewhere else
        nrOfLayers: int = 3
        self.layers = np.linspace(-1.0, 1.0, num=nrOfLayers)
        cppnInputs: int = 4
        # cppnOutputs: int = nrOfLayers - 1
        cppnOutputs = 1

        self.n_inputs = population_configuration.n_inputs
        self.n_outputs = population_configuration.n_outputs


        # Substrate
        hiddenLayersWidth: int = self.n_inputs

        self.substrateNeurons: List[NeuronGene] = []
        for y in self.layers:
            
            if y == -1.0:
                for x in np.linspace(-1.0, 1.0, num=self.n_inputs):
                    self.substrateNeurons.append(NeuronGene(NeuronType.INPUT, len(self.substrateNeurons), y, x))        

            elif y == 1.0:
                for x in np.linspace(-1.0, 1.0, num=self.n_outputs):
                    self.substrateNeurons.append(NeuronGene(NeuronType.OUTPUT, len(self.substrateNeurons), y, x))
            
            else:
                for x in np.linspace(-1.0, 1.0, num=hiddenLayersWidth):
                    self.substrateNeurons.append(NeuronGene(NeuronType.HIDDEN, len(self.substrateNeurons), y, x))


        print("Nodes in substrate: {}".format(len(self.substrateNeurons)))

        population_configuration._data["n_inputs"] = cppnInputs
        population_configuration._data["n_outputs"] = cppnOutputs
        self.neat = NEAT(population_configuration, mutation_rates)

    def epoch(self, update_data: PopulationUpdate) -> List[Phenotype]:
        self.neat.population.updatePopulation(update_data)

        print("Reproducing")
        cppns = self.neat.population.reproduce()
        substrates = []
        for i, c in enumerate(cppns):
            # print("epoch:", i)
            substrates.append(self.createSubstrate(c))
            
            print("\rCreated substrates: %d/%d"%(i, len(cppns)), end='')

        print("")

        return substrates

    # def update(self, X):


    def createSubstrate(self, cppn: Genome) -> Phenotype:
            
        cppnPheno = cppn.createPhenotype()

        # Visualize().update(cppnPheno)

        graph = nx.DiGraph()
        # graph.add_nodes_from([(n.ID,  {"activation": n.activation, "type": n.neuronType}) for n in self.substrateNeurons if n.neuronType != NeuronType.HIDDEN])
        graph.add_nodes_from([(n.ID,  {"activation": n.activation, "type": n.neuronType, "pos": (n.x, n.y)}) for n in self.substrateNeurons])
        
        paths = [list(nx.all_simple_paths(cppnPheno.graph, source=n[0], target=[o[0] for o in cppnPheno.out_nodes])) for n in cppnPheno.in_nodes]
        num_of_paths = len([p for p in paths if len(p) > 0])
        # print("Nr. of paths in cppn: {}".format(num_of_paths))
        if num_of_paths == 0:
            return Phenotype(graph, cppn.ID)

        # nrOfInputs = self.n_inputs
        # nrOfOutputs = self.n_outputs

        layers = [list(g) for k, g in groupby(self.substrateNeurons, lambda n: n.y)]

        coordinates = []
        for l in layers:
            singleLayer = np.array([n for n in l])
            coordinates.append(singleLayer)
        coordinates = np.array(coordinates)

        for i in range(coordinates.shape[0] - 1):
            leftNeuronLayer = coordinates[i]

            X = np.array([(n.x, n.y) for n in leftNeuronLayer])
            X_IDs = np.array([n.ID for n in leftNeuronLayer])
            X_data = {"data": X, "IDs": X_IDs}

            for j in range(i+1, coordinates.shape[0]):
                rightNeuronLayer = coordinates[j]

                Y = np.array([(n.x, n.y) for n in rightNeuronLayer])
                Y_IDs = np.array([n.ID for n in rightNeuronLayer])
                Y_data = {"data": Y, "IDs": Y_IDs}

                substrateCUDA = SubstrateCUDA(cppnPheno)
                outputs = substrateCUDA.update(X_data, Y_data)

                graph.add_weighted_edges_from(outputs)

        # isolated_hidden = [n for n in nx.isolates(graph) if graph.nodes[n]['type'] == NeuronType.HIDDEN]
        # graph.remove_nodes_from(isolated_hidden)

        return Phenotype(graph, cppn.ID)
