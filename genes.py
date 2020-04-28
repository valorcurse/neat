from __future__ import annotations

from enum import Enum
from typing import List, Dict, Optional, Any
from icontract import invariant, require


import math
import random
import numpy as np
import networkx as nx
from numba import jit
from copy import deepcopy, copy

import neat.phenotypes
from neat.neatTypes import NeuronType
from neat.innovations import Innovations
from neat.utils import fastCopy

class Species:
    pass

class Phase(Enum):
    COMPLEXIFYING = 0
    PRUNING = 1

class SpeciationType(Enum):
    COMPATIBILITY_DISTANCE = 0
    NOVELTY = 1

class FuncsEnum(Enum):
    tanh = 0
    sin = 1
    cos = 2
    sigmoid = 3
    leakyRelu = 4
    linear = 5
    maxout = 6


class MutationRates:
    def __init__(self) -> None:

        self.newSpeciesTolerance = 5.0

        self.crossoverRate = 0.3
        self.chanceToAddNeuron = 0.03
        self.chanceToAddLink = 0.15

        self.mutationRate = 0.9
        self.chanceToMutateWeight = 0.8
        self.chanceToMutateBias = 0.7
        self.chanceToMutateActivation = 0.03

        self.chanceToDeleteLink = 0.0

        self.probabilityOfWeightReplaced = 0.1
        self.maxWeightPerturbation = 0.5

        self.mpcMargin = 20

class NeuronGene:
    def __init__(self, neuronType: NeuronType, ID: int, y: float, x: float = 0.0) -> None:
        self.neuronType = neuronType
        
        self.ID = ID
        
        self.y = y
        self.x = x
        
        # self.activation = self.relu
        # self.activation = self.leakyRelu
        self.activation = self.sigmoid
        self.bias = 0.0

        # self.activations = [self.sigmoid, self.tanh, self.sin, self.cos]
        self.activations = [self.sigmoid, self.tanh, self.sin, self.cos, self.leakyRelu, self.linear]

    def __repr__(self):
        return "NeuronGene(Type={0}, ID={1}, x={2}, y={3})".format(self.neuronType, self.ID, self.x, self.y)

    def __hash__(self):
        return hash((self.ID, self.x, self.y))


    @staticmethod
    @jit
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))   # Sigmoid

    @staticmethod
    @jit
    def tanh(x: float) -> float:
        return math.tanh(x)                   # Tanh

    @staticmethod
    @jit
    def relu(x: float) -> float:
        return max(x, 0)             # ReLu

    @staticmethod
    @jit
    def leakyRelu(x: float) -> float:
        return x if x > 0.0 else x * 0.01   # Leaky ReLu

    @staticmethod
    @jit
    def sin(x: float) -> float:
        return math.sin(x)                    # Sin

    @staticmethod
    @jit
    def cos(x: float) -> float:
        return math.cos(x)                     # Cos

    @staticmethod
    @jit
    def gaussian(x: float) -> float:
        return math.exp((-x)**2)

    @staticmethod
    @jit
    def linear(x: float) -> float:
        return x

    @staticmethod
    @jit
    def maxout(x: float) -> float:
        return x

    def __eq__(self, other: Any) -> bool:
        return other is not None and self.ID == other.ID

class LinkGene:

    def __init__(self, fromNeuron: NeuronGene, toNeuron: NeuronGene, ID: int, weight: float):
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron

        self.enabled = True

        self.weight = weight

        self.ID = ID

    def __lt__(self, other: LinkGene) -> bool:
        return self.ID < other.ID

    def __hash__(self):
        return hash((self.fromNeuron, self.toNeuron))

    def __eq__(self, other: object) -> bool:
        return self.ID == other.ID if isinstance(other, LinkGene) else NotImplemented

    def __repr__(self):
        return "LinkGene(ID={}, from={}, to={}, weight={})".format(self.ID, self.fromNeuron.ID, self.toNeuron.ID, self.weight)




class Genome:

    def __init__(self, ID: int, inputs: int, outputs: int, innovations: Innovations, neurons: List[NeuronGene] = [], links: List[LinkGene] = [], parents: List[Genome]=[]) -> None:
        self.ID = ID
        self.innovations: Innovations = innovations
        self.parents = parents

        self.links = fastCopy(links)
        # self.links = deepcopy(links)


        self.inputs = inputs
        self.outputs = outputs
        self.neurons: List[NeuronGene] = fastCopy(neurons)
        # self.neurons: List[NeuronGene] = deepcopy(neurons)

        if len(self.neurons) == 0:
            for _ in range(self.inputs):
                self._addNeuron(NeuronType.INPUT).activation = NeuronGene.linear
            for _ in range(self.outputs):
                self._addNeuron(NeuronType.OUTPUT).activation = NeuronGene.sigmoid
        # else:
        #     self.neurons = fastCopy(neurons)
            # self.neurons = deepcopy(neurons)

        self.data = {}

        self.fitness: float = -1000.0
        # self.objectives = []

        self.adjustedFitness: float = -1000.0
        self.novelty: float = -1000.0
        
        # For printing
        self.distance: float = -1000.0
        
        self.species: Optional[Species] = None

        # Check whether a link is pointing to a non existing node
        # neuron_ids = set([n.ID for n in self.neurons])
        # links_ids = set([l.fromNeuron.ID for l in self.links] + [l.toNeuron.ID for l in self.links])
        # assert len(links_ids - neuron_ids) == 0, \
        #     "Found an edge with {} neuron(s) that do not exist: {}".format(len(links_ids - neuron_ids), links_ids - neuron_ids)

    def __lt__(self, other: Genome) -> bool:
        return self.fitness < other.fitness

    def __deepcopy__(self, memo):

        # copy_object = Genome(self.ID, self.inputs, self.outputs, self.innovations, self.neurons, self.links, self.parents)
        # copy_object.data = deepcopy(self.data)
        # return copy_object
        cls = self.__class__  # Extract the class of the object
        result = cls.__new__(cls)  # Create a new instance of the object based on extracted class
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, v)

        setattr(result, "links", deepcopy(self.links, memo))
        setattr(result, "neurons", deepcopy(self.neurons, memo))

        # setattr(result, "inputs", self.inputs)
        # setattr(result, "outputs", self.outputs)
        # setattr(result, "innovations", self.innovations)
        # setattr(result, "fitness", self.fitness)
        # setattr(result, "species", self.fitness)
        # setattr(result, "distance", self.fitness)
        # for k, v in self.__dict__.items():
        #     setattr(result, k, deepcopy(v, memo))
        return result

    def __repr__(self):
        return "Genome(ID={}, neurons={}, links={})".format(self.ID, self.neurons, self.links)

    def getLinksIn(self, neuron: NeuronGene) -> List[LinkGene]:
        return [l for l in self.links if l.toNeuron == neuron]

    def getLinksOut(self, neuron: NeuronGene) -> List[LinkGene]:
        return [l for l in self.links if l.fromNeuron == neuron]

    def isNeuronValid(self, neuron: NeuronGene) -> bool:
        return len(self.getLinksIn(neuron)) >= 1 or len(self.getLinksOut(neuron)) >= 1

    def calculateCompatibilityDistance(self, other: Genome) -> float:
        disjointRate = 1.0
        matchedRate = 0.5

        n_genes = max(len(other.neurons) + len(other.links), len(self.neurons) + len(self.links))

        own_links = set(l for l in self.links)
        other_links = set(l for l in other.links)
        disjoint_links = (own_links - other_links).union(other_links - own_links)
        
        matching_links = list(zip(own_links.intersection(other_links), other_links.intersection(own_links)))
        weight_difference = 0.0
        for left, right in matching_links:
            weight_difference += math.fabs(left.weight - right.weight)

        # linkDistance = (disjointRate * len(disjoint_links) / n_genes) + weight_difference * matchedRate
        # linkDistance = (disjointRate * len(disjoint_links) / n_genes)
        linkDistance = disjointRate * len(disjoint_links) + weight_difference * matchedRate

        own_neurons = set(n.ID for n in self.neurons)
        other_neurons = set(n.ID for n in other.neurons)

        # Difference between both sets
        disjoint_neurons = (own_neurons - other_neurons).union(other_neurons - own_neurons)

        # neuronDistance = (disjointRate * len(disjoint_neurons) / n_genes)
        neuronDistance = disjointRate * len(disjoint_neurons)

        distance: float = linkDistance + neuronDistance
        self.distance = distance
        
        return self.distance

    def addRandomLink(self, mutationRates) -> bool:
        if (random.random() > mutationRates.chanceToAddLink):
            return False

        fromNeuron: Optional[NeuronGene] = None
        toNeuron: Optional[NeuronGene] = None

        fromNeurons = [neuron for neuron in self.neurons
                       if (neuron.neuronType in [NeuronType.INPUT, NeuronType.HIDDEN])]

        toNeurons = [neuron for neuron in self.neurons
                     if (neuron.neuronType in [NeuronType.OUTPUT, NeuronType.HIDDEN])]
        
        triesToAddLink = 10  
        while (triesToAddLink > 0):

            triesToAddLink -= 1

            fromNeuron = random.choice(fromNeurons)
            toNeuron = random.choice([n for n in toNeurons if n.y > fromNeuron.y])

            if fromNeuron is None or toNeuron is None:
                continue

            # If link already exists
            link = next((l for l in self.links if (l.fromNeuron.ID == fromNeuron.ID) and (l.toNeuron.ID == toNeuron.ID)), None)

            if link is not None:
                if link.enabled == False:
                    link.enabled = True
                    return True
                else:
                    continue
            else:
                self.addLink(fromNeuron, toNeuron)
                return True

        return False

    def addLink(self, fromNeuron: NeuronGene, toNeuron: NeuronGene, weight: float = 1.0) -> LinkGene:
        ID = self.innovations.createNewLinkInnovation(fromNeuron.ID, toNeuron.ID)

        link = LinkGene(fromNeuron, toNeuron, ID, weight)
        self.links.append(link)

        return link


    def removeRandomLink(self, mutationRates) -> None:
        if len(self.links) == 0 or random.random() > mutationRates.chanceToDeleteLink:
            return

        randomLink = random.choice(self.links)

        # self.removeLink(randomLink)
        randomLink.enabled = False



    # fromNeuron and toNeuron can only be None if the neuronType is NeuronType.INPUT or NeuronType.OUTPUT
    @require(lambda neuronType, fromNeuron, toNeuron: False if neuronType == NeuronType.HIDDEN and (fromNeuron is None or toNeuron is None) else True)
    def _addNeuron(self, neuronType: NeuronType, fromNeuron: Optional[NeuronGene] = None, toNeuron: Optional[NeuronGene] = None) -> NeuronGene:
        y = None
        if neuronType == NeuronType.INPUT:
            y = -1.0
        elif neuronType == NeuronType.OUTPUT:
            y = 1.0
        else:
            y = (fromNeuron.y + toNeuron.y) / 2.0

        
        fromID = fromNeuron.ID if fromNeuron else None
        toID = toNeuron.ID if toNeuron else None

        ID = None
        if neuronType == NeuronType.HIDDEN:
            ID = self.innovations.createNewNeuronInnovation(fromID, toID)
        else:
            notHidden = [n for n in self.neurons if n.neuronType != NeuronType.HIDDEN]
            ID = -len(notHidden)

        newNeuron = NeuronGene(neuronType, ID, y)

        self.neurons.append(newNeuron)

        sameYNeurons = [n for n in self.neurons if n.y == y]
        for n, x in zip(sameYNeurons, np.linspace(-1.0, 1.0, num=len(sameYNeurons))):
            n.x = x

        self.neurons.sort(key=lambda n: n.y, reverse=False)

        return newNeuron

    def addRandomNeuron(self, mutationRates) -> bool:

        if len(self.links) == 0 or random.random() > mutationRates.chanceToAddNeuron:
            return False

        random_link = random.choice(self.links)

        originalWeight = random_link.weight
        fromNeuron = random_link.fromNeuron
        toNeuron = random_link.toNeuron

        newNeuron = self._addNeuron(NeuronType.HIDDEN, fromNeuron, toNeuron)

        self.addLink(fromNeuron, newNeuron, originalWeight)
        self.addLink(newNeuron, toNeuron)

        random_link.enabled = False

        return True

    def mutateWeights(self, mutationRates: MutationRates) -> bool:
        if len(self.links) == 0 or random.random() > mutationRates.chanceToMutateWeight:
            return False

        random_link = random.choice(self.links)
        if random.random() > mutationRates.probabilityOfWeightReplaced:
            random_link.weight += np.random.normal(0, mutationRates.maxWeightPerturbation, 1)[0]
            # mutation = np.random.normal(0, mutationRates.maxWeightPerturbation, 1)[0]
            # random_link.weight = np.clip(random_link.weight + mutation, -5.0, 5.0)


        else:
            # random_link.weight = np.clip(np.random.normal(0, mutationRates.maxWeightPerturbation, 1)[0], -5.0, 5.0)
            random_link.weight = np.random.normal(0, mutationRates.maxWeightPerturbation, 1)[0]

        return True

    def mutateBias(self, mutationRates: MutationRates) -> bool:
        if random.random() > mutationRates.chanceToMutateBias:
            return False

        neurons = [n for n in self.neurons if n.neuronType in [NeuronType.HIDDEN, NeuronType.OUTPUT]]
        random_neuron = random.choice(neurons)
        random_neuron.bias += np.random.normal(0, mutationRates.maxWeightPerturbation, 1)[0]

        return True

    def mutateActivation(self, mutationRates: MutationRates) -> bool:
        if (random.random() > mutationRates.chanceToMutateActivation):
            return False

        neurons = [n for n in self.neurons if n.neuronType in [NeuronType.HIDDEN, NeuronType.OUTPUT]]
        random_neuron = random.choice(neurons)
        random_neuron.activation = random.choice(random_neuron.activations)

        return True

    def mutate(self, mutationRates: MutationRates) -> None:

        mutation_functions = [lambda: self.addRandomNeuron(mutationRates),
                              lambda: self.addRandomLink(mutationRates),
                              lambda: self.removeRandomLink(mutationRates),
                              lambda: self.mutateWeights(mutationRates),
                              lambda: self.mutateBias(mutationRates),
                              lambda: self.mutateActivation(mutationRates)]

        mutation_successful = False
        while len(mutation_functions) > 0 and not mutation_successful:
            r= random.randint(0, len(mutation_functions) - 1)
            random_mutation = mutation_functions[r]
            # random_mutation = random.choice(mutation_functions)
            # print(random_mutation)
            mutation_successful = random_mutation()

            mutation_functions.remove(random_mutation)

        # random_mutation = random.choice(mutation_functions)
        # mutation_successful = random_mutation()

        self.links.sort()


    def createPhenotype(self) -> neat.phenotypes.Phenotype:
        from neat.phenotypes import SNeuron
        pheno_graph = nx.DiGraph()
        genome_graph = nx.DiGraph()

        neurons = fastCopy(self.neurons)
        # neurons = deepcopy(self.neurons)

        for neuron in self.neurons:
            # Add all nodes to the genome graph
            genome_graph.add_node(neuron.ID, activation=neuron.activation, type=neuron.neuronType, bias=neuron.bias, pos=(neuron.x, neuron.y))

            # Only add input/output nodes to the phenotype
            if neuron.neuronType != NeuronType.HIDDEN:
                pheno_graph.add_node(neuron.ID, activation=neuron.activation, type=neuron.neuronType, bias=neuron.bias,
                                     pos=(neuron.x, neuron.y))

                neurons.remove(neuron)


        for link in self.links:
            genome_graph.add_edge(link.fromNeuron.ID, link.toNeuron.ID, weight=link.weight)

        # Find all paths between every input and output node
        for input in [n for n in self.neurons if n.neuronType == NeuronType.INPUT]:
            for output in [n for n in self.neurons if n.neuronType == NeuronType.OUTPUT]:

                for nodes_in_path in nx.all_simple_paths(genome_graph, source=input.ID, target=output.ID):
                    for i in range(len(nodes_in_path) - 1):
                        u = nodes_in_path[i]
                        v = nodes_in_path[i+1]
                        weight = genome_graph.get_edge_data(u, v)['weight']

                        # If this node hasn't been found before, add it
                        if not pheno_graph.has_node(v):
                            found_neuron = next((n for n in neurons if n.ID == v), None)

                            assert found_neuron != None, "Found an edge pointing to a neuron that does not exist."

                            pheno_graph.add_node(found_neuron.ID, activation=found_neuron.activation, type=found_neuron.neuronType,
                                                 bias=found_neuron.bias,
                                                 pos=(found_neuron.x, found_neuron.y))

                        # If this edge hasn't been added yet, do it
                        if not pheno_graph.has_edge(u ,v):
                            pheno_graph.add_edge(u, v, weight=weight)

        phenotype = neat.phenotypes.Phenotype(pheno_graph, self.ID)
        phenotype.genome = self

        return phenotype

