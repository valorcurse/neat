from __future__ import annotations

from enum import Enum
from typing import List, Dict, Optional, Any
from icontract import invariant, require


import math
import random
import numpy as np
import networkx as nx
from numba import jit
from copy import deepcopy

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


class MutationRates:
    def __init__(self) -> None:
        self.crossoverRate = 0.3

        self.newSpeciesTolerance = 1.0

        self.chanceToMutateBias = 0.7

        self.chanceToAddNeuron = 0.05
        # self.chanceToAddNeuron = 0.5
        self.chanceToAddLink = 0.15
        # self.chanceToAddLink = 0.8

        self.chanceToAddRecurrentLink = 0.05

        self.chanceToDeleteNeuron = 0.05
        self.chanceToDeleteLink = 0.4

        self.chanceToMutateWeight = 0.8
        self.mutationRate = 0.9
        self.probabilityOfWeightReplaced = 0.1
        self.maxWeightPerturbation = 0.5

        self.mpcMargin = 20

class NeuronGene:
    def __init__(self, neuronType: NeuronType, ID: int, y: float, x: float = 0.0) -> None:
        self.neuronType = neuronType
        
        self.ID = ID
        
        self.y = y
        self.x = x
        
        # self.activation = self.tanh
        self.activation = self.leakyRelu
        self.bias = 0.0

        # self.activations = [self.sigmoid, self.tanh, self.sin, self.cos]
        self.activations = [self.sigmoid, self.tanh, self.sin, self.cos, self.leakyRelu]

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
        

        self.inputs = inputs
        self.outputs = outputs
        self.neurons: List[NeuronGene] = fastCopy(neurons)

        if len(self.neurons) == 0:
            for _ in range(self.inputs):
                self._addNeuron(NeuronType.INPUT)
            for _ in range(self.outputs):
                self._addNeuron(NeuronType.OUTPUT)
        else:
            self.neurons = fastCopy(neurons)

        self.data = {}

        self.fitness: float = -1000.0
        self.adjustedFitness: float = -1000.0
        self.novelty: float = -1000.0
        
        # For printing
        self.distance: float = -1000.0
        
        self.species: Optional[Species] = None

        # Check whether a link is pointing to a non existing node
        neuron_ids = set([n.ID for n in self.neurons])
        links_ids = set([l.fromNeuron.ID for l in self.links] + [l.toNeuron.ID for l in self.links])
        assert len(links_ids - neuron_ids) == 0, \
            "Found an edge with {} neuron(s) that do not exist: {}".format(len(links_ids - neuron_ids), links_ids - neuron_ids)

    def __lt__(self, other: Genome) -> bool:
        return self.fitness < other.fitness

    def __deepcopy__(self, memodict={}):
        copy_object = Genome(self.ID, self.inputs, self.outputs, self.innovations, self.neurons, self.links, self.parents)
        copy_object.data = deepcopy(self.data)
        return copy_object

    def getLinksIn(self, neuron: NeuronGene) -> List[LinkGene]:
        return [l for l in self.links if l.toNeuron == neuron]

    def getLinksOut(self, neuron: NeuronGene) -> List[LinkGene]:
        return [l for l in self.links if l.fromNeuron == neuron]

    def isNeuronValid(self, neuron: NeuronGene) -> bool:
        return len(self.getLinksIn(neuron)) >= 1 or len(self.getLinksOut(neuron)) >= 1

    def calculateCompatibilityDistance(self, other: Genome) -> float:
        if self.ID == other.ID:
            return 0.0

        disjointRate = 1.0
        matchedRate = 0.4

        n_genes = max(len(other.neurons) + len(other.links), len(self.neurons) + len(self.links))

        own_links = set(l for l in self.links)
        other_links = set(l for l in other.links)
        disjoint_links = (own_links - other_links).union(other_links - own_links)
        
        intersecting_neurons = list(zip(own_links.intersection(other_links), other_links.intersection(own_links)))
        weight_difference = 0.0
        for left, right in intersecting_neurons:
            weight_difference += math.fabs(left.weight - right.weight)

        linkDistance = (disjointRate * len(disjoint_links) / n_genes) + weight_difference * matchedRate

        own_neurons = set(n.ID for n in self.neurons)
        other_neurons = set(n.ID for n in other.neurons)

        # Difference between both sets
        disjoint_neurons = (own_neurons - other_neurons).union(other_neurons - own_neurons)

        neuronDistance = (disjointRate * len(disjoint_neurons) / n_genes)

        distance: float = linkDistance + neuronDistance
        self.distance = distance
        
        return distance

    def addRandomLink(self, mutationRates) -> None:
        if (random.random() > mutationRates.chanceToAddLink):
            return

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
                    return
                else:
                    continue
            else:
                self.addLink(fromNeuron, toNeuron)
                return

    def addLink(self, fromNeuron: NeuronGene, toNeuron: NeuronGene, weight: float = 1.0) -> LinkGene:
        ID = self.innovations.createNewLinkInnovation(fromNeuron.ID, toNeuron.ID)

        link = LinkGene(fromNeuron, toNeuron, ID, weight)
        self.links.append(link)

        return link


    def removeRandomLink(self, mutationRates) -> None:
        if len(self.links) == 0 or random.random() > mutationRates.chanceToAddLink:
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
            ID = self.innovations.createNewNeuronInnovation(neuronType, fromID, toID)
        else:
            notHidden = [n for n in self.neurons if n.neuronType != NeuronType.HIDDEN]
            ID = -len(notHidden)

        newNeuron = NeuronGene(neuronType, ID, y)

        # newNeuron = innovations.createNewNeuron(y, neuronType, fromNeuron, toNeuron)

        self.neurons.append(newNeuron)

        sameYNeurons = [n for n in self.neurons if n.y == y]
        for n, x in zip(sameYNeurons, np.linspace(-1.0, 1.0, num=len(sameYNeurons))):
            n.x = x

        self.neurons.sort(key=lambda n: n.y, reverse=False)

        return newNeuron

    def addRandomNeuron(self, mutationRates) -> None:

        if len(self.links) == 0 or random.random() > mutationRates.chanceToAddNeuron:
            return

        random_link = random.choice(self.links)

        originalWeight = random_link.weight
        fromNeuron = random_link.fromNeuron
        toNeuron = random_link.toNeuron

        newNeuron = self._addNeuron(NeuronType.HIDDEN, fromNeuron, toNeuron)

        self.addLink(fromNeuron, newNeuron, originalWeight)
        self.addLink(newNeuron, toNeuron)

        random_link.enabled = False


    def mutateWeights(self, mutationRates: MutationRates) -> None:
            random_links = np.random.permutation(self.links)
            for link in random_links:
                if random.random() > mutationRates.chanceToMutateWeight:
                    continue

                if random.random() < mutationRates.mutationRate:
                    # link.weight += random.gauss(0.0, mutationRates.maxWeightPerturbation)
                    link.weight += np.random.normal(0, mutationRates.maxWeightPerturbation, 1)[0]
                else:
                    link.weight += np.random.normal(0, 1, 1)[0]
                    # link.weight = random.gauss(0.0, 1.0)

                return

    def mutateBias(self, mutationRates: MutationRates) -> None:
            random_neurons = np.random.permutation(self.neurons)
            for neuron in random_neurons:

                if random.random() < mutationRates.chanceToMutateBias:
                    # link.weight += random.gauss(0.0, mutationRates.maxWeightPerturbation)
                    neuron.bias += np.random.normal(0, mutationRates.maxWeightPerturbation, 1)[0]

                return

    def mutateActivation(self, mutationRates: MutationRates) -> None:
        neurons = [n for n in self.neurons if n.neuronType in [NeuronType.HIDDEN, NeuronType.OUTPUT]]
        for n in neurons:
            if (random.random() > mutationRates.chanceToAddNeuron):
                continue

            n.activation = random.choice(n.activations)


    def mutate(self, mutationRates: MutationRates) -> None:

        mutation_functions = [lambda: self.addRandomNeuron(mutationRates),
                              lambda: self.addRandomLink(mutationRates),
                              lambda: self.removeRandomLink(mutationRates),
                              lambda: self.mutateWeights(mutationRates),
                              lambda: self.mutateBias(mutationRates)]

        rand = random.randint(0, len(mutation_functions) - 1)
        mutation_functions[rand]()


        # if (random.random() < mutationRates.chanceToAddNeuron):
        #     self.addRandomNeuron()
        #
        #
        # if (random.random() < mutationRates.chanceToAddLink):
        #     self.addRandomLink()
        #
        # if (random.random() < 1.0 - mutationRates.chanceToAddLink):
        #     self.removeRandomLink()
        #
        # self.mutateWeights(mutationRates)
        # self.mutateBias(mutationRates)
        # self.mutateActivation(mutationRates)
        # self.mutateActivation(mutationRates)

        self.links.sort()


    def createPhenotype(self) -> neat.phenotypes.Phenotype:
        from neat.phenotypes import SNeuron
        pheno_graph = nx.DiGraph()
        dummy_graph = nx.DiGraph()

        neurons = fastCopy(self.neurons)

        for neuron in self.neurons:
            dummy_graph.add_node(neuron.ID, activation=neuron.activation, type=neuron.neuronType, bias=neuron.bias, pos=(neuron.x, neuron.y))

            if neuron.neuronType != NeuronType.HIDDEN:
                pheno_graph.add_node(neuron.ID, activation=neuron.activation, type=neuron.neuronType, bias=neuron.bias,
                                     pos=(neuron.x, neuron.y))

                neurons.remove(neuron)


        for link in self.links:
            dummy_graph.add_edge(link.fromNeuron.ID, link.toNeuron.ID, weight=link.weight)

        all_paths = set()
        for input in [n for n in self.neurons if n.neuronType == NeuronType.INPUT]:
            for output in [n for n in self.neurons if n.neuronType == NeuronType.OUTPUT]:

                for path in nx.all_simple_paths(dummy_graph, source=input.ID, target=output.ID):
                    for i in range(len(path) - 1):
                        u = path[i]
                        v = path[i+1]
                        weight = dummy_graph.get_edge_data(u, v)['weight']

                        if not pheno_graph.has_node(v):
                            found_neuron = next((n for n in neurons if n.ID == v), None)

                            assert found_neuron != None, "Found an edge pointing to a neuron that does not exist."

                            pheno_graph.add_node(found_neuron.ID, activation=found_neuron.activation, type=found_neuron.neuronType,
                                                 bias=found_neuron.bias,
                                                 pos=(found_neuron.x, found_neuron.y))

                        if not pheno_graph.has_edge(u ,v):
                            pheno_graph.add_edge(u, v, weight=weight)

        # added_neurons = []


        # Add input nodes to graph
        # for neuron in [n for n in self.neurons if n.neuronType == NeuronType.INPUT]:
        #     phenoGraph.add_node(neuron.ID, activation=neuron.activation, type=neuron.neuronType, bias=neuron.bias, pos=(neuron.x, neuron.y))


        # Add links from input nodes to queue
        # queue: Queue = Queue()
        # for link in [l for l in self.links if phenoGraph.has_node(l.fromNeuron.ID)]:
        #     queue.put(link)

        # Add output nodes to graph
        # for neuron in [n for n in self.neurons if n.neuronType == NeuronType.OUTPUT]:
        #     phenoGraph.add_node(neuron.ID, activation=neuron.activation, type=neuron.neuronType, bias=neuron.bias, pos=(neuron.x, neuron.y))
        #
        # nodesVisited: Dict[int, SNeuron] = {}
        # while not queue.empty():
        #     link: LinkGene = queue.get()
        #
        #     phenoGraph.add_edge(link.fromNeuron.ID, link.toNeuron.ID, weight=link.weight)
        #
        #     toNeuron = link.toNeuron
        #     if not phenoGraph.has_node(toNeuron.ID):
        #         phenoGraph.add_node(toNeuron.ID, activation=toNeuron.activation, type=toNeuron.neuronType, bias=toNeuron.bias,
        #                             pos=(toNeuron.x, toNeuron.y))
        #
        #     for l in [l for l in self.links if l.fromNeuron == toNeuron]:
        #         print(l)
        #         queue.put(l)



            # phenotypeNeurons.append(neuron)
            # phenoGraph.add_node(neuron.ID, activation=neuron.activation, type=neuron.neuronType, bias=neuron.bias, pos=(neuron.x, neuron.y))

            # nodesVisited[neuron.ID] = neuron

            # if neuron.ID not in linksDict:
            #     continue
            #
            # for link in linksDict[neuron.ID]:
            #     if not link.enabled:
            #         continue
            #
            #     fromNeuron = None
            #     # If we haven't visited this node before, add it to the queue
            #     if link.fromNeuron.ID not in nodesVisited:
            #         fromNeuron = SNeuron(link.fromNeuron)
            #         queue.put(SNeuron(link.fromNeuron))
            #
            #     else:
            #         fromNeuron = nodesVisited[link.fromNeuron.ID]



        # print(phenoGraph.nodes.data())
        # print("outputs: ", len([n for n in self.neurons if n.neuronType == NeuronType.OUTPUT]), len([n for n,d in phenoGraph.out_degree() if d == 0]))
        phenotype = neat.phenotypes.Phenotype(dummy_graph, self.ID)
        phenotype.genome = self

        return phenotype

