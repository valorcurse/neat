from __future__ import annotations

from enum import Enum
from typing import List, Dict, Optional, Any
from icontract import invariant, require


import random
import itertools
from queue import Queue
from copy import  deepcopy


import math
import numpy as np
import networkx as nx
from numba import jit

import neat.phenotypes
from neat.neatTypes import NeuronType
from neat.innovations import Innovations, Innovation
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


# activations = [
#     lambda x: 1.0 / (1.0 + math.exp(-x)),   # Sigmoid
#     lambda x: np.tanh(x),                   # Tanh
#     lambda x: np.maximum(x, 0),             # ReLu
#     lambda x: x if x > 0.0 else x * 0.01,   # Leaky ReLu
#     lambda x: np.sin(x),                    # Sin
#     lambda x: np.cos(x)                     # Cos
# ]


class MutationRates:
    def __init__(self) -> None:
        self.crossoverRate = 0.3

        self.newSpeciesTolerance = 5.0

        self.chanceToMutateBias = 0.7

        self.chanceToAddNeuron = 0.2
        # self.chanceToAddNeuron = 0.5
        self.chanceToAddLink = 0.5
        # self.chanceToAddLink = 0.8

        self.chanceToAddRecurrentLink = 0.05

        self.chanceToDeleteNeuron = 0.05
        self.chanceToDeleteLink = 0.4

        self.chanceToMutateWeight = 0.8
        self.mutationRate = 0.9
        self.probabilityOfWeightReplaced = 0.1
        self.maxWeightPerturbation = 0.5

        self.mpcMargin = 20


class DummyInnovations(Innovations):
    def __init__(self) -> None:
        self.listOfInnovations: List[Innovation] = []
        self.innovationNumber = 0
        self.currentNeuronID = 1

    def createNewLinkInnovation(self, fromID: int, toID: int) -> int:
        self.innovationNumber += 1
        return self.innovationNumber;


    def createNewNeuronInnovation(self, neuronType: NeuronType, fromID: Optional[int], toID: Optional[int]) -> int:
        self.innovationNumber += 1
        return self.innovationNumber;

class NeuronGene:
    def __init__(self, neuronType: NeuronType, ID: int, y: float, x: float = 0.0) -> None:
        self.neuronType = neuronType
        
        self.ID = ID
        
        self.y = y
        self.x = x
        
        self.activation = self.tanh

        # self.activations = [self.sigmoid, self.tanh, self.sin, self.cos]
        self.activations = [self.sigmoid, self.tanh, self.sin, self.cos, self.leakyRelu]

    def __repr__(self):
        return "NeuronGene(Type={0}, ID={1}, x={2}, y={3})".format(self.neuronType, self.ID, self.x, self.y)


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

    def __eq__(self, other: object) -> bool:
        return self.ID == other.ID if isinstance(other, LinkGene) else NotImplemented





# @invariant(lambda self: len([n for n in self.neurons if n.neuronType == NeuronType.INPUT]) == self.inputs, ValueError("Number of INPUT neurons incorrect."))
# @invariant(lambda self: len([n for n in self.neurons if n.neuronType == NeuronType.OUTPUT]) == self.outputs, ValueError("Number of OUTPUT neurons incorrect."))
class Genome:

    def __init__(self, ID: int, inputs: int, outputs: int, innovations: Innovations, neurons: List[NeuronGene] = [], links: List[LinkGene] = [], parents: List[Genome]=[]) -> None:
        self.ID = ID
        self.innovations: Innovations = innovations
        self.parents = parents

        self.links = fastCopy(links)
        

        self.inputs = inputs
        self.outputs = outputs
        self.neurons: List[NeuronGene] = []

        if len(self.neurons) == 0:
            for _ in range(self.inputs):
                new_neuron = self._addNeuron(NeuronType.INPUT)
                new_neuron.activation = new_neuron.tanh
            for _ in range(self.outputs):
                new_neuron = self._addNeuron(NeuronType.OUTPUT)
                new_neuron.activation = new_neuron.tanh
        else:
            self.neurons = fastCopy(neurons)

        self.fitness: float = -1000.0
        self.adjustedFitness: float = -1000.0
        self.novelty: float = -1000.0
        
        self.milestone: float = -1000.0

        # For printing
        self.distance: float = -1000.0
        
        self.species: Optional[Species] = None

    def __lt__(self, other: Genome) -> bool:
        return self.fitness < other.fitness

    def __deepcopy__(self, memodict={}):
        copy_object = Genome(self.ID, self.inputs, self.outputs, self.innovations, self.neurons, self.links, self.parents)
        return copy_object

    def getLinksIn(self, neuron: NeuronGene) -> List[LinkGene]:
        return [l for l in self.links if l.toNeuron == neuron]

    def getLinksOut(self, neuron: NeuronGene) -> List[LinkGene]:
        return [l for l in self.links if l.fromNeuron == neuron]

    def isNeuronValid(self, neuron: NeuronGene) -> bool:
        return len(self.getLinksIn(neuron)) >= 1 or len(self.getLinksOut(neuron)) >= 1

    def calculateCompatibilityDistance(self, other: Genome) -> float:
        disjointRate = 1.0
        matchedRate = 0.4

        disjointedLinks = 0.0
        weightDifferences = []

        combinedIndexes = list(set(
            [l.ID for l in self.links] + [l.ID for l in other.links]))
        combinedIndexes.sort()
        
        selfLinksDict = {l.ID: l for l in self.links}
        otherLinksDict = {l.ID: l for l in other.links}

        for i in combinedIndexes:
            selfLink = selfLinksDict.get(i)
            otherLink = otherLinksDict.get(i)


            if ((selfLink is None) or (otherLink is None)):
                disjointedLinks += 1.0
            else:
                weightDifferences.append(math.fabs(selfLink.weight - otherLink.weight))
        
        longestLinks = max(1.0, max(len(other.links), len(self.links)))
        weightDifference = 0.0 if len(weightDifferences) == 0 else np.sum(weightDifferences)

        linkDistance = (disjointRate * disjointedLinks / longestLinks) + weightDifference * matchedRate

        disjointedNeurons = 0.0

        combinedNeurons = list(set(
            [n.ID for n in self.neurons if n.neuronType == NeuronType.HIDDEN] + 
            [n.ID for n in other.neurons if n.neuronType == NeuronType.HIDDEN]))
        combinedNeurons.sort()
        
        selfNeuronsDict = {n.ID: n for n in self.neurons}
        otherNeuronDict = {n.ID: n for n in other.neurons}


        for i in combinedNeurons:
            selfNeuron = selfNeuronsDict.get(i)
            otherNeuron = otherNeuronDict.get(i)


            if (selfNeuron is None or otherNeuron is None):
                disjointedNeurons += 1.0

        longestNeurons = max(1.0, max(len(other.neurons), len(self.neurons)))

        neuronDistance = (disjointRate * disjointedNeurons / longestNeurons) + matchedRate

        distance: float = linkDistance + neuronDistance
        self.distance = distance
        
        return distance
        # return linkDistance

    def addRandomLink(self) -> None:

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

            if toNeuron is None:
                continue

            # If link already exists
            link = next(
                (l for l in self.links if (l.fromNeuron.ID == fromNeuron.ID) and (l.toNeuron.ID == toNeuron.ID)), 
                None)

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


    def removeRandomLink(self) -> None:
        if (len(self.links) == 0):
            return

        randomLink = random.choice(self.links)

        # self.removeLink(randomLink)
        randomLink.enabled = False

    # def removeLink(self, link: LinkGene) -> None:

    #     fromNeuron = link.fromNeuron

    #     self.links.remove(link)

    #     if fromNeuron.neuronType == NeuronType.HIDDEN and not self.isNeuronValid(fromNeuron):
    #         self.removeNeuron(fromNeuron)

    #     toNeuron = link.toNeuron
    #     if toNeuron.neuronType == NeuronType.HIDDEN and not self.isNeuronValid(toNeuron):
    #         self.removeNeuron(toNeuron)

    # def createNewNeuron(self, y: float, neuronType: NeuronType, fromNeuron: Optional[NeuronGene] = None, 
    #         toNeuron: Optional[NeuronGene] = None, neuronID: Optional[int] = None) -> NeuronGene:
        
    #     if (neuronID is None):
    #         neuronID = self.currentNeuronID
    #         self.currentNeuronID += 1
        
    #     fromID = fromNeuron.ID if fromNeuron else None
    #     toID = toNeuron.ID if toNeuron else None

    #     ID = self.createNewNeuronInnovation(fromID, toID, neuronID)

    #     return NeuronGene(neuronType, neuronID, ID, y)


    # fromNeuron and toNeuron can only be None if the neuronType is NeuronType.INPUT or NeuronType.OUTPUT
    @require(lambda neuronType, fromNeuron, toNeuron: False if neuronType == NeuronType.HIDDEN and (fromNeuron is None or toNeuron is None) else True)
    def _addNeuron(self, neuronType: NeuronType, fromNeuron: Optional[NeuronGene] = None, toNeuron: Optional[NeuronGene] = None) -> NeuronGene:
        y = None
        if neuronType == NeuronType.INPUT:
            y = -1.0
        elif neuronType == NeuronType.OUTPUT:
            y = 1.0
        else:
            y = (fromNeuron.y + toNeuron.y) / 2

        
        fromID = fromNeuron.ID if fromNeuron else None
        toID = toNeuron.ID if toNeuron else None


        notHidden = [n for n in self.neurons if n.neuronType != NeuronType.HIDDEN]
        ID = len(notHidden) if notHidden else self.innovations.createNewNeuronInnovation(neuronType, fromID, toID)
        newNeuron = NeuronGene(neuronType, ID, y)

        # newNeuron = innovations.createNewNeuron(y, neuronType, fromNeuron, toNeuron)

        self.neurons.append(newNeuron)

        sameYNeurons = [n for n in self.neurons if n.y == y]
        for n, x in zip(sameYNeurons, np.linspace(-1.0, 1.0, num=len(sameYNeurons))):
            n.x = x

        self.neurons.sort(key=lambda x: x.y, reverse=False)

        return newNeuron

    def addRandomNeuron(self) -> None:

        if (len(self.links) < 1):
            return

        maxRand = len(self.links)

        possibleLinks = [l for l in self.links[:maxRand]]

        if (len(possibleLinks) == 0):
            return

        chosenLink = random.choice(possibleLinks)
        

        originalWeight = chosenLink.weight
        fromNeuron = chosenLink.fromNeuron
        toNeuron = chosenLink.toNeuron

        newNeuron = self._addNeuron(NeuronType.HIDDEN, fromNeuron, toNeuron)

        self.addLink(fromNeuron, newNeuron, originalWeight)
        self.addLink(newNeuron, toNeuron)

        chosenLink.enabled = False
        # self.removeLink(chosenLink)

    # def removeRandomNeuron(self) -> None:
    #     # Get all the hidden neurons which do not have multiple incoming AND outgoing links

    #     possibleNeurons = [n for n in self.neurons if n.neuronType == NeuronType.HIDDEN 
    #         and ((len(self.getLinksOut(n)) == 1 and len(self.getLinksIn(n)) >= 1)
    #         or (len(self.getLinksOut(n)) >= 1 and len(self.getLinksIn(n)) == 1))]

    #     if (len(possibleNeurons) == 0):
    #         return

    #     randomNeuron = random.choice(possibleNeurons)
    #     self.removeNeuron(randomNeuron)

    # def removeNeuron(self, neuron: NeuronGene) -> None:
    #     linksIn = self.getLinksIn(neuron)
    #     linksOut = self.getLinksOut(neuron)
        
    #     if len(linksIn) > 1 and len(linksOut) > 1:
    #         # if random.random() < 0.1:
    #         for link in neuron.linksIn:
    #             self.removeLink(link)

    #         for link in neuron.linksOut:
    #             self.removeLink(link)
    #     else:
    #         if len(linksOut) == 1:
    #             patchThroughNeuron = linksOut[0].toNeuron


    #             for link in linksIn:
    #                 originNeuron = link.fromNeuron

    #                 self.removeLink(link)
    #                 self.addLink(originNeuron, patchThroughNeuron, link.weight)

    #         elif len(linksIn) == 1:
    #             originNeuron = linksIn[0].fromNeuron

    #             for link in linksIn:
    #                 patchThroughNeuron = link.toNeuron

    #                 self.removeLink(link)
    #                 self.addLink(originNeuron, patchThroughNeuron, link.weight)
        
    #     if neuron in self.neurons:
    #         self.neurons.remove(neuron)


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

    def mutateActivation(self, mutationRates: MutationRates) -> None:
        neurons = [n for n in self.neurons if n.neuronType in [NeuronType.HIDDEN, NeuronType.OUTPUT]]
        for n in neurons:
            if (random.random() > mutationRates.chanceToAddNeuron):
                continue

            n.activation = random.choice(n.activations)


    def mutate(self, mutationRates: MutationRates) -> None:
        if (random.random() < mutationRates.chanceToAddNeuron):
            self.addRandomNeuron()


        if (random.random() < mutationRates.chanceToAddLink):
            self.addRandomLink()

        # if (random.random() < 1.0 - mutationRates.chanceToAddLink):
        #     self.removeRandomLink()

        # self.mutateWeights(mutationRates)
        # self.mutateActivation(mutationRates)

        self.links.sort()


    def createPhenotype(self) -> neat.phenotypes.Phenotype:
        from neat.phenotypes import SNeuron
        phenoGraph = nx.DiGraph()

        queue: Queue = Queue()
        for neuronGene in [n for n in self.neurons if n.neuronType != NeuronType.HIDDEN]:
            newNeuron = SNeuron(neuronGene)
            queue.put(newNeuron)
            


        linksDict: Dict[int, List[LinkGene]] = {}
        for key, group in itertools.groupby(self.links, key=lambda x: x.toNeuron.ID):
            linksDict[key] = list(group)

        nodesVisited: Dict[int, SNeuron] = {}
        while not queue.empty():
            neuron: SNeuron = queue.get()

            # phenotypeNeurons.append(neuron)
            phenoGraph.add_node(neuron.ID, activation=neuron.activation, type=neuron.neuronType, pos=(neuron.x, neuron.y))

            nodesVisited[neuron.ID] = neuron

            if neuron.ID not in linksDict:
                continue

            for link in linksDict[neuron.ID]:
                if not link.enabled:
                    continue

                fromNeuron = None
                # If we haven't visited this node before, add it to the queue
                if link.fromNeuron.ID not in nodesVisited:
                    fromNeuron = SNeuron(link.fromNeuron)
                    queue.put(SNeuron(link.fromNeuron))

                else:
                    fromNeuron = nodesVisited[link.fromNeuron.ID]

                phenoGraph.add_edge(fromNeuron.ID, neuron.ID, weight=link.weight)

        # print(phenoGraph.nodes.data())
        # print("outputs: ", len([n for n in self.neurons if n.neuronType == NeuronType.OUTPUT]), len([n for n,d in phenoGraph.out_degree() if d == 0]))
        phenotype = neat.phenotypes.Phenotype(phenoGraph, self.ID)
        phenotype.genome = self
        return phenotype

