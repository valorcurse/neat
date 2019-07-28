from __future__ import annotations

from typing import List, Set, Dict, Tuple, Optional, Any

import random
from random import randint

import math
from math import cos, sin, atan, ceil, floor
from sklearn.preprocessing import normalize
from enum import Enum
import itertools
import pickle

import numpy as np
from matplotlib import pyplot
import matplotlib.patches as patches

from prettytable import PrettyTable

from neat.phenotypes import CNeuralNet, SLink, SNeuron, NeuronType

class Species:
    pass

class Phase(Enum):
    COMPLEXIFYING = 0
    PRUNING = 1

class SpeciationType(Enum):
    COMPATIBILITY_DISTANCE = 0
    NOVELTY = 1

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
        self.crossoverRate = 0.7

        self.newSpeciesTolerance = 2.0

        self.chanceToMutateBias = 0.7

        self.chanceToAddNeuron = 0.05
        # self.chanceToAddNeuron = 0.5
        self.chanceToAddLink = 0.4
        # self.chanceToAddLink = 0.8

        self.chanceToAddRecurrentLink = 0.05

        self.chanceToDeleteNeuron = 0.05
        self.chanceToDeleteLink = 0.4

        self.chanceToMutateWeight = 0.8
        self.mutationRate = 0.9
        self.probabilityOfWeightReplaced = 0.1
        self.maxWeightPerturbation = 0.5

        self.mpcMargin = 20

class InnovationType(Enum):
    NEURON = 0
    LINK = 1

class SInnovation:
    def __init__(self, innovationType: InnovationType, innovationID: int, start: Optional[int], end: Optional[int], neuronID: Optional[int]) -> None:
        self.innovationType = innovationType
        self.innovationID = innovationID
        self.start = start
        self.end = end
        self.neuronID = neuronID

    def __eq__(self, other: object) -> bool:
        return self.innovationType == other if isinstance(other, SInnovation) else NotImplemented

class NeuronGene:
    def __init__(self, neuronType: NeuronType, ID: int, y: float, innovationID: int) -> None:
        self.neuronType = neuronType
        self.ID = ID
        self.splitY = y
        self.innovationID = innovationID
        
        self.activation = self.tanh

        self.activations = [self.sigmoid, self.tanh, self.sin, self.cos, self.gaussian]

    def sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))   # Sigmoid
    
    def tanh(self, x: float) -> float:
        return np.tanh(x)                   # Tanh
    
    def relu(self, x: float) -> float:
        return np.maximum(x, 0)             # ReLu
    
    def leakyRelu(self, x: float) -> float:
        return x if x > 0.0 else x * 0.01   # Leaky ReLu
    
    def sin(self, x: float) -> float:
        return np.sin(x)                    # Sin
    
    def cos(self, x: float) -> float:
        return np.cos(x)                     # Cos

    def gaussian(self, x: float) -> float:
        return math.exp((-x)**2)

    def __eq__(self, other: Any) -> bool:
        return other is not None and self.innovationID == other.innovationID

class LinkGene:

    def __init__(self, fromNeuron: NeuronGene, toNeuron: NeuronGene, enabled: bool, innovationID: int, weight: float, recurrent: bool=False):
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron

        self.weight = weight

        self.enabled = enabled

        self.recurrent = recurrent

        self.innovationID = innovationID

    def __lt__(self, other: LinkGene) -> bool:
        return self.innovationID < other.innovationID

    def __eq__(self, other: object) -> bool:
        return self.innovationID == other.innovationID if isinstance(other, LinkGene) else NotImplemented

class Innovations:
    def __init__(self) -> None:
        self.listOfInnovations: List[SInnovation] = []

        self.currentNeuronID = 1

    def createNewLinkInnovation(self, fromID: int, toID: int) -> int:
        ID: int = innovations.checkInnovation(fromID, toID, InnovationType.LINK)

        if (ID == -1):
            newInnovation = SInnovation(InnovationType.LINK, len(self.listOfInnovations), fromID, toID, None)
            self.listOfInnovations.append(newInnovation)
            ID = len(self.listOfInnovations) - 1

        return ID;

    def createNewLink(self, fromNeuron: NeuronGene, toNeuron: NeuronGene, enabled: bool, weight: float, recurrent: bool=False) -> LinkGene:
        ID = self.createNewLinkInnovation(fromNeuron.ID, toNeuron.ID)

        return LinkGene(fromNeuron, toNeuron, enabled, ID, weight, recurrent)

    def createNewNeuronInnovation(self, fromID: Optional[int], toID: Optional[int], neuronID: int) -> int:
        ID: int = innovations.checkInnovation(
            fromID, 
            toID, 
            InnovationType.NEURON,
            neuronID)
        
        if (ID == -1):
            newInnovation = SInnovation(InnovationType.NEURON, len(self.listOfInnovations),
                                        fromID, toID, neuronID)

            self.listOfInnovations.append(newInnovation)

            ID = len(self.listOfInnovations) - 1

        return ID;

    def createNewNeuron(self, y: float, neuronType: NeuronType, fromNeuron: Optional[NeuronGene] = None, toNeuron: Optional[NeuronGene] = None, neuronID: Optional[int] = None) -> NeuronGene:
        if (neuronID is None):
            neuronID = self.currentNeuronID
            self.currentNeuronID += 1
        
        fromID = fromNeuron.ID if fromNeuron else None
        toID = toNeuron.ID if toNeuron else None

        innovationID = self.createNewNeuronInnovation(fromID, toID, neuronID)

        return NeuronGene(neuronType, neuronID, y, innovationID)
    
    def checkInnovation(self, start: Optional[int], end: Optional[int], innovationType: InnovationType, neuronID: Optional[int] = None) -> int:
        matchingInnovations = [innovation for innovation in self.listOfInnovations 
                if innovation.start == start 
                and innovation.end == end
                and innovation.neuronID == neuronID
                and innovation.innovationType == innovationType]


        return matchingInnovations[0].innovationID if len(matchingInnovations) > 0 else -1

    def printTable(self) -> None:
        table = PrettyTable(["ID", "type", "start", "end", "neuron ID"])
        for innovation in self.listOfInnovations:
            if (innovation.innovationType == InnovationType.NEURON and innovation.neuronID is not None and innovation.neuronID < 0):
                continue

            table.add_row([
                innovation.innovationID,
                innovation.innovationType, 
                innovation.start if innovation.start else "None", 
                innovation.end if innovation.end else "None", 
                innovation.neuronID])

        print(table)



# Global innovations database
global innovations
innovations = Innovations()

class Genome:

    # def __init__(self, ID: int, neurons: List[NeuronGene], links: List[LinkGene], inputs: int, 
    def __init__(self, ID: int, numberOfNeurons: List[NeuronGene], links: List[LinkGene], inputs: int, 
        outputs: int, parents: List[Genome]=[]) -> None:
        self.ID = ID
        self.parents = parents

        # Are these necessary?
        self.inputs = inputs
        self.outputs = outputs
        ##########################

        self.links = pickle.loads(pickle.dumps(links, -1))
        self.neurons = pickle.loads(pickle.dumps(neurons, -1))
        
        if (len(self.neurons) == 0):
            for n in range(inputs):
                newNeuron = innovations.createNewNeuron(0.0, NeuronType.INPUT, neuronID = -n-1)
                self.neurons.append(newNeuron)

            for n in range(outputs):
                newNeuron = innovations.createNewNeuron(1.0, NeuronType.OUTPUT, neuronID = -inputs-n-1)
                self.neurons.append(newNeuron)

        self.fitness: float = 0.0
        self.adjustedFitness: float = 0.0
        self.novelty: float = 0.0
        
        self.milestone: float = 0.0

        # For printing
        self.distance: float = 0.0
        
        self.species: Optional[Species] = None            

    def __lt__(self, other: Genome) -> bool:
        return self.fitness < other.fitness

    def __deepcopy__(self, memodict={}):
        copy_object = Genome(self.ID, deepcopy(self.neurons), deepcopy(self.links), self.inputs, self.outputs, self.parents)
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
            [l.innovationID for l in self.links] + [l.innovationID for l in other.links]))
        combinedIndexes.sort()
        
        selfLinksDict = {l.innovationID: l for l in self.links}
        otherLinksDict = {l.innovationID: l for l in other.links}

        for i in combinedIndexes:
            selfLink = selfLinksDict.get(i)
            otherLink = otherLinksDict.get(i)



            if ((selfLink is None) or (otherLink is None)):
                disjointedLinks += 1.0
            else:
                weightDifferences.append(math.fabs(selfLink.weight - otherLink.weight))
        
        longestLinks = max(1.0, max(len(other.links), len(self.links)))
        # longestLinks = 1.0 if longestLinks <= 20 else longestLinks
        # weightDifference = 0.0 if len(weightDifferences) == 0 else np.mean(weightDifferences)
        weightDifference = 0.0 if len(weightDifferences) == 0 else np.sum(weightDifferences)

        linkDistance = (disjointRate * disjointedLinks / longestLinks) + weightDifference * matchedRate
        # print(linkDistance)

        disjointedNeurons = 0.0
        biasDifferences = []

        combinedNeurons = list(set(
            [n.innovationID for n in self.neurons if n.neuronType == NeuronType.HIDDEN] + 
            [n.innovationID for n in other.neurons if n.neuronType == NeuronType.HIDDEN]))
        combinedNeurons.sort()
        
        selfNeuronsDict = {n.innovationID: n for n in self.neurons}
        otherNeuronDict = {n.innovationID: n for n in other.neurons}


        for i in combinedNeurons:
            selfNeuron = selfNeuronsDict.get(i)
            otherNeuron = otherNeuronDict.get(i)


                # otherNeuron.innovationID if otherNeuron else "None"))

            if (selfNeuron is None or otherNeuron is None):
                disjointedNeurons += 1.0
            else:
                biasDifferences.append(math.fabs(selfNeuron.bias - otherNeuron.bias))

        longestNeurons = max(1.0, max(len(other.neurons), len(self.neurons)))
        # longestNeurons = 1.0 if longestNeurons <= 20 else longestNeurons
        biasDifference = 0.0 if len(biasDifferences) == 0 else np.mean(biasDifferences)

        neuronDistance = (disjointRate * disjointedNeurons / longestNeurons) + biasDifference * matchedRate

        distance: float = linkDistance + neuronDistance
        self.distance = distance
        
        return distance
        # return linkDistance

    def addRandomLink(self, chanceOfLooped: float) -> None:

        fromNeuron: Optional[NeuronGene] = None
        toNeuron: Optional[NeuronGene] = None
        recurrent: bool = False

        # Add recurrent link
        # if (random.random() < chanceOfLooped and len(self.neurons) > (self.inputs + self.outputs)):
        #     possibleNeurons = [n for n in self.neurons
        #         if not n.recurrent and n.neuronType == NeuronType.HIDDEN]

        #     if (len(possibleNeurons) == 0):
        #         return

        #     loopNeuron = random.choice(possibleNeurons)
        #     fromNeuron = toNeuron = loopNeuron
        #     recurrent = loopNeuron.recurrent = True

        # else:

        keepLoopRunning = True
        fromNeurons = [neuron for neuron in self.neurons
                       if (neuron.neuronType in [NeuronType.INPUT, NeuronType.HIDDEN])]

        toNeurons = [neuron for neuron in self.neurons
                     if (neuron.neuronType in [NeuronType.OUTPUT, NeuronType.HIDDEN])]
        
        triesToAddLink = 10  
        while (triesToAddLink > 0):
            
            fromNeuron = random.choice(fromNeurons)
            toNeuron = random.choice(toNeurons)

            # If link already exists
            alreadyExists = next(
                (l for l in self.links if (l.fromNeuron.ID == fromNeuron.ID) and (l.toNeuron.ID == toNeuron.ID)), 
                None)

            if (not alreadyExists and fromNeuron.ID != toNeuron.ID and fromNeuron.splitY < toNeuron.splitY):
                break
            else:
                fromNeuron = toNeuron = None

            triesToAddLink -= 1

        if (fromNeuron is None or toNeuron is None):
            return

        # if (fromNeuron.splitY > toNeuron.splitY):
            # recurrent = True

        self.addLink(fromNeuron, toNeuron, recurrent)
    
    def addLink(self, fromNeuron: NeuronGene, toNeuron: NeuronGene, weight: float = 1.0, recurrent: bool = False) -> None:
        link = innovations.createNewLink(fromNeuron, toNeuron, True, 1.0, recurrent)
        self.links.append(link)


    def removeRandomLink(self) -> None:
        if (len(self.links) == 0):
            return

        randomLink = random.choice(self.links)

        self.removeLink(randomLink)

    def removeLink(self, link: LinkGene) -> None:

        fromNeuron = link.fromNeuron

        self.links.remove(link)

        if fromNeuron.neuronType == NeuronType.HIDDEN and not self.isNeuronValid(fromNeuron):
            self.removeNeuron(fromNeuron)

        toNeuron = link.toNeuron
        if toNeuron.neuronType == NeuronType.HIDDEN and not self.isNeuronValid(toNeuron):
            self.removeNeuron(toNeuron)


    def addNeuron(self) -> None:

        if (len(self.links) < 1):
            return

        maxRand = len(self.links)

        possibleLinks = [l for l in self.links[:maxRand] if l.enabled]

        if (len(possibleLinks) == 0):
            return

        chosenLink = random.choice(possibleLinks)
        

        originalWeight = chosenLink.weight
        fromNeuron = chosenLink.fromNeuron
        toNeuron = chosenLink.toNeuron

        newDepth = (fromNeuron.splitY + toNeuron.splitY) / 2

        newNeuron = innovations.createNewNeuron(newDepth, NeuronType.HIDDEN, fromNeuron, toNeuron)

        self.addLink(fromNeuron, newNeuron)
        self.addLink(newNeuron, toNeuron)

        self.removeLink(chosenLink)

        self.neurons.append(newNeuron)
        self.neurons.sort(key=lambda x: x.splitY, reverse=False)


    def removeRandomNeuron(self) -> None:
        # Get all the hidden neurons which do not have multiple incoming AND outgoing links

        possibleNeurons = [n for n in self.neurons if n.neuronType == NeuronType.HIDDEN 
            and ((len(self.getLinksOut(n)) == 1 and len(self.getLinksIn(n)) >= 1)
            or (len(self.getLinksOut(n)) >= 1 and len(self.getLinksIn(n)) == 1))]

        if (len(possibleNeurons) == 0):
            return

        randomNeuron = random.choice(possibleNeurons)
        self.removeNeuron(randomNeuron)

    def removeNeuron(self, neuron: NeuronGene) -> None:
        linksIn = self.getLinksIn(neuron)
        linksOut = self.getLinksOut(neuron)
        
        if len(linksIn) > 1 and len(linksOut) > 1:
            # if random.random() < 0.1:
            for link in neuron.linksIn:
                self.removeLink(link)

            for link in neuron.linksOut:
                self.removeLink(link)
        else:
            if len(linksOut) == 1:
                patchThroughNeuron = linksOut[0].toNeuron


                for link in linksIn:
                    originNeuron = link.fromNeuron

                    self.removeLink(link)
                    self.addLink(originNeuron, patchThroughNeuron, link.weight)

            elif len(linksIn) == 1:
                originNeuron = linksIn[0].fromNeuron

                for link in linksIn:
                    patchThroughNeuron = link.toNeuron

                    self.removeLink(link)
                    self.addLink(originNeuron, patchThroughNeuron, link.weight)
        
        if neuron in self.neurons:
            self.neurons.remove(neuron)


    def mutateWeights(self, mutationRates: MutationRates) -> None:
            for link in self.links:
                if (random.random() > (1 - mutationRates.chanceToMutateWeight)):
                    continue

                if (random.random() < mutationRates.mutationRate):
                    link.weight += random.gauss(0.0, mutationRates.maxWeightPerturbation)
                    link.weight = min(1.0, max(-1.0, link.weight))

                # elif (random.random() < replacementProbability):
                else:
                    link.weight = random.gauss(0.0, mutationRates.maxWeightPerturbation)


    def mutateBias(self, mutationRates: MutationRates) -> None:
        neurons = [n for n in self.neurons if n.neuronType in [NeuronType.HIDDEN, NeuronType.OUTPUT]]
        for n in neurons:
            if (random.random() > (1 - mutationRates.chanceToMutateWeight)):
                    continue

            if (random.random() < mutationRates.mutationRate):
                n.bias += random.gauss(0.0, mutationRates.maxWeightPerturbation)
                n.bias = min(1.0, max(-1.0, n.bias))
            else:
            # elif (random.random() < mutationRates.replacementProbability):
                n.bias = random.gauss(0.0, mutationRates.maxWeightPerturbation)

    def mutateActivation(self, mutationRates: MutationRates) -> None:
        neurons = [n for n in self.neurons if n.neuronType in [NeuronType.HIDDEN, NeuronType.OUTPUT]]
        for n in neurons:
            if (random.random() > (1 - mutationRates.chanceToAddNeuron)):
                continue

            n.activation = random.choice(n.activations)


    def mutate(self, phase: Phase, mutationRates: MutationRates) -> None:
        # div = max(1,(self.chanceToAddNeuron*2 + self.chanceToAddLink*2))
        # r = random.random()
        # if r < (self.chanceToAddNeuron/div):
        #     baby.addNeuron()
        # elif r < ((self.chanceToAddNeuron + self.chanceToAddNeuron)/div):
        #     baby.removeNeuron()
        # elif r < ((self.chanceToAddNeuron + self.chanceToAddNeuron +
        #            self.chanceToAddLink)/div):
        #     baby.addLink(self.chanceToAddRecurrentLink,
        #              self.numOfTriesToFindLoopedLink, self.numOfTriesToAddLink)
        # elif r < ((self.chanceToAddNeuron + self.chanceToAddNeuron +
        #            self.chanceToAddLink + self.chanceToAddLink)/div):
        #     baby.removeLink()


        # if phase == Phase.COMPLEXIFYING:
        if (random.random() < mutationRates.chanceToDeleteNeuron):
            self.removeRandomNeuron()

        if (random.random() < mutationRates.chanceToDeleteLink):
            self.removeRandomLink()

        if (random.random() < mutationRates.chanceToAddNeuron):
            self.addNeuron()

        if (random.random() < mutationRates.chanceToAddLink):
            self.addRandomLink(mutationRates.chanceToAddRecurrentLink)

        # elif phase == Phase.PRUNING:

        self.mutateWeights(mutationRates)
        # self.mutateBias(mutationRates)

        self.mutateActivation(mutationRates)

        self.links.sort()

    # def createPhenotype(self) -> CNeuralNet:
    #     phenotypeNeurons = []

    #     for neuron in self.neurons:
    #         newNeuron = SNeuron(neuron.neuronType,
    #                     neuron.ID,
    #                     neuron.activation,
    #                     neuron.splitY)

    #         phenotypeNeurons.append(newNeuron)

    #     for link in self.links:
    #         if (link.enabled):
    #             fromNeuron = next((neuron
    #                                for neuron in phenotypeNeurons if (neuron.ID == link.fromNeuron.ID)), None)
    #             toNeuron = next((neuron
    #                              for neuron in phenotypeNeurons if (neuron.ID == link.toNeuron.ID)), None)

    #             if (not fromNeuron) or (not toNeuron):
    #                 continue

    #             tmpLink = SLink(fromNeuron,
    #                             toNeuron,
    #                             link.weight,
    #                             link.recurrent)

    #             toNeuron.linksIn.append(tmpLink)

    #     return CNeuralNet(phenotypeNeurons, self.ID)

    def createPhenotype(self) -> CNeuralNet:
        phenotypeNeurons = []

        for neuron in self.neurons:
            newNeuron = SNeuron(neuron.neuronType,
                        neuron.ID,
                        neuron.activation,
                        neuron.splitY)

            phenotypeNeurons.append(newNeuron)

        for link in self.links:
            if (link.enabled):
                fromNeuron = next((neuron
                                   for neuron in phenotypeNeurons if (neuron.ID == link.fromNeuron.ID)), None)
                toNeuron = next((neuron
                                 for neuron in phenotypeNeurons if (neuron.ID == link.toNeuron.ID)), None)

                if (not fromNeuron) or (not toNeuron):
                    continue

                tmpLink = SLink(fromNeuron,
                                toNeuron,
                                link.weight,
                                link.recurrent)

                toNeuron.linksIn.append(tmpLink)

        return CNeuralNet(phenotypeNeurons, self.ID)

