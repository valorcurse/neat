from typing import List, Optional
from enum import Enum

from neat.neatTypes import NeuronType

class InnovationType(Enum):
    NEURON = 0
    LINK = 1

class Innovation:
    def __init__(self, innovationType: InnovationType, innovationID: int, start: Optional[int], end: Optional[int]) -> None:
        self.innovationType = innovationType
        self.innovationID = innovationID
        self.start = start
        self.end = end

    def __eq__(self, other: object) -> bool:
        return self.innovationType == other if isinstance(other, Innovation) else NotImplemented

class Innovations:
    def __init__(self) -> None:
        self.listOfInnovations: List[Innovation] = []

        self.currentNeuronID = 1

    def createNewLinkInnovation(self, fromID: int, toID: int) -> int:
        ID: int = self.checkInnovation(fromID, toID, InnovationType.LINK)

        if (ID == -1):
            newInnovation = Innovation(InnovationType.LINK, len(self.listOfInnovations), fromID, toID)
            self.listOfInnovations.append(newInnovation)
            ID = len(self.listOfInnovations) - 1

        return ID;

    # def createNewLink(self, fromNeuron: NeuronGene, toNeuron: NeuronGene, weight: float) -> LinkGene:
    #     ID = self.createNewLinkInnovation(fromNeuron.ID, toNeuron.ID)

    #     return LinkGene(fromNeuron, toNeuron, ID, weight)

    def createNewNeuronInnovation(self, neuronType: NeuronType, fromID: Optional[int], toID: Optional[int]) -> int:
        matchingInnovation = self.checkInnovation(fromID, toID, InnovationType.NEURON)
        ID: int = len(self.listOfInnovations) if neuronType != NeuronType.HIDDEN or matchingInnovation == -1 else -1
        
        newInnovation = Innovation(InnovationType.NEURON, ID, fromID, toID)

        self.listOfInnovations.append(newInnovation)
        
        return ID;

    # def createNewNeuron(self, y: float, neuronType: NeuronType, fromNeuron: Optional[NeuronGene] = None, 
    #         toNeuron: Optional[NeuronGene] = None, neuronID: Optional[int] = None) -> NeuronGene:
        
    #     if (neuronID is None):
    #         neuronID = self.currentNeuronID
    #         self.currentNeuronID += 1
        
    #     fromID = fromNeuron.ID if fromNeuron else None
    #     toID = toNeuron.ID if toNeuron else None

    #     innovationID = self.createNewNeuronInnovation(fromID, toID, neuronID)

    #     return NeuronGene(neuronType, neuronID, innovationID, y)
    
    def checkInnovation(self, start: Optional[int], end: Optional[int], innovationType: InnovationType, neuronID: Optional[int] = None) -> int:
        matchingInnovations = [innovation for innovation in self.listOfInnovations 
                if innovation.start == start 
                and innovation.end == end
                and innovation.innovationType == innovationType]


        return matchingInnovations[0].innovationID if len(matchingInnovations) > 0 else -1

    # def printTable(self) -> None:
    #     table = PrettyTable(["ID", "type", "start", "end", "neuron ID"])
    #     for innovation in self.listOfInnovations:
    #         if (innovation.innovationType == InnovationType.NEURON):
    #             continue

    #         table.add_row([
    #             innovation.innovationID,
    #             innovation.innovationType, 
    #             innovation.start if innovation.start else "None", 
    #             innovation.end if innovation.end else "None"])

    #     print(table)