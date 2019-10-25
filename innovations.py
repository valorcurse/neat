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

    def createNewNeuronInnovation(self, neuronType: NeuronType, fromID: Optional[int], toID: Optional[int]) -> int:
        matchingInnovation = self.checkInnovation(fromID, toID, InnovationType.NEURON)
        # ID: int = len(self.listOfInnovations) if neuronType != NeuronType.HIDDEN or matchingInnovation == -1 else -1
        ID: int = len(self.listOfInnovations) if matchingInnovation == -1 else -1

        newInnovation = Innovation(InnovationType.NEURON, ID, fromID, toID)

        self.listOfInnovations.append(newInnovation)
        
        return ID

    def checkInnovation(self, start: Optional[int], end: Optional[int], innovationType: InnovationType, neuronID: Optional[int] = None) -> int:
        for innovation in self.listOfInnovations:
            if innovation.start == start and innovation.end == end and innovation.innovationType == innovationType:
                return innovation.innovationID

        return -1