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

    def __repr__(self):
        return "Innovation: {} | ID: {} | from: {} | to: {})".format(self.innovationType, self.innovationID, self.start, self.end)

class Innovations:
    def __init__(self) -> None:
        self.listOfInnovations: List[Innovation] = []

        self.currentNeuronID = 1

    def createNewLinkInnovation(self, fromID: int, toID: int) -> int:
        ID: int = self.checkInnovation(fromID, toID, InnovationType.LINK)

        if (ID == -1):
            ID = len(self.listOfInnovations)
            newInnovation = Innovation(InnovationType.LINK, ID, fromID, toID)
            self.listOfInnovations.append(newInnovation)

        return ID

    def createNewNeuronInnovation(self, fromID: Optional[int], toID: Optional[int]) -> int:
        ID = self.checkInnovation(fromID, toID, InnovationType.NEURON)

        if (ID == -1):
            ID = len(self.listOfInnovations)
            newInnovation = Innovation(InnovationType.NEURON, ID, fromID, toID)
            self.listOfInnovations.append(newInnovation)

        return ID

    def checkInnovation(self, start: Optional[int], end: Optional[int], innovationType: InnovationType, neuronID: Optional[int] = None) -> int:
        for innovation in self.listOfInnovations:
            if innovation.start == start and innovation.end == end and innovation.innovationType == innovationType:
                return innovation.innovationID

        return -1