from typing import List

from neat.neat import Species


class Population:

    def __init__(self, species: List[Species]):
        self.species = species

    def reproduce(self) -> List:
        pass