from typing import List

from neat.phenotypes import Phenotype

class Evaluation:

    def __init__(self):
        self.num_of_envs = 0

    def evaluate(self, phenotypes: List[Phenotype]):
        pass