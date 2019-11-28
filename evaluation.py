from typing import List, Tuple

import numpy as np

from neat.phenotypes import Phenotype

class Evaluation:

    def __init__(self):
        self.num_of_envs = 0

    def evaluate(self, phenotypes: List[Phenotype]) -> Tuple[np.ndarray, np.ndarray]:
        pass