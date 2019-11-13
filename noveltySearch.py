import numpy as np
from scipy.spatial import cKDTree

class NoveltySearch:

    def __init__(self, num_of_behaviors):
        self.num_of_behaviors = num_of_behaviors
        self.novelty_map = np.empty((1, self.num_of_behaviors))
        # Sparseness threshold as percentage of farthest distance between 2 points
        # p_threshold: float = farthestDistance*0.03
        self.p_threshold: float = 0.01
        self.k = 10 # Nr of neighbors to compare to

    def calculate_novelty(self, behaviors) -> np.ndarray:

        behaviors = behaviors if type(behaviors) == np.ndarray else np.array([behaviors])

        sparsenesses = np.empty(behaviors.shape[0])

        for i, behavior in enumerate(behaviors):
            sparseness = 0.0
            if self.novelty_map.shape[0] > 1:
                kdtree = cKDTree(self.novelty_map)

                neighbours = kdtree.query(behavior, self.k)[0]
                neighbours = neighbours[neighbours < 1E308]

                sparseness = (1/self.k)*np.sum(neighbours)
                sparsenesses[i] = sparseness

            if (self.novelty_map.size < self.k or sparseness > self.p_threshold):
                self.novelty_map = np.vstack((self.novelty_map, behavior))

        print("Novelty map size: {}".format(self.novelty_map.shape[0]))
        print(sparsenesses)
        return sparsenesses

    def reset(self):
        self.novelty_map = np.empty((1, self.num_of_behaviors))