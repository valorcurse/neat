import numpy as np
from scipy.spatial import cKDTree

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from neat.visualize import Visualize

class NoveltySearch:

    def __init__(self, num_of_behaviors):
        self.num_of_behaviors = num_of_behaviors
        self.novelty_map = np.empty((1, self.num_of_behaviors))
        self.novelty_dict = {}
        # Sparseness threshold as percentage of farthest distance between 2 points
        # p_threshold: float = farthestDistance*0.03
        self.p_threshold: float = 0.01
        self.k = 15 # Nr of neighbors to compare to


    def calculate_novelty(self, behaviors, fitnesses) -> (np.ndarray, np.ndarray):

        behaviors = behaviors if type(behaviors) == np.ndarray else np.array([behaviors])

        sparsenesses = np.empty(behaviors.shape[0])

        for i, (behavior, fitness) in enumerate(zip(behaviors, fitnesses)):
            sparseness = 0.0
            if self.novelty_map.shape[0] > 1:
                kdtree = cKDTree(self.novelty_map)

                neighbours = kdtree.query(behavior, self.k)[0]
                neighbours = neighbours[neighbours < 1E308]


                sparseness = (1/self.k)*np.sum(neighbours)
                sparsenesses[i] = sparseness

            if (self.novelty_map.shape[0] < self.k or sparseness > self.p_threshold):
                self.novelty_dict[tuple(behavior)] = fitness
                self.novelty_map = np.vstack((self.novelty_map, behavior))

                # plt.figure(1)
                # plt.scatter(behavior[0], behavior[1])

        # print("Novelty map size: {}".format(self.novelty_map.shape[0]))
        # print(sparsenesses)
        return sparsenesses

    def calculate_local_competition(self, behaviors, fitnesses) -> (np.ndarray, np.ndarray):

        behaviors = behaviors if type(behaviors) == np.ndarray else np.array([behaviors])

        sparsenesses = np.zeros(behaviors.shape[0])
        local_fitnesses = np.zeros(behaviors.shape[0])
        for i, (behavior, fitness) in enumerate(zip(behaviors, fitnesses)):
            sparseness = 0.0
            if self.novelty_map.shape[0] > self.k:
                kdtree = cKDTree(self.novelty_map)

                neighbours, neighbours_indexes = kdtree.query(behavior, self.k)
                neighbours = neighbours[neighbours < 1E308]


                sparseness = (1 / self.k) * np.sum(neighbours)
                sparsenesses[i] = sparseness

                local_fitness = len([n for n in kdtree.data[neighbours_indexes] if tuple(n) in self.novelty_dict and self.novelty_dict[tuple(n)] < fitness])
                local_fitnesses[i] = local_fitness

            if (self.novelty_map.shape[0] <= self.k or sparseness > self.p_threshold):
                self.novelty_dict[tuple(behavior)] = fitness
                self.novelty_map = np.vstack((self.novelty_map, behavior))



        # print("Novelty map size: {}".format(self.novelty_map.shape[0]))
        # print(sparsenesses)
        return (sparsenesses, local_fitnesses)

    def reset(self):
        self.novelty_map = np.zeros((1, self.num_of_behaviors))
        self.novelty_dict = {}