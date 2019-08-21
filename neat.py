from typing import List, Set, Dict, Tuple, Optional

import random
from random import randint

import math
import numpy as np
from enum import Enum

import itertools
from copy import copy
from copy import deepcopy
from operator import attrgetter
from collections import OrderedDict

from prettytable import PrettyTable

import pickle

# import genes
from neat.genes import NeuronType, Genome, LinkGene, NeuronGene, MutationRates, Phase, SpeciationType
from neat.phenotypes import Phenotype
# from neat.population import PopulationConfiguration
# from neat.SpeciatedPopulation import SpeciatedPopulation
from neat.mapelites import MapElites, MapElitesConfiguration
from neat.innovations import Innovations


class NEAT:

    def __init__(self, numberOfGenomes: int, numOfInputs: int, numOfOutputs: int,
        populationConfiguration: MapElitesConfiguration, mutationRates: MutationRates=MutationRates()) -> None:

        self.innovations = Innovations()

        self.mutationRates: MutationRates = mutationRates
        self.phenotypes: List[Phenotype] = []


        self.phase: Phase = Phase.COMPLEXIFYING
        self.speciationType: SpeciationType = SpeciationType.NOVELTY

        self.milestone: float = 0.01



        links: List[LinkGene] = []

        # self.population = SpeciatedPopulation(numberOfGenomes, mutationRates)
        self.population = MapElites(numberOfGenomes, numOfInputs, numOfOutputs, self.innovations, mutationRates, populationConfiguration)

    def epoch(self, fitness: List[float], features: Optional[List[float]] = None) -> List[Phenotype]:
        
        self.population.updateFitness(fitness)

        genomes = self.population.reproduce()
        
        newPhenotypes = []
        for genome in genomes:
            newPhenotypes.append(genome.createPhenotype())

        self.phenotypes = newPhenotypes

        return self.phenotypes