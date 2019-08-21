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
from neat.phenotypes import Phenotype
# from neat.innovations import Innovations
import neat.innovations
from neat.types import NeuronType
from neat.mapelites import MapElites, MapElitesConfiguration, MapElitesUpdate
from neat.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration
from neat.population import PopulationUpdate, PopulationConfiguration, Population
from neat.genes import Genome, LinkGene, NeuronGene, MutationRates, Phase, SpeciationType

class NEAT:

    def __init__(self, population_configuration: PopulationConfiguration, mutation_rates: MutationRates=MutationRates()) -> None:

        self.innovations = neat.innovations.Innovations()

        self.mutation_rates: MutationRates = mutation_rates
        self.phenotypes: List[Phenotype] = []


        self.phase: Phase = Phase.COMPLEXIFYING
        self.speciationType: SpeciationType = SpeciationType.NOVELTY

        self.milestone: float = 0.01


        # If using MapElites
        if isinstance(population_configuration, MapElitesConfiguration):
            self.population: Population = MapElites(population_configuration, self.innovations, mutation_rates)

        # If using regular speciated population
        elif isinstance(population_configuration, SpeciesConfiguration):
            self.population = SpeciatedPopulation(population_configuration, self.innovations, mutation_rates)

    # fitness: List[float], features: Optional[List[float]] = None
    def epoch(self, update: PopulationUpdate) -> List[Phenotype]:
        
        self.population.updatePopulation(update)

        genomes = self.population.reproduce()
        
        newPhenotypes = []
        for genome in genomes:
            newPhenotypes.append(genome.createPhenotype())

        self.phenotypes = newPhenotypes

        return self.phenotypes