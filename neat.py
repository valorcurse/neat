from typing import List

import neat.innovations
from neat.phenotypes import Phenotype
from neat.mapelites import MapElites, MapElitesConfiguration
from neat.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration
from neat.population import PopulationUpdate, PopulationConfiguration, Population
from neat.genes import MutationRates, Phase, SpeciationType

class NEAT:

    def __init__(self, population_configuration: PopulationConfiguration, mutation_rates: MutationRates=MutationRates()) -> None:

        self.population_configuration = population_configuration
        self.innovations = neat.innovations.Innovations()

        self.mutation_rates: MutationRates = mutation_rates
        self.phenotypes: List[Phenotype] = []


        self.phase: Phase = Phase.COMPLEXIFYING
        self.speciationType: SpeciationType = SpeciationType.NOVELTY

        self.milestone: float = 0.01


        # If using MapElites
        if isinstance(self.population_configuration, MapElitesConfiguration):
            self.population: Population = MapElites(population_configuration, self.innovations, mutation_rates)

        # If using regular speciated population
        elif isinstance(self.population_configuration, SpeciesConfiguration):
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