import pytest
from neat.genes import Genome, MutationRates, NeuronGene, LinkGene
from neat.neatTypes import NeuronType
from neat.innovations import Innovations
from neat.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration, SpeciesUpdate
import numpy as np
import math
import networkx as nx

pop_size = 2
inputs = 2
outputs = 2
behavior_dimensions = 10
obj_ranges = []

@pytest.fixture(scope='module')
def population():
    pop_config = SpeciesConfiguration(pop_size, inputs, outputs)
    mutation_rates = MutationRates()
    mutation_rates.newSpeciesTolerance = 1.0

    population = SpeciatedPopulation(pop_config, Innovations(), mutation_rates)

    return population

def test_initialization(population):

    assert len(population.genomes) == pop_size

    assert len(population.species) == 1
    assert len(population.species[0].members) == pop_size



def test_updatePopulation(population):
    fitness = 5.0

    fitnesses = np.array([fitness] * pop_size)
    update = SpeciesUpdate(fitnesses)
    population.updatePopulation(update)

    assert all([g.fitness for g in population.genomes] == fitnesses)

def test_oneSpecies(population):

    population.calculateSpawnAmount()

    # Assert it's within +-1 population, rounding errors are fine
    assert pop_size - 1 <= population.species[0].numToSpawn <= pop_size + 1

    fitness = 5.0
    fitnesses = np.array([fitness] * pop_size)
    update = SpeciesUpdate(fitnesses)

    population.updatePopulation(update)

    # Assert it's within +-1 population, rounding errors are fine
    assert pop_size - 1 <= population.species[0].numToSpawn <= pop_size + 1

def test_compatibilityDistance(population):
    population.genomes = []

    inputs = [NeuronGene(NeuronType.INPUT, -1, -1.0, 0.0), NeuronGene(NeuronType.INPUT, -2, -1.0, 1.0)]
    outputs = [NeuronGene(NeuronType.OUTPUT, -3, 1.0, 0.0), NeuronGene(NeuronType.OUTPUT, -4, 1.0, 1.0)]

    g1 = population.newGenome(
        neurons=inputs + outputs,
        links=[
            LinkGene(inputs[0], outputs[1], 1, 1.0),
            LinkGene(inputs[1], outputs[0], 2, 1.0),
        ]
    )

    g2 = population.newGenome(
        neurons=inputs + outputs,
        links=[
            LinkGene(inputs[0], outputs[1], 1, 0.5),
            LinkGene(inputs[1], outputs[0], 2, 0.5),
        ]
    )

    assert g1.calculateCompatibilityDistance(g2) == 0.5
    assert g2.calculateCompatibilityDistance(g1) == 0.5

    g1.links = [
        LinkGene(inputs[0], outputs[1], 1, 1.0),
        LinkGene(inputs[1], outputs[0], 2, 1.0),
        LinkGene(inputs[0], outputs[0], 3, 1.0)
    ]

    g2.links=[
        LinkGene(inputs[0], outputs[1], 1, 0.5),
        LinkGene(inputs[1], outputs[0], 2, 0.5),
        LinkGene(inputs[1], outputs[1], 4, 0.5)
    ]

    assert g1.calculateCompatibilityDistance(g2) == 2.5
    assert g2.calculateCompatibilityDistance(g1) == 2.5

def test_speciation(population):
    population.genomes = []

    inputs = [NeuronGene(NeuronType.INPUT, -1, -1.0, 0.0), NeuronGene(NeuronType.INPUT, -2, -1.0, 1.0)]
    outputs = [NeuronGene(NeuronType.OUTPUT, -3, 1.0, 0.0), NeuronGene(NeuronType.OUTPUT, -4, 1.0, 1.0)]

    g1 = population.newGenome(
        neurons=inputs + outputs,
        links=[
            LinkGene(inputs[0], outputs[1], 1, 1.0),
            LinkGene(inputs[1], outputs[0], 2, 1.0),
            LinkGene(inputs[0], outputs[0], 3, 1.0)
        ]
    )

    g2 = population.newGenome(
        neurons=inputs + outputs,
        links=[
            LinkGene(inputs[0], outputs[1], 1, 0.5),
            LinkGene(inputs[1], outputs[0], 2, 0.5),
            LinkGene(inputs[1], outputs[1], 4, 0.5)
        ]
    )

    assert g1.calculateCompatibilityDistance(g2) == 2.5
    assert g2.calculateCompatibilityDistance(g1) == 2.5

    assert len(population.species) == 1
    population.species[0].leader = g1

    population.speciate(g2)
    new_species = population.species[1]

    assert len(population.species) == 2
    assert len(new_species.members) == 1
    assert new_species.leader == g2


def test_elitism(population):
    population.genomes = []

    g1 = population.newGenome()
    g1.fitness = 100

    g2 = population.newGenome()
    g2.fitness = 90

    g3 = population.newGenome()
    g3.fitness = 80

    assert len(population.genomes) == 3

    population.reproduce()

    assert g1 in population.genomes
    assert g2 in population.genomes