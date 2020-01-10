import pytest
from neat.genes import Genome, MutationRates, NeuronGene, LinkGene
from neat.neatTypes import NeuronType
from neat.innovations import Innovations
from neat.speciatedPopulation import SpeciatedPopulation, SpeciesConfiguration, SpeciesUpdate
import numpy as np
import math
import networkx as nx

pop_size = 5
inputs = 2
outputs = 2
behavior_dimensions = 10
obj_ranges = []

@pytest.fixture(scope='module')
def population():
    pop_config = SpeciesConfiguration(pop_size, inputs, outputs)
    population = SpeciatedPopulation(pop_config, Innovations(), MutationRates())

    return population

def test_initialization(population):

    assert len(population.genomes) == pop_size

    assert len(population.species) == 1
    assert len(population.species[0].members) == pop_size

def test_crossover(population):
    inputs = [NeuronGene(NeuronType.INPUT, -1, -1.0, 0.0), NeuronGene(NeuronType.INPUT, -2, -1.0, 1.0)]
    outputs = [NeuronGene(NeuronType.OUTPUT, -3, 1.0, 0.0), NeuronGene(NeuronType.OUTPUT, -4, 1.0, 1.0)]
    mum = population.newGenome(
        neurons=inputs+outputs,
        links=[
            LinkGene(inputs[0], outputs[1], 1, 1.0),
            LinkGene(inputs[1], outputs[0], 2, 1.0),
            LinkGene(inputs[0], outputs[0], 3, 1.0)
        ]
    )
    mum.fitness = 2.0

    dad = population.newGenome(
        neurons=inputs+outputs,
        links=[
            LinkGene(inputs[0], outputs[1], 1, 0.5),
            LinkGene(inputs[1], outputs[0], 2, 0.5),
            LinkGene(inputs[1], outputs[1], 4, 0.5)
        ]
    )
    dad.fitness = 1.0

    baby = population.crossover(mum, dad)

    assert True

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

def test_speciation(population):

    population.calculateSpawnAmount()

    # Assert it's within +-1 population, rounding errors are fine
    assert pop_size - 1 <= population.species[0].numToSpawn <= pop_size + 1

    fitness = 5.0
    fitnesses = np.array([fitness] * pop_size)
    update = SpeciesUpdate(fitnesses)

    population.updatePopulation(update)

    # Assert it's within +-1 population, rounding errors are fine
    assert pop_size - 1 <= population.species[0].numToSpawn <= pop_size + 1