import pytest
from neat.neatTypes import NeuronType
from neat.genes import Genome, MutationRates, NeuronGene, LinkGene
from neat.innovations import Innovations
from neat.population import Population, PopulationConfiguration

pop_size = 5
inputs = 2
outputs = 2

@pytest.fixture(scope='module')
def population():
    pop_config = PopulationConfiguration(pop_size, inputs, outputs)
    population = Population(pop_config, Innovations(), MutationRates())

    return population


def test_crossover(population):
    inputs = [NeuronGene(NeuronType.INPUT, -1, -1.0, 0.0), NeuronGene(NeuronType.INPUT, -2, -1.0, 1.0)]
    outputs = [NeuronGene(NeuronType.OUTPUT, -3, 1.0, 0.0), NeuronGene(NeuronType.OUTPUT, -4, 1.0, 1.0)]

    mum_disjoint_link = LinkGene(inputs[0], outputs[0], 3, 1.0)
    mum = population.newGenome(
        neurons=inputs+outputs,
        links=[
            LinkGene(inputs[0], outputs[1], 1, 1.0),
            LinkGene(inputs[1], outputs[0], 2, 1.0),
            mum_disjoint_link
        ]
    )


    dad_disjoint_link = LinkGene(inputs[1], outputs[1], 4, 0.5)
    dad = population.newGenome(
        neurons=inputs+outputs,
        links=[
            LinkGene(inputs[0], outputs[1], 1, 0.5),
            LinkGene(inputs[1], outputs[0], 2, 0.5),
            dad_disjoint_link
        ]
    )

    # Test crossover when the mum is more fit
    mum.fitness = 2.0
    dad.fitness = 1.0
    baby = population.crossover(mum, dad)


    assert len(baby.links) == len(mum.links)
    assert mum_disjoint_link in baby.links

    # Test crossover when the dad is more fit
    mum.fitness = 1.0
    dad.fitness = 2.0
    baby = population.crossover(mum, dad)

    assert len(baby.links) == len(dad.links)
    assert dad_disjoint_link in baby.links

    # Test crossover when they're equally fit
    mum.fitness = 1.0
    dad.fitness = 1.0
    baby = population.crossover(mum, dad)

    # When equally fit, it chooses the simpler organism
    assert len(baby.links) == min(len(mum.links), len(dad.links))
    assert dad_disjoint_link in baby.links
