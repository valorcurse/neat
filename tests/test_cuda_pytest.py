from neat.phenotypes import Phenotype
from neat.neatTypes import NeuronType
from neat.phenotypes import FeedforwardCUDA
import numpy as np
import math
import networkx as nx

def test_single_edges():
    G = nx.DiGraph()
    G.add_nodes_from([
        (10, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(-1.0, -1.0)}),
        (20, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(0.0, -1.0)}),
        (30, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (60, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(-1.0, 1.0)}),
        (70, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(1.0, 1.0)})
    ])

    G.add_weighted_edges_from([
        (20, 60, 0.4),
        (30, 70, 0.1),
        ])

    phenotype = Phenotype(G, 0)

    inputs = np.array([1, 1, 1])

    feedforward_highest = FeedforwardCUDA([phenotype])

    result = feedforward_highest.update(np.array([inputs]))[0]

    answers = np.tanh([0.4, 0.1])

    np.testing.assert_array_almost_equal(result, answers)


def test_multiple_edges():
    G = nx.DiGraph()
    G.add_nodes_from([
        (10, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(-1.0, -1.0)}),
        (20, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(0.0, -1.0)}),
        (30, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (60, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(-1.0, 1.0)}),
        (70, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(1.0, 1.0)})
    ])

    G.add_weighted_edges_from([
        (20, 60, 0.4),
        (30, 60, 0.1),
        ])

    phenotype = Phenotype(G, 0)

    inputs = np.array([1, 1, 1])

    feedforward_highest = FeedforwardCUDA([phenotype])

    result = feedforward_highest.update(np.array([inputs]))[0]

    answers = np.tanh([0.5, 0.0])

    np.testing.assert_array_almost_equal(result, answers)

def test_hidden_nodes():
    G = nx.DiGraph()
    G.add_nodes_from([
        (10, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(-1.0, -1.0)}),
        (20, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(0.0, -1.0)}),
        (30, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (40, {"activation": np.tanh, "type": NeuronType.HIDDEN}),
        (60, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(-1.0, 1.0)}),
        (70, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(1.0, 1.0)})
    ])

    G.add_weighted_edges_from([
        (20, 40, 0.4),
        (40, 60, 0.1),
    ])

    phenotype = Phenotype(G, 0)

    inputs = np.array([1, 1, 1])

    feedforward_highest = FeedforwardCUDA([phenotype])

    result = feedforward_highest.update(np.array([inputs]))[0]


    answers = np.tanh([math.tanh(math.tanh(0.4)*0.1), 0.0])

    np.testing.assert_array_almost_equal(result, answers, decimal=4)

def test_different_input():
    G = nx.DiGraph()
    G.add_nodes_from([
        (10, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(-1.0, -1.0)}),
        (20, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(0.0, -1.0)}),
        (30, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (40, {"activation": np.tanh, "type": NeuronType.HIDDEN}),
        (60, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(-1.0, 1.0)}),
        (70, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(1.0, 1.0)})
    ])

    G.add_weighted_edges_from([
        (10, 40, 0.7),
        (20, 40, 0.4),
        (40, 60, 0.1),
        (40, 70, 0.25),
    ])

    phenotype = Phenotype(G, 0)

    inputs = np.array([0.5, 0.2, 0.7])

    feedforward_highest = FeedforwardCUDA([phenotype])

    result = feedforward_highest.update(np.array([inputs]))[0]

    answers = [0.0, 0.0]
    hidden_node = math.tanh(inputs[0] * 0.7 + inputs[1] * 0.4)
    answers[0] = math.tanh(hidden_node * 0.1)
    answers[1] = math.tanh(hidden_node * 0.25)

    np.testing.assert_array_almost_equal(result, answers)