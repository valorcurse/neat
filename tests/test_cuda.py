from neat.phenotypes import Phenotype
from neat.neatTypes import NeuronType
from neat.phenotypes import SequentialCUDA
import numpy as np
import math
import networkx as nx

def test_single_edges():
    G = nx.DiGraph()
    G.add_nodes_from([
        (10, {"activation": np.tanh, "type": NeuronType.INPUT, "bias":0.0, "pos":(-1.0, -1.0)}),
        (20, {"activation": np.tanh, "type": NeuronType.INPUT, "bias":0.0, "pos":(0.0, -1.0)}),
        (30, {"activation": np.tanh, "type": NeuronType.INPUT, "bias":0.0, "pos":(1.0, -1.0)}),
        (60, {"activation": np.tanh, "type": NeuronType.OUTPUT, "bias":0.0, "pos":(-1.0, 1.0)}),
        (70, {"activation": np.tanh, "type": NeuronType.OUTPUT, "bias":0.0, "pos":(1.0, 1.0)})
    ])

    G.add_weighted_edges_from([
        (20, 60, 0.4),
        (30, 70, 0.1),
    ])

    phenotype = Phenotype(G, 0)

    inputs = np.array([0.5, 0.5, 0.5])

    feedforward_highest = SequentialCUDA()

    result = feedforward_highest.update([phenotype], [inputs])[0]

    norm_inputs = np.tanh(inputs)
    answers = np.tanh([norm_inputs[0]*0.4, norm_inputs[1]*0.1])

    np.testing.assert_array_almost_equal(result, answers)


def test_multiple_edges():
    G = nx.DiGraph()
    G.add_nodes_from([
        (1, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(-1.0, -1.0)}),
        (2, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(0.0, -1.0)}),
        (3, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (6, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(-1.0, 1.0)}),
        (7, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(1.0, 1.0)})
    ])

    G.add_weighted_edges_from([
        (2, 6, 0.4),
        (3, 6, 0.1)
    ])

    phenotype = Phenotype(G, 0)

    inputs = np.array([1, 1, 1])

    feedforward_highest = SequentialCUDA()

    result = feedforward_highest.update([phenotype], [inputs])[0]

    norm_inputs = np.tanh(inputs)
    answer = norm_inputs[1]*0.4 + norm_inputs[2]*0.1
    answers = np.tanh([answer, 0.0])

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

    feedforward_highest = SequentialCUDA()

    result = feedforward_highest.update([phenotype], [inputs])[0]

    norm_inputs = np.tanh(inputs)
    hidden = math.tanh(norm_inputs[1]*0.4)
    answers = np.tanh([hidden*0.1, 0.0])

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

    feedforward_highest = SequentialCUDA()

    result = feedforward_highest.update([phenotype], [inputs])[0]

    answers = [0.0, 0.0]
    hidden_node = math.tanh(math.tanh(inputs[0]) * 0.7 + math.tanh(inputs[1]) * 0.4)
    answers[0] = math.tanh(hidden_node * 0.1)
    answers[1] = math.tanh(hidden_node * 0.25)

    np.testing.assert_array_almost_equal(result, answers)

def test_multiple_phenotypes():
    G1 = nx.DiGraph()
    G1.add_nodes_from([
        (0, {"activation": np.tanh, "type": NeuronType.INPUT, "bias": 0.0, "pos": (-1.0, -1.0)}),
        (-1, {"activation": np.tanh, "type": NeuronType.INPUT, "bias": 0.0, "pos": (0.0, -1.0)}),

        (6, {"activation": np.tanh, "type": NeuronType.HIDDEN, "bias": 0.0, "pos": (-1.0, 0.0)}),
        (10, {"activation": np.tanh, "type": NeuronType.HIDDEN, "bias": 0.0, "pos": (0.0, 0.0)}),
        (3, {"activation": np.tanh, "type": NeuronType.HIDDEN, "bias": 0.0, "pos": (1.0, 0.0)}),

        (-2, {"activation": np.tanh, "type": NeuronType.OUTPUT, "bias": 0.0, "pos": (-1.0, 1.0)}),
    ])

    G1.add_weighted_edges_from([
        (0, -2, 0.15286614111378544),
        (-1, -2, 0.8345164457693686),
        (0, 3, 0.15286614111378544),
        (3, -2, 1.0),
        (0, 6, 0.15286614111378544),
        (6, 3, 1.0),
        (6, 10, 1.0),
        (10, 3, 1.0)
    ])

    phenotype1 = Phenotype(G1, 0)

    G2 = nx.DiGraph()
    G2.add_nodes_from([
        (0, {"activation": np.tanh, "type": NeuronType.INPUT, "bias": 0.0, "pos": (-1.0, -1.0)}),
        (-1, {"activation": np.tanh, "type": NeuronType.INPUT, "bias": 0.0, "pos": (0.0, -1.0)}),

        (6, {"activation": np.tanh, "type": NeuronType.HIDDEN, "bias": 0.0, "pos": (-1.0, 0.0)}),
        (3, {"activation": np.tanh, "type": NeuronType.HIDDEN, "bias": 0.0, "pos": (1.0, 0.0)}),

        (-2, {"activation": np.tanh, "type": NeuronType.OUTPUT, "bias": 0.0, "pos": (-1.0, 1.0)}),
    ])

    G2.add_weighted_edges_from([
        (0, -2, 0.15286614111378544),
        (-1, -2, 0.8345164457693686),
        (0, 3, 0.15286614111378544),
        (3, -2, 1.0),
        (0, 6, 0.15286614111378544),
        (6, 3, 1.0)
    ])

    phenotype2 = Phenotype(G2, 0)

    inputs = np.array([1, 1])

    feedforward_highest = SequentialCUDA()

    result_one = feedforward_highest.update([phenotype1], [inputs])

    result_both = feedforward_highest.update([phenotype1, phenotype2], [inputs, inputs])

    assert result_one[0] == result_both[0]

    # norm_inputs = np.tanh(inputs)
    # hidden = math.tanh(norm_inputs[1]*0.4)
    # answers = np.tanh([hidden*0.1, 0.0])
    #
    # np.testing.assert_array_almost_equal(result, answers, decimal=4)

def test_custom():
    G = nx.DiGraph()
    G.add_nodes_from([
        (1, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(-1.0, -1.0)}),
        (2, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (3, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (4, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (5, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (6, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (7, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (8, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (9, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (10, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}), #
        (11, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}), #
        (12, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (13, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (14, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (15, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (16, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (17, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (18, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (19, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (20, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (21, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (22, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (23, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),
        (24, {"activation": np.tanh, "type": NeuronType.INPUT, "pos":(1.0, -1.0)}),

        (25, {"activation": np.tanh, "type": NeuronType.HIDDEN, "pos":(1.0, -1.0)}),

        (26, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(-1.0, 1.0)}),
        (27, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(-1.0, 1.0)}),
        (28, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(-1.0, 1.0)}),
        (29, {"activation": np.tanh, "type": NeuronType.OUTPUT, "pos":(-1.0, 1.0)}),
    ])

    G.add_weighted_edges_from([
        (10, 25, 1.0),
        (11, 25, 1.0),
        (25, 27, 1.0),
    ])

    phenotype = Phenotype(G, 0)

    inputs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.89, 0.996, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    feedforward_highest = SequentialCUDA()

    result = feedforward_highest.update([phenotype], [inputs])[0]
    print(result)
    print(feedforward_highest.mem[0])
    # norm_inputs = np.tanh(inputs)
    # answers = np.tanh([norm_inputs[0]*0.4, norm_inputs[1]*0.1])
    #
    # np.testing.assert_array_almost_equal(result, answers)