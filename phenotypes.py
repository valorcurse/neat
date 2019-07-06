from __future__ import annotations

from typing import List, Set, Dict, Tuple, Optional, Any
from enum import Enum

import math
from math import cos, sin, atan, ceil, floor

from scipy import special

import numpy as np


image_width = 34
image_height = 31
horizontal_distance_between_layers = 10
vertical_distance_between_neurons = 5
neuron_radius = 1
number_of_neurons_in_widest_layer = 24

class NeuronType(Enum):
    INPUT = 0
    HIDDEN = 1
    BIAS = 2
    OUTPUT = 3
    LINK = 4

class SNeuron:
    def __init__(self, neuronType: NeuronType, neuronID: int, bias: float, y: float):
        self.linksIn: List[SLink] = []
        # self.linksOut = []

        self.sumActivation = 0.0
        self.bias = bias
        self.output = 0.0

        self.neuronType = neuronType

        self.ID = neuronID

        self.posX: float = 0.0
        self.posY: float = 0.0
        self.splitY = y

    def __line_between_two_neurons(self, neuron1: List[float], neuron2: List[float]) -> None:
        angle = 0.0
        if (neuron1[0] != neuron2[0] and neuron1[1] != neuron2[1]):
            angle = atan((neuron2[0] - neuron1[0]) / float(neuron2[1] - neuron1[1]))

        x_adjustment = 0
        y_adjustment = 0
        line = pyplot.Line2D(
            (neuron1[0] - x_adjustment, neuron2[0] + x_adjustment), 
            (neuron1[1] - y_adjustment, neuron2[1] + y_adjustment))
        pyplot.gca().add_line(line)

    def draw(self) -> None:
        # print("Drawing neuron", self.posX, self.posY)
        circle = pyplot.Circle(
            (self.posX, self.posY), 
            radius=neuron_radius, fill=(self.neuronType == NeuronType.BIAS))
        pyplot.gca().add_patch(circle)
        # pyplot.annotate(str(self.ID), xy=(self.posX - neuron_radius/4, self.posY))
        # pyplot.annotate(str("{:1.2f}".format(self.bias)), xy=(self.posX - neuron_radius/2, self.posY - neuron_radius/2))

        for l in self.linksIn:
            fromNeuron = l.fromNeuron
            if (l.recurrent):
                patchArrow = patches.FancyArrowPatch(
                    (self.posX + 0.3, self.posY),
                    (self.posX - 0.4, self.posY),
                    connectionstyle='arc3, rad=-4.0',    # Default
                    mutation_scale=5,
                    color="red"
                )
                pyplot.gca().add_patch(patchArrow)
            else:
                self.__line_between_two_neurons(
                    [fromNeuron.posX, fromNeuron.posY], 
                    [self.posX, self.posY])

            pyplot.annotate(str("{:1.2f}".format(l.weight)), xy=(self.posX - (self.posX - fromNeuron.posX)/2, self.posY - (self.posY - fromNeuron.posY)/2))


            pyplot.pause(0.001)

class SLink:

    def __init__(self, fromNeuron: SNeuron, toNeuron: SNeuron, weight: float, recurrent: bool = False) -> None:
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron

        self.weight = weight

        self.recurrent = recurrent

class CNeuralNet:

    def __init__(self, neurons: List[SNeuron], ID: int) -> None:
        self.neurons = neurons
        self.ID = ID

        self.toDraw = False
        self.layers: List[Layer] = []
        uniqueDepths = sorted(set([n.splitY for n in self.neurons]))
        # print("Depths:", uniqueDepths)
        for d in uniqueDepths:
            # if d == 0:
            #     continue

            # neuronsToDraw = [n for n in self.neurons 
            #     if n.neuronType in [NeuronType.HIDDEN, NeuronType.OUTPUT] and n.splitY == d]
            neuronsToDraw = [n for n in self.neurons if n.splitY == d]
            self.layers.append(Layer(self, neuronsToDraw))


    def draw(self) -> None:
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5,color='#888'),
            hoverinfo='none',
            mode='lines')

        pyplot.clf()
        for layer in self.layers:
            layer.draw()

        pyplot.xticks([]), pyplot.yticks([])  # to hide tick values on X and Y axis
        pyplot.axis('scaled')
        pyplot.gca().relim()
        pyplot.gca().autoscale_view()
        pyplot.ion()
        pyplot.show()
        pyplot.draw()
        pyplot.pause(0.001)

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def tanh(self, x: float) -> float:
        return np.tanh(x)

    def relu(self, x: float) -> float:
        return np.maximum(x, 0)

    def leakyRelu(self, x: float) -> float:
        return x if x > 0.0 else x * 0.01

    def activation(self, x: float) -> float:
        # return self.leakyRelu(x)
        return self.tanh(x)

    def calcOutput(self, neuron: SNeuron) -> float:
        linksIn = neuron.linksIn
        if (len(linksIn) > 0):
            return self.sigmoid(neuron.bias + np.sum([self.calcOutput(linkIn.fromNeuron) * linkIn.weight for linkIn in linksIn]))
        else:
            return neuron.output

    def updateRecursively(self, inputs: List[float]) -> List[float]:
        inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
        for value, neuron in zip(inputs, inputNeurons):
            neuron.output = value

        outputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.OUTPUT]

        return [self.calcOutput(outputNeuron) for outputNeuron in outputNeurons]

    def update(self, inputs: List[float]) -> List[float]:
        # Set input neurons values
        inputNeurons = [neuron for neuron in self.neurons if neuron.neuronType == NeuronType.INPUT]
        for value, neuron in zip(inputs, inputNeurons):
            neuron.output = value

        for currentNeuron in self.neurons[len(inputNeurons):]:
            linksIn = currentNeuron.linksIn

            if len(linksIn) == 0:
                currentNeuron.output = 0.0
            else:
                # output = np.sum(np.array([link.fromNeuron.output * link.weight for link in linksIn]))
                output = np.array([link.fromNeuron.output * link.weight for link in linksIn])
                # currentNeuron.output = self.activation(currentNeuron.bias + np.sum(output))
                currentNeuron.output = self.activation(currentNeuron.bias + np.sum(output))
                
        # print(table)
        return [n.output for n in self.neurons if n.neuronType == NeuronType.OUTPUT]

class Layer():
    def __init__(self, network: CNeuralNet, neuronsToDraw: List[SNeuron]) -> None:
        self.previous_layer = self.__get_previous_layer(network)
        self.x = self.__calculate_layer_x_position()
        
        self.neurons = neuronsToDraw
        self.__intialise_neurons()

    def __intialise_neurons(self) -> None:
        startY = self.__calculate_top_margin_so_layer_is_centered(len(self.neurons))
        for neuron in self.neurons:
            neuron.posX = self.x
            neuron.posY = startY
            startY += vertical_distance_between_neurons

    def __calculate_top_margin_so_layer_is_centered(self, number_of_neurons: int) -> float:
        return vertical_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_x_position(self) -> int:
        if self.previous_layer:
            return self.previous_layer.x + horizontal_distance_between_layers
        else:
            return horizontal_distance_between_layers + neuron_radius

    def __get_previous_layer(self, network: CNeuralNet) -> Optional[Layer]:
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def draw(self) -> None:
        for neuron in self.neurons:
            neuron.draw()