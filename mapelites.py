from typing import List, Dict, Optional, Tuple, Text

from neat.population import Population, PopulationConfiguration
from neat.neat import MutationRates
from neat.genes import Genome, NeuronGene, LinkGene, Phase

import random
import numpy as np
from copy import deepcopy

class Feature:

	def __init__(self, name, minimum, maximum):
		self.name = name
		self.min = minimum
		self.max = maximum

class MapElitesConfiguration(PopulationConfiguration):
	def __init__(self, mapResolution: int, features: List[Text]):
		self.mapResolution = mapResolution
		self.features = features


class MapElites(Population):

	def __init__(self, populationSize: int, inputs: List[NeuronGene], outputs: List[NeuronGene], 
		mutationRates: MutationRates, configuration: MapElitesConfiguration):
		
		super().__init__(populationSize, mutationRates)

		self.configuration = configuration

		self.inputs = inputs
		self.outputs = outputs

		self.archive = np.full([configuration.mapResolution]*len(configuration.features), None, dtype=Genome)
		self.performances = np.zeros([configuration.mapResolution]*len(configuration.features))

	def initiate(self, neurons: List[NeuronGene], links: List[LinkGene], 
		numOfInputs: int, numOfOutputs: int, parents=[]):
		
		# for i in range(self.populationSize):
		genome = self.newGenome(neurons, links)
		genome.parents = [genome]


	def newGenome(self, neurons: List[NeuronGene], links: List[LinkGene], parents=[]):
		
		genome = Genome(self.currentGenomeID, neurons, links, len(self.inputs), len(self.outputs), parents)
		# self.genomes.append(genome)
		self.currentGenomeID += 1

		return genome

	def reproduce(self):
		possibleCandidates = self.archive[self.archive != None]

		if len(possibleCandidates) == 0:
			neurons = self.inputs
			neurons.extend(self.outputs)

			return self.newGenome(neurons, [])

		member = np.random.choice(possibleCandidates)
		# while (member is None):
			# member = np.random.choice(self.archive)

		baby: Optional[Genome] = None
		if (random.random() > self.mutationRates.crossoverRate):
			baby = deepcopy(member)
			baby.mutate(Phase.COMPLEXIFYING, self.mutationRates)

		else:
			otherMember = np.random.choice(possibleCandidates)

			baby = self.crossover(member, otherMember)


		return baby

	def updateArchive(self, candidate, fitness, features):

		featuresIndexes = []
		for idx, feature in enumerate(features):
			archiveFeature = self.configuration.features[idx]

			if feature < archiveFeature.min:
				archiveFeature.min = feature
			elif feature > archiveFeature.max:
				archiveFeature.max = feature

			index: int = int((archiveFeature.max - feature)/(archiveFeature.max - archiveFeature.min))
			index = int(index * self.configuration.mapResolution)
			index = max(0, index - 1)
			featuresIndexes.append(index)

		tupleIndex = tuple(featuresIndexes)
		archivedCandidate = self.archive[tupleIndex]
		archivedPerformance = self.performances[tupleIndex]

		if (fitness > archivedPerformance):
			self.archive[tupleIndex] = candidate
			self.performances[tupleIndex] = fitness


		# possibleCandidates = self.archive[self.archive != None]
		# total = pow(self.configuration.mapResolution, len(self.configuration.features))
		# print("Archive: %d/%d"%(len(possibleCandidates), total))



