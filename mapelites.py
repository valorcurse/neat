from typing import List, Dict, Optional, Tuple, Text

from neat.population import Population, PopulationConfiguration
from neat.neat import MutationRates
from neat.genes import Genome, NeuronGene, LinkGene, Phase
from neat.innovations import Innovations

import random
import numpy as np
import pickle

class Feature:

	def __init__(self, name, minimum, maximum):
		self.name = name
		self.min = minimum
		self.max = maximum

class MapElitesConfiguration(PopulationConfiguration):
	def __init__(self, mapResolution: int, features: List[Feature]):
		self.mapResolution = mapResolution
		self.features = features


class MapElites(Population):

	# def __init__(self, populationSize: int, inputs: List[NeuronGene], outputs: List[NeuronGene], 
	# 	mutationRates: MutationRates, configuration: MapElitesConfiguration):


	def __init__(self, populationSize: int, inputs: int, outputs: int, innovations: Innovations,
		mutationRates: MutationRates, configuration: MapElitesConfiguration):
		
		super().__init__(populationSize, innovations, mutationRates)

		self.configuration = configuration
		self.innovations = innovations

		self.inputs = inputs
		self.outputs = outputs

		self.archive = np.full([configuration.mapResolution]*len(configuration.features), None, dtype=Genome)
		self.performances = np.zeros([configuration.mapResolution]*len(configuration.features))
		self.archivedGenomes: List[Genome] = []


	def randomInitialization(self) -> List[Genome]:
		randomPop: List[Genome] = []
		for _ in range(100):
			genome: Genome = self.newGenome()
			for _ in range(50):
				genome.mutate(Phase.COMPLEXIFYING, self.mutationRates)
			randomPop.append(genome)

		return randomPop

	def newGenome(self, neurons: List[NeuronGene] = [], links: List[LinkGene] = [], parents=[]):				
		genome = Genome(self.currentGenomeID, self.inputs, self.outputs, self.innovations, neurons, links, parents)
		self.currentGenomeID += 1

		return genome

	def reproduce(self):

		if len(self.archivedGenomes) == 0:
			newGenome = self.newGenome()
			self.archivedGenomes.append(newGenome)
			
			return newGenome

		member = random.choice(self.archivedGenomes)

		baby: Optional[Genome] = None
		if (random.random() > self.mutationRates.crossoverRate):
			baby = self.newGenome(member.neurons, member.links, member.parents)
			baby.mutate(Phase.COMPLEXIFYING, self.mutationRates)
		else:
			otherMember = random.choice(self.archivedGenomes)
			baby = self.crossover(member, otherMember)

		return baby

	def updateArchive(self, candidate, fitness, features) -> bool:

		featuresIndexes = []
		for idx, feature in enumerate(features):
			archiveFeature = self.configuration.features[idx]

			if feature < archiveFeature.min:
				archiveFeature.min = feature
			elif feature > archiveFeature.max:
				archiveFeature.max = feature

			# print("(%f - %f)/(%f - %f))"%(archiveFeature.max, feature, archiveFeature.max, archiveFeature.min))

			relativePosition: float = (archiveFeature.max - feature)/(archiveFeature.max - archiveFeature.min)
			index = int(relativePosition * self.configuration.mapResolution)
			index = max(0, index - 1)
			featuresIndexes.append(index)

		tupleIndex = tuple(featuresIndexes)
		archivedCandidate = self.archive[tupleIndex]
		archivedPerformance = self.performances[tupleIndex]

		if (fitness > archivedPerformance):
			if archivedCandidate is not None:
				self.archivedGenomes.remove(archivedCandidate)
			
			self.archive[tupleIndex] = candidate
			self.archivedGenomes.append(candidate)

			self.performances[tupleIndex] = fitness

			return True

		return False

		# possibleCandidates = self.archive[self.archive != None]
		# total = pow(self.configuration.mapResolution, len(self.configuration.features))
		# print("Archive: %d/%d"%(len(possibleCandidates), total))



