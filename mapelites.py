from typing import List, Optional

from neat.innovations import Innovations
from neat.genes import Genome, NeuronGene, LinkGene, MutationRates
from neat.population import Population, PopulationConfiguration, PopulationUpdate


import random
import numpy as np
from icontract import require


class Feature:

	def __init__(self, name, minimum, maximum):
		self.name = name
		self.min = minimum
		self.max = maximum

class MapElitesConfiguration(PopulationConfiguration):
	def __init__(self, mapResolution: int, features: List[Feature], n_inputs: int, n_outputs: int):
		self.data = {
			"mapResolution": mapResolution,
			"features": features,
			"n_inputs": n_inputs,
			"n_outputs": n_outputs
		}

		

class MapElitesUpdate(PopulationUpdate):
	def __init__(self, fitness: List[float], features: List[List[float]]):
		self._data = {
			"fitness": fitness,
			"features": features
		}


class MapElites(Population):

	@require(lambda configuration: isinstance(configuration, MapElitesConfiguration))
	def __init__(self, 
			configuration: PopulationConfiguration, innovations: Innovations, mutationRates: MutationRates
		):
		
		super().__init__(configuration, innovations, mutationRates)

		self.configuration = configuration
		self.innovations = innovations

		self.inputs = self.configuration.inputs
		self.outputs = self.configuration.outputs

		self.archive = np.full([configuration.mapResolution*len(configuration.features)], None, dtype=Genome)
		self.performances = np.zeros([configuration.mapResolution]*len(configuration.features))
		self.archivedGenomes: List[Genome] = []


	def randomInitialization(self) -> List[Genome]:
		randomPop: List[Genome] = []
		for _ in range(100):
			genome: Genome = self.newGenome()
			for _ in range(50):
				genome.mutate(self.mutationRates)
			randomPop.append(genome)

		return randomPop

	def newGenome(self, neurons: List[NeuronGene] = [], links: List[LinkGene] = [], parents=[]):				
		genome = Genome(self.currentGenomeID, self.inputs, self.outputs, self.innovations, neurons, links, parents)
		self.currentGenomeID += 1

		return genome

	def reproduce(self) -> List[Genome]:

		if len(self.archivedGenomes) == 0:
			newGenome = self.newGenome()
			self.archivedGenomes.append(newGenome)
			
			return newGenome

		member = random.choice(self.archivedGenomes)

		baby: Optional[Genome] = None
		if (random.random() > self.mutationRates.crossoverRate):
			baby = self.newGenome(member.neurons, member.links, member.parents)
			baby.mutate(self.mutationRates)

		else:
			otherMember = random.choice(self.archivedGenomes)
			baby = self.crossover(member, otherMember)

		self.genomes = [baby]

		return self.genomes

	@require(lambda data: isinstance(data, MapElitesUpdate))
	def updatePopulation(self, update: PopulationUpdate) -> None:
		self.updateArchive(update.fitness, update.features)
		

	def updateArchive(self, fitness: List[float], features) -> None:

		for candidate in self.genomes:
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

		# possibleCandidates = self.archive[self.archive != None]
		# total = pow(self.configuration.mapResolution, len(self.configuration.features))
		# print("Archive: %d/%d"%(len(possibleCandidates), total))



