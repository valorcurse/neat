from typing import List, Optional

from neat.innovations import Innovations
from neat.genes import Genome, NeuronGene, LinkGene, MutationRates
from neat.populations.population import Population, PopulationConfiguration, PopulationUpdate


import random
import numpy as np
from icontract import require


class Feature:

	def __init__(self, name, minimum, maximum):
		self.name = name
		self.min = minimum
		self.max = maximum

class MapElitesConfiguration(PopulationConfiguration):
	def __init__(self, mapResolution: int, pop_size: int, features: int, n_inputs: int, n_outputs: int):
		self._data = {
			"mapResolution": mapResolution,
			"pop_size": pop_size,
			"features": features,
			"n_inputs": n_inputs,
			"n_outputs": n_outputs
		}

		

class MapElitesUpdate(PopulationUpdate):
	def __init__(self, fitness, features):
		self._data = {
			"fitness": fitness,
			"features": features
		}


class MapElites(Population):

	@require(lambda configuration: isinstance(configuration, MapElitesConfiguration))
	def __init__(self, configuration: PopulationConfiguration, innovations: Innovations, mutationRates: MutationRates):
		
		super().__init__(configuration, innovations, mutationRates)

		self.configuration = configuration
		self.innovations = innovations

		self.inputs = self.configuration.n_inputs
		self.outputs = self.configuration.n_outputs

		archive_dim = tuple([configuration.mapResolution]*configuration.features)
		print("Map elites archive size: {}".format(pow(configuration.mapResolution, configuration.features)))
		# self.archive = np.full(archive_dim, None, dtype=Genome)
		self.archive = {}
		self.archivedFeatures = {}
		# self.performances = np.zeros(archive_dim)
		self.performances = {}
		self.archivedGenomes: List[Genome] = []
		# self.archivedGenomes: List[Genome] = self.randomInitialization()

	def __len__(self):
		return len(self.archivedGenomes)


	def randomInitialization(self) -> List[Genome]:
		randomPop: List[Genome] = []
		for _ in range(100):
			genome: Genome = self.newGenome()
			for _ in range(100):
				genome.mutate(self.mutationRates)
			randomPop.append(genome)

		return randomPop

	def population_and_fitnesses(self):
		return self.archive.values()

	def population(self):
		return [g['genome'] for g in self.archive.values()]

	def newGenome(self, neurons: List[NeuronGene] = [], links: List[LinkGene] = [], parents=[]):				
		genome = Genome(self.currentGenomeID, self.inputs, self.outputs, self.innovations,
						neurons, links, parents)

		self.currentGenomeID += 1

		return genome

	def reproduce(self) -> List[Genome]:

		if len(self.archivedGenomes) == 0:
			newGenome = self.newGenome()
			self.archivedGenomes.append(newGenome)

			self.genomes.append(newGenome)

			# return self.genomes


		babies = []
		for _ in range(self.configuration.pop_size):
			member = random.choice(self.archivedGenomes)

			baby: Optional[Genome] = None
			if random.random() > self.mutationRates.crossoverRate:
				baby = self.newGenome(member.neurons, member.links, member.parents)
				baby.mutate(self.mutationRates)

			else:
				otherMember = random.choice(self.archivedGenomes)

				baby = self.crossover(member, otherMember)

			babies.append(baby)

		self.genomes = babies

		return self.genomes

	@require(lambda update: isinstance(update, MapElitesUpdate))
	def updatePopulation(self, update: PopulationUpdate) -> None:
		self.updateArchive(update.fitness, update.features)
		

	def updateArchive(self, fitnesses: List[float], features) -> None:

		for candidate_i, candidate in enumerate(self.genomes):

			feature = features[candidate_i]
			fitness = fitnesses[candidate_i]


			relativePosition: float = (1.0 + feature) / 2.0
			index = relativePosition * self.configuration.mapResolution
			index = np.clip(0, index - 1, self.configuration.mapResolution).astype(np.int)
			tupleIndex = tuple(index)

			if tupleIndex in self.archive:

				archivedCandidate = self.archive[tupleIndex]

				if (fitness > archivedCandidate["fitness"]):
					if archivedCandidate["genome"] in self.archivedGenomes:
						self.archivedGenomes.remove(archivedCandidate["genome"])

					self.archive[tupleIndex] = {"genome": candidate, "fitness": fitness}
					self.archivedGenomes.append(candidate)
					# self.archivedGenomes = sorted(self.archivedGenomes, key=lambda a: a['fitness'])

			else:
				self.archive[tupleIndex] = {"genome": candidate, "fitness": fitness}
