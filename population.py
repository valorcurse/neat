from typing import List, Dict, Optional, Tuple

import random
import pickle

from neat.neat import MutationRates
from neat.genes import Genome, NeuronType, LinkGene, NeuronGene

class Species:
    numGensAllowNoImprovement = 20

    def __init__(self, speciesID: int, leader: Genome):
        self.ID: int = speciesID

        self.members: List[Genome] = []
        self.addMember(leader)
        self.leader: Genome = leader

        self.age: int = 0
        self.numToSpawn: int = 0


        self.youngAgeThreshold: int = 10
        self.youngAgeBonus: float = 1.5
        self.oldAgeThreshold: int = 50
        self.oldAgePenalty: float = 0.5

        self.highestFitness: float = 0.0
        self.generationsWithoutImprovement: int = 0

        self.milestone: float = leader.milestone

        self.stagnant: bool = False

    def __contains__(self, key: int) -> bool:
        return key in [m.ID for m in self.members]

    def isMember(self, genome: Genome) -> bool:
        return (genome.ID in [m.ID for m in self.members])

    def addMember(self, genome: Genome) -> None:
        self.members.append(genome)
        genome.species = self

    def best(self) -> Genome:
        return max(self.members)


    def spawn(self) -> Genome:
        return random.choice(self.members)

    def adjustFitnesses(self) -> None:
        # avgMilestone = np.average([m.milestone for m in self.members])
        
        # self.members = [m for m in self.members if m.milestone >= avgMilestone]

        for m in self.members:
            m.adjustedFitness = m.fitness / len(self.members)

            # if self.age <= self.youngAgeThreshold:
            #     m.adjustedFitness *= self.youngAgeBonus

            # if self.age >= self.oldAgeThreshold:
            #     m.adjustedFitness *= self.oldAgePenalty

    def becomeOlder(self, alone: bool) -> None:
        self.age += 1

        highestFitness = max([m.fitness for m in self.members])

        if alone:
            return

        # Check if species is stagnant
        if (highestFitness < self.highestFitness):
            self.generationsWithoutImprovement += 1
        else:
            self.generationsWithoutImprovement = 0
            self.highestFitness = highestFitness

        if (self.generationsWithoutImprovement >= self.numGensAllowNoImprovement):
            self.stagnant = True

class PopulationConfiguration:
	
	def __init__(self):
		pass

class Population:

    def __init__(self, populationSize: int, mutationRates: MutationRates):
        self.genomes: List[Genome] = []
        
        self.species: List[Species] = []
        self.speciesNumber: int = 0

        self.mutationRates = mutationRates

        self.currentGenomeID: int = 0
        self.populationSize: int = populationSize

        self.genomes: List[Genome] = []

        self.averageInterspeciesDistance: float = 0.0
        self.numOfInputs = 0
        self.numOfOutputs = 0

    def initiate(self, neurons: List[NeuronGene], links: List[LinkGene], 
    	numOfInputs: int, numOfOutputs: int, parents=[]):
    	
    	self.numOfInputs = numOfInputs
    	self.numOfOutputs = numOfOutputs

    	for i in range(self.populationSize):
            genome = self.newGenome(neurons, links)
            genome.parents = [genome]

    def newGenome(self, neurons: List[NeuronGene], links: List[LinkGene], parents=[]):
    	
    	genome = Genome(self.currentGenomeID, neurons, links, self.numOfInputs, self.numOfOutputs, parents)
    	self.genomes.append(genome)
    	self.currentGenomeID += 1

    	return genome

    def speciate(self) -> None:

        # Find best leader for species from the new population
        unspeciated = list(range(0, len(self.genomes)))
        for s in self.species:
            compareMember = s.leader

            s.members = []

            candidates: List[Tuple[float, int]] = []
            for i in unspeciated:
                genome = self.genomes[i]

                distance = genome.calculateCompatibilityDistance(compareMember)

                if (distance < max(self.mutationRates.newSpeciesTolerance, self.averageInterspeciesDistance)):
                    candidates.append((distance, i))

            if len(candidates) == 0:
                self.species.remove(s)
                continue

            _, bestCandidate = min(candidates, key=lambda x: x[0])

            s.leader = self.genomes[bestCandidate]
            s.members.append(s.leader)
            unspeciated.remove(bestCandidate)

        # Distribute genomes to their closest species
        for i in unspeciated:
            genome = self.genomes[i]

            # closestDistance = self.mutationRates.newSpeciesTolerance
            closestDistance = max(self.mutationRates.newSpeciesTolerance, self.averageInterspeciesDistance)
            closestSpecies = None
            for s in self.species:
                distance = genome.calculateCompatibilityDistance(s.leader)
                # If genome falls within tolerance of species
                if (distance < closestDistance):
                    closestDistance = distance
                    closestSpecies = s

            if (closestSpecies is not None): # If found a compatible species
                # closestSpecies.members.append(genome)
                closestSpecies.addMember(genome)

            else: # Else create a new species
                chance: float = random.random()

                parentSpecies: Optional[Species] = random.choice(genome.parents).species

                # if (chance >= 0.1) and parentSpecies is not None:
                #     parentSpecies.addMember(genome)
                # else:
                self.speciesNumber += 1
                self.species.append(Species(self.speciesNumber, genome))

        # Calculate average interspecies distance
        if len(self.species) > 1:
            totalDistance: float = 0.0
            for s in self.species:
                randomSpecies: Species = random.choice([r for r in self.species if r is not s])

                totalDistance += s.leader.calculateCompatibilityDistance(randomSpecies.leader)
            
            self.averageInterspeciesDistance = max(self.mutationRates.newSpeciesTolerance, (totalDistance/len(self.species)))

            print("averageInterspeciesDistance: " + str(self.averageInterspeciesDistance))

    def crossover(self, mum: Genome, dad: Genome) -> Genome:
        
        best = None

        # If both parents perform equally, choose the simpler one
        if (mum.fitness == dad.fitness):
            if (len(mum.links) == len(dad.links)):
                best = random.choice([mum, dad])
            else:
                best = mum if len(mum.links) < len(dad.links) else dad
        else:
            best = mum if mum.fitness > dad.fitness else dad

        # Copy input and output neurons
        babyNeurons = [pickle.loads(pickle.dumps(n, -1)) for n in best.neurons
                       if (n.neuronType in [NeuronType.INPUT, NeuronType.OUTPUT])]

        combinedIndexes = list(set(
            [l.innovationID for l in mum.links] + [l.innovationID for l in dad.links]))
        combinedIndexes.sort()
        
        mumDict: Dict[int, LinkGene] = {l.innovationID: l for l in mum.links}
        dadDict: Dict[int, LinkGene] = {l.innovationID: l for l in dad.links}

        # print("-------------------------------------------------")
        babyLinks: List[LinkGene] = []
        for i in combinedIndexes:
            mumLink: Optional[LinkGene] = mumDict.get(i)
            dadLink: Optional[LinkGene] = dadDict.get(i)
            
            if (mumLink is None):
                if (dadLink is not None and best == dad):
                    babyLinks.append(pickle.loads(pickle.dumps(dadLink, -1)))

            elif (dadLink is None):
                if (mumLink is not None and best == mum):
                    babyLinks.append(pickle.loads(pickle.dumps(mumLink, -1)))

            else:
                babyLinks.append(random.choice([mumLink, dadLink]))

        for link in babyLinks:

            if (link.fromNeuron.innovationID not in [n.innovationID for n in babyNeurons]):
                babyNeurons.append(pickle.loads(pickle.dumps(link.fromNeuron, -1)))

            if (link.toNeuron.innovationID not in [n.innovationID for n in babyNeurons]):
                babyNeurons.append(pickle.loads(pickle.dumps(link.toNeuron, -1)))

        babyNeurons.sort(key=lambda x: x.y, reverse=False)

        return self.newGenome(babyNeurons, babyLinks, [mum, dad])


    def reproduce(self) -> List:
        pass

