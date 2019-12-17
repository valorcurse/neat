from typing import List, Dict, Optional

import random

from neat.phenotypes import Phenotype
from neat.innovations import Innovations
from neat.genes import Genome, LinkGene, NeuronGene, MutationRates
import neat.phenotypes as phenos


class ParameterDict:
    def __init__(self):
        self._data = {}

    def __getattr__(self, attr):
        return self._data[attr]

    # def __setattr__(self, name, value):
    #     self._data.__dict__[name] = value

class PopulationConfiguration(ParameterDict):
    pass

class PopulationUpdate(ParameterDict):
    pass

class Population:

    def __init__(self, configuration: PopulationConfiguration, innovations: Innovations, mutationRates: MutationRates):
        self.genomes: List[Genome] = []

        self.innovations = innovations
        self.mutationRates = mutationRates

        self.currentGenomeID: int = 0

        self.numOfInputs = configuration.n_inputs
        self.numOfOutputs = configuration.n_outputs

    def population(self):
        pass

    def newGenome(self, neurons: List[NeuronGene], links: List[LinkGene], parents=[]):


        genome = Genome(self.currentGenomeID, self.numOfInputs, self.numOfOutputs, self.innovations, neurons, links, parents)
        genome.parents = parents if len(parents) > 0 else [genome, genome]
        self.genomes.append(genome)
        self.currentGenomeID += 1

        return genome

    def updatePopulation(self, update_data: PopulationUpdate) -> None:
        pass

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
        babyNeurons = [n for n in best.neurons if (n.neuronType != phenos.NeuronType.HIDDEN)]

        combinedIndexes = list(set(
            [l.ID for l in mum.links] + [l.ID for l in dad.links]))
        combinedIndexes.sort()
        
        mumDict: Dict[int, LinkGene] = {l.ID: l for l in mum.links}
        dadDict: Dict[int, LinkGene] = {l.ID: l for l in dad.links}

        # Crossover the links
        babyLinks: List[LinkGene] = []
        for i in combinedIndexes:
            mumLink: Optional[LinkGene] = mumDict.get(i)
            dadLink: Optional[LinkGene] = dadDict.get(i)
            
            if (mumLink is None):
                if (dadLink is not None and best == dad):
                    babyLinks.append(dadLink)

            elif (dadLink is None):
                if (mumLink is not None and best == mum):
                    babyLinks.append(mumLink)

            else:
                babyLinks.append(random.choice([mumLink, dadLink]))

        # Copy the neurons connected to the crossedover links
        for link in babyLinks:

            if (link.fromNeuron.ID not in [n.ID for n in babyNeurons]):
                babyNeurons.append(link.fromNeuron)

            if (link.toNeuron.ID not in [n.ID for n in babyNeurons]):
                babyNeurons.append(link.toNeuron)

        babyNeurons.sort(key=lambda x: x.y, reverse=False)

        return self.newGenome(babyNeurons, babyLinks, [mum, dad])


    def create_phenotypes(self) -> List[Phenotype]:
        return [g.createPhenotype() for  g in self.genomes]

    def refine_behaviors(self, eval_env):
        pass

    def reproduce(self) -> List:
        pass

