from typing import List, Optional

from neat.population import Population
from neat.genes import Genome
# from neat.neat import Species

import math
import random
from copy import deepcopy

from neat.genes import Phase

class DefaultPopulation(Population):

    def reproduce(self):

        newPop = []
        for s in self.species:
            # numToSpawn = s.numToSpawn

            # members = deepcopy(s.members)
            s.members.sort(reverse=True, key=lambda x: x.fitness)

            topPercent = int(math.ceil(0.01 * len(s.members)))
            print("topPercent: " + str(topPercent))
            # Grabbing the top 2 performing genomes
            for topMember in s.members[:topPercent]:
                newPop.append(topMember)
                # s.members.remove(topMember)
                s.numToSpawn -= 1

            # Only select members who got past the milestone
            # s.members = [m for m in s.members if m.milestone >= s.milestone]

            # s.members.sort(reverse=True, key=lambda x: x.milestone)
            # Only use the survival threshold fraction to use as parents for the next generation.
            cutoff = int(math.ceil(0.1 * len(s.members)))
            # Use at least two parents no matter what the threshold fraction result is.
            cutoff = max(cutoff, 2)
            s.members = s.members[:cutoff]

            # if (s.numToSpawn <= 0 or len(s.members) <= 0):
            #     continue

            for i in range(s.numToSpawn):
                baby: Optional[Genome] = None

                # if (self.phase == Phase.PRUNING or random.random() > self.mutationRates.crossoverRate):
                if (random.random() > self.mutationRates.crossoverRate):
                    baby = deepcopy(random.choice(s.members))
                    baby.mutate(Phase.COMPLEXIFYING, self.mutationRates)
                else:
                    # g1 = random.choice(members)
                    # g2 = random.choice(members)
                    
                    # Tournament selection
                    randomMembers = [random.choice(s.members) for _ in range(5)]
                    g1 = sorted(randomMembers, key=lambda x: x.fitness)[0]
                    g2 = sorted(randomMembers, key=lambda x: x.fitness)[0]
                    
                    baby = self.crossover(g1, g2)

                self.currentGenomeID += 1
                baby.ID = self.currentGenomeID

                # baby.mutate(Phase.COMPLEXIFYING, self.mutationRates)
                # baby.mutate(self.phase, self.mutationRates)

                newPop.append(baby)

        self.genomes = newPop