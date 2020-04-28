from typing import Tuple, List

if __name__ == '__main__':
    from neat.hyperneat import HyperNEAT
    from neat.phenotypes import Phenotype, CppnCUDA
    from neat.evaluation import Evaluation
    from neat.populations.multiobjectivePopulation import MOConfiguration

    import numpy as np

    class TestOrganism(Evaluation):

        def __init__(self):
            print("Creating envs...")
            self.num_of_envs = envs_size

            # X, Y = np.mgrid[0:64, 0:64]
            # xy = np.vstack((X.flatten(), Y.flatten())).T
            #
            # # self.inputs = np.array([xy] * 100)
            # # self.inputs = np.array([xy] * 100)
            # self.inputs = np.repeat(xy, 10, axis=0)
            self.feedforward = CppnCUDA()
            print("Done.")

        def evaluate(self, phenotypes: List[Phenotype]) -> Tuple[np.ndarray, np.ndarray]:

            print("Evaluating...")
            results = self.feedforward.update(phenotypes*4096)
            print("Done.")

            split_results = np.array([results[i:i + 64*64] for i in range(0, len(results), 64*64)])[:, :, 0]
            return (np.zeros(len(phenotypes)), split_results)


    envs_size = 10

    inputs = 2
    outputs = 1
    behavior_dimensions = 64 * 64
    pop_size = 100
    popconfig = MOConfiguration(pop_size, inputs, outputs, behavior_dimensions)
    hyperneat = HyperNEAT(TestOrganism(), popconfig)

    for _ in hyperneat.epoch():
        pass