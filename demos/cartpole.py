from typing import Tuple, List

if __name__ == '__main__':
    import gym
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    import numpy as np
    from prettytable import PrettyTable


    from neat.neat import NEAT
    from neat.neatTypes import NeuronType
    from neat.evaluation import Evaluation
    from neat.phenotypes import Phenotype, SequentialCUDA
    from neat.populations.speciatedPopulation import SpeciesConfiguration


    env_name = "CartPole-v1"
    def make_env(env_id, seed):
        def _f():
            env = gym.make(env_id)
            env.seed(seed)
            return env

        return _f

    def run_env_once(phenotype, env):
        feedforward_highest = SequentialCUDA()
        states = env.reset()

        done = False

        final_reward = 0.0
        while not done:
            actions = feedforward_highest.update([phenotype], np.array([states]))

            states, reward, done, info = env.step(np.argmax(actions[0]))

            final_reward += reward

            env.render()

        print("Final rewards: {}".format(final_reward))


    class TestOrganism(Evaluation):

        def __init__(self):
            print("Creating envs...")
            self.envs = SubprocVecEnv([make_env(env_name, seed) for seed in range(envs_size)])
            self.num_of_envs = envs_size
            self.feedforward = SequentialCUDA()

            print("Done.")

        def evaluate(self, phenotypes: List[Phenotype]) -> Tuple[np.ndarray, np.ndarray]:

            states = self.envs.reset()

            num_of_runs = 3

            fitnesses = np.zeros(len(self.envs.remotes), dtype=np.float64)


            done = False
            done_tracker = np.zeros(len(self.envs.remotes), dtype=np.int32)

            diff = abs(len(phenotypes) - len(self.envs.remotes))
            if diff < 0:
                done_tracker[diff:] = num_of_runs

            while not done:

                actions = self.feedforward.update(phenotypes, states[:len(phenotypes)])
                actions = np.pad(actions, ((0, diff), (0, 0)), 'constant')

                states, rewards, dones, info = self.envs.step(np.argmax(actions, axis=1))

                fitnesses[done_tracker < num_of_runs] += rewards[done_tracker < num_of_runs]

                # Finish run if the robot fell
                envs_run_done = dones == True
                done_tracker[envs_run_done] += dones[envs_run_done]
                done = all(r >= num_of_runs for r in done_tracker)

                # Reset the done envs
                for i in np.where(dones == True)[0]:
                    remote = self.envs.remotes[i]
                    remote.send(('reset', None))
                    # If we don't receive, the remote will not reset properly
                    reset_obs = remote.recv()[0]
                    states[i] = reset_obs

                # self.envs.render()

            final_fitnesses = []
            fitnesses_t = fitnesses.T
            for i in range(fitnesses_t.shape[0]):
                fitness = fitnesses_t[i]
                mean = np.sum(fitness)/num_of_runs

                final_fitnesses.append(mean)

            return (np.array(final_fitnesses[:len(phenotypes)]), np.zeros((len(phenotypes), 0)))

    env = gym.make(env_name)
    pop_size = 100
    envs_size = 100
    inputs = 4
    outputs = 2
    pop_config = SpeciesConfiguration(pop_size, inputs, outputs)
    neat = NEAT(TestOrganism(), pop_config)

    highest_fitness = -1000.0
    for _ in neat.epoch():
        print("Epoch {}/{}".format(neat.epochs, 150))

        most_fit = max([{"fitness": g.fitness, "genome": g} for g in neat.population.genomes], key=lambda e: e["fitness"])

        if most_fit["fitness"] > highest_fitness:
            print("New highescore: {:1.2f}".format(most_fit["fitness"]))
            run_env_once(most_fit["genome"].createPhenotype(), env)
            highest_fitness = most_fit["fitness"]

        for s in neat.population.species:
            i = np.argmax([m.fitness for m in s.members])
            print("{}: {}".format(s.ID, s.members[i]))

        table = PrettyTable(
            ["ID", "age", "members", "max fitness", "avg. distance", "stag", "neurons", "links", "avg. weight",
             "max. compat.", "to spawn"])
        for s in neat.population.species:
            table.add_row([
                # Species ID
                s.ID,
                # Age
                s.age,
                # Nr. of members
                len(s.members),
                # Max fitness
                "{:1.4f}".format(max([m.fitness for m in s.members])),
                # Average distance
                "{:1.4f}".format(max([m.distance for m in s.members])),
                # Stagnation
                s.generationsWithoutImprovement,
                # Neurons
                # "{:1.2f}".format(np.mean([len([n for n in p.graph.nodes.data() if n[1]['type'] == NeuronType.HIDDEN]) for p in neat.phenotypes])),
                "{:1.2f}".format(np.mean(
                    [len([n for n in m.createPhenotype().graph.nodes.data() if n[1]['type'] == NeuronType.HIDDEN]) for m
                     in
                     s.members])),
                # Links
                # "{:1.2f}".format(np.mean([len(p.graph.edges) for p in neat.phenotypes])),
                "{:1.2f}".format(np.mean([len(m.createPhenotype().graph.edges) for m in s.members])),
                # Avg. weight
                "{:1.2f}".format(np.mean([l.weight for m in s.members for l in m.links])),
                # Max. compatiblity
                "{:1.2}".format(np.max([m.calculateCompatibilityDistance(s.leader) for m in s.members])),
                # Nr. of members to spawn
                s.numToSpawn])

        if most_fit["fitness"] == 500:
            print("Done!")
            print(most_fit["genome"])
            # best_phenotype = most_fit["genome"].createPhenotype()
            # pos = [n[1]['pos'] for n in best_phenotype.graph.nodes.data()]

            break

        print(table)
