from typing import Tuple, List


if __name__ == '__main__':

    import gym
    import numpy as np
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    from prettytable import PrettyTable

    # import neat.hyperneat as hn
    from neat.neat import NEAT
    from neat.visualize import Visualize
    from neat.neatTypes import NeuronType
    from neat.evaluation import Evaluation
    from neat.phenotypes import FeedforwardCUDA, Phenotype
    from neat.populations.weightagnosticPopulation import WeightAgnosticConfiguration

    import time
    from multiprocessing import Queue

    env_name = "BipedalWalker-v2"
    nproc = 4

    pop_size = 100
    envs_size = 100
    max_stagnation = 25
    # encoding_dim = 8
    features_dimensions = 14
    behavior_steps = 50
    behavior_dimensions = features_dimensions * behavior_steps

    q = Queue()
    vis = Visualize()
    vis.start()

    def make_env(env_id, seed):
        def _f():
            env = gym.make(env_id)
            env.seed(seed)
            return env

        return _f


    def run_env_once(phenotype, env):
        feedforward_highest = FeedforwardCUDA()
        states = env.reset()

        done = False
        last_distance = 0.0
        distance_stagnation = 0

        final_reward = 0.0
        while not done:
            actions = feedforward_highest.update([phenotype], np.array([states]))

            states, reward, done, info = env.step(actions[0])
            pos = info["pos"]

            final_reward += reward
            if pos <= last_distance:
                distance_stagnation += 1
            else:
                distance_stagnation = 0

            if distance_stagnation >= 100:
                done = True
            #
            last_distance = pos

            env.render()

        print("Final rewards: {}".format(final_reward))

    def pad_matrix(all_states, matrix_width):
        padded = []
        for row in all_states:
            # row = all_states[:, i].flatten()

            row = np.pad(row, (0, abs(matrix_width - row.shape[0])), 'constant')
            padded.append(row)

        return np.array(padded)

    class TestOrganism(Evaluation):

        def __init__(self):
            print("Creating envs...")
            self.envs = SubprocVecEnv([make_env(env_name, seed) for seed in range(envs_size)])
            self.num_of_envs = envs_size
            self.feedforward = FeedforwardCUDA()
            print("Done.")

        def evaluate(self, phenotypes: List[Phenotype]) -> Tuple[np.ndarray, np.ndarray]:

            states = self.envs.reset()

            num_of_runs = 1

            fitnesses = np.zeros(len(self.envs.remotes), dtype=np.float32)
            behaviors = np.zeros((len(self.envs.remotes), 0), dtype=np.float32)

            # behaviors = []

            done = False
            done_tracker = np.zeros(len(self.envs.remotes), dtype=np.int32)

            diff = len(phenotypes) - len(self.envs.remotes)
            if diff < 0:
                done_tracker[diff:] = num_of_runs

            # distances = np.zeros(len(self.envs.remotes))
            last_distances = np.zeros(len(self.envs.remotes))
            stagnations = np.zeros(len(self.envs.remotes))

            max_steps = 10
            steps = max_steps

            while not done:

                actions = self.feedforward.update(phenotypes, states[:len(phenotypes)])
                actions = np.pad(actions, (0, abs(diff)), 'constant')

                states, rewards, dones, info = self.envs.step(actions)

                pos = np.round(np.array([i['pos'] for i in info]), 2)

                # Only keep track of rewards for the right run
                # padded_rewards = np.pad(np.asmatrix(rewards), [(0, num_of_runs - 1), (0, 0)], mode='constant')
                # rolled_rewards = np.array(
                #     [np.roll(padded_rewards[:, i], done_tracker[i]) for i in range(len(done_tracker))]).T
                # fitnesses += rolled_rewards
                fitnesses[done_tracker < num_of_runs] += rewards[done_tracker < num_of_runs]

                # Finish run if it has not moved for a certain amount of frames
                stagnated_distances = pos == last_distances
                stagnations += stagnated_distances


                stopped_moving = stagnations >= 100
                dones[stopped_moving == True] = stopped_moving[stopped_moving == True]

                # Reset stagnations
                stagnations[stopped_moving == True] = 0
                last_distances = pos

                # Finish run if the robot fell
                envs_run_done = dones == True
                done_tracker[envs_run_done] += dones[envs_run_done]
                done = all(r >= num_of_runs for r in done_tracker)

                if steps == max_steps:
                    steps = 0
                    relevant_states = states[:, :features_dimensions]
                    # behaviors += relevant_states
                    behaviors = np.append(behaviors, relevant_states, axis=1)

                steps += 1

                # Reset the done envs
                for i in np.where(dones == True)[0]:
                    remote = self.envs.remotes[i]
                    remote.send(('reset', None))
                    # If we don't receive, the remote will not reset properly
                    reset_obs = remote.recv()[0]
                    states[i] = reset_obs

                # print(done_tracker)
                # print(done)
                # self.envs.render()

            mean_fitnesses = fitnesses / num_of_runs

            # fitnesses_t = fitnesses.T
            # behaviors_t = behaviors.T
            # for i in range(fitnesses_t.shape[0]):
            #     fitness = fitnesses_t[i]
            #     mean = np.sum(fitness)/num_of_runs
            #
            #     mean_fitnesses.append(mean)

            # mean_behaviors = np.mean(behaviors, axis=1)

            padded_behaviors = np.pad(behaviors, ((0, 0), (0, abs(behavior_dimensions - behaviors.shape[1]))), 'constant')

            return (mean_fitnesses[:len(phenotypes)], padded_behaviors[:len(phenotypes)])



    env = gym.make(env_name)
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]

    print("Inputs: {} | Outputs: {}".format(inputs, outputs))

    def run():
        print("Creating neat object")
        # pop_config = WeightAgnosticConfiguration(pop_size, inputs, outputs)
        pop_config = WeightAgnosticConfiguration(pop_size, inputs, outputs, behavior_dimensions)
        neat = NEAT(TestOrganism(), pop_config)

        highest_fitness = -1000.0

        last_best = neat.population.genomes[0]

        start = time.time()
        for _ in neat.epoch():

            # if neat.epochs == 150:
            #     break

            # print("Epoch Time: {}".format(time.time() - start))
            # random_phenotype = random.choice(neat.phenotypes)
            most_fit =  max([(g.fitness, g) for g in neat.population.genomes], key=lambda e: e[0])
            # most_novel =  max([(g.novelty, g) for g in neat.population.genomes], key=lambda e: e[0])
            # best_phenotype =  max(neat.phenotypes, key=lambda e: e.fitness)

            # if max_fitness[0] >= highest_fitness:
            #     run_env_once(max_fitness[1].createPhenotype(), env)
            #     highest_fitness = max_fitness[0]

            # run_env_once(most_novel[1].createPhenotype(), env)
            # run_env_once(most_fit[1].createPhenotype(), env)
            # run_env_once(random_phenotype, env)

            if most_fit[0] > highest_fitness:
                print("New highescore: {:1.2f}".format(most_fit[0]))
                # run_env_once(most_fit[1].createPhenotype(), env)
                highest_fitness = most_fit[0]

            # print("Highest fitness all-time: {}".format(highest_fitness))

            print("Epoch {}/{}".format(neat.epochs, 150))
            print("Time: {}".format(time.time() - start))
            print("Highest fitness ({}): {:1.2f}/{:1.2f}".format(most_fit[1].ID, most_fit[0], highest_fitness))
            # run_env_once(most_fit[1].createPhenotype(), env)

            found_last = next((m for m in neat.population.genomes if m.ID == last_best.ID), None)
            if found_last is not None:
                print("Last best ({}): {:1.2f}/{:1.2f}".format(last_best.ID, found_last.fitness, highest_fitness))
                # run_env_once(last_best.createPhenotype(), env)
            else:
                print("Last best genome is no longer in population.")

            table = PrettyTable(["ID", "age", "members", "max fitness", "avg. distance", "stag", "neurons", "links", "avg. weight", "max. compat.", "to spawn"])
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
                        [len([n for n in m.createPhenotype().graph.nodes.data() if n[1]['type'] == NeuronType.HIDDEN]) for m in
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

            print(table)

            start = time.time()
            # print("########## Epoch {} ##########".format(neat.epochs))


            last_best = most_fit[1]

        return highest_fitness

    # runs = []
    run()

    # print("Final fitnesses: {}".format(runs))

    env.close()