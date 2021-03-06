from typing import Tuple, List


if __name__ == '__main__':

    import gym
    import numpy as np
    import pyglet
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    from prettytable import PrettyTable

    # import neat.hyperneat as hn
    import neat.visualize as dash_vis
    from neat.neat import NEAT
    from neat.visualize import Visualize
    from neat.neatTypes import NeuronType
    from neat.evaluation import Evaluation
    from neat.phenotypes import FeedforwardCUDA, Phenotype
    from neat.multiobjectivePopulation import MOConfiguration

    import time
    from multiprocessing import Queue
    import random

    env_name = "BipedalWalker-v2"
    nproc = 4

    envs_size = 100
    pop_size = 100
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
        distance = 0.0
        last_distance = 0.0
        distance_stagnation = 0

        # image = env.render(mode='rgb_array')
        # images = []
        # activations = []
        while not done:
            actions = feedforward_highest.update([phenotype], np.array([states]))

            # print(actions)
            states, reward, done, info = env.step(actions[0])
            distance += np.around(states[2], decimals=2)

            if distance <= last_distance:
                distance_stagnation += 1
            else:
                distance_stagnation = 0

            if distance_stagnation >= 100:
                done = True

            last_distance = distance

            # activations.append(feedforward_highest.mem[0])
            # images.append(env.render(mode='rgb_array'))
            env.render()

        # dash_vis.dash_queue.put([phenotype.graph, activations, images])
        # norm_acts = (np.array(activations)+1.0)/2.0
        # dash_vis.dash_queue.put([phenotype.graph, activations])

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

            states_32 = np.float32(states)
            actions = self.feedforward.update(phenotypes, states_32)

            fitnesses = np.zeros((num_of_runs, len(self.envs.remotes)), dtype=np.float64)
            all_states = np.zeros((num_of_runs, len(self.envs.remotes), behavior_dimensions))
            state_indexes = np.zeros((num_of_runs, len(self.envs.remotes)), dtype=np.int32)


            done = False
            done_tracker = np.zeros(len(self.envs.remotes), dtype=np.int32)

            diff = len(phenotypes) - len(self.envs.remotes)
            if diff < 0:
                done_tracker[diff:] = num_of_runs

            last_distances = np.zeros(len(self.envs.remotes))
            stagnations = np.zeros(len(self.envs.remotes))

            max_steps = 10
            steps = max_steps

            while not done:
                states_32 = np.float32(states)
                actions = self.feedforward.update(phenotypes, states_32[:len(phenotypes)])
                actions = np.pad(actions, (0, abs(diff)), 'constant')

                states, rewards, dones, info = self.envs.step(actions)

                pos = np.round(np.array([i['pos'] for i in info]), 2)

                # Only keep track of rewards for the right run
                padded_rewards = np.pad(np.asmatrix(rewards), [(0, num_of_runs - 1), (0, 0)], mode='constant')
                rolled_rewards = np.array(
                    [np.roll(padded_rewards[:, i], done_tracker[i]) for i in range(len(done_tracker))]).T
                fitnesses += rolled_rewards

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

                # Reset the done envs
                for i in np.where(dones == True)[0]:
                    remote = self.envs.remotes[i]
                    remote.send(('reset', None))
                    # If we don't receive, the remote will not reset properly
                    reset_obs = remote.recv()[0]
                    states[i] = reset_obs

                if steps == max_steps:
                    steps = 0
                    relevant_states = states[:, :features_dimensions]
                    for i in range(done_tracker.size):
                        for row in range(relevant_states.shape[0]):
                            level = done_tracker[i]

                            if level >= num_of_runs:
                                continue

                            s = relevant_states[row]

                            start = state_indexes[level][row]
                            end = start+len(s)

                            all_states[level][row][start:end] = s

                            # print(all_states[level][row].shape)
                            # print(all_states[level][row])

                            state_indexes[level][row] = end % behavior_dimensions

                steps += 1


            final_fitnesses = []
            final_states = []

            fitnesses_t = fitnesses.T
            for i in range(fitnesses_t.shape[0]):
                fitness = fitnesses_t[i]
                index = np.argmax(fitness)

                states = all_states[index]

                final_fitnesses.append(fitness[index])
                final_states.append(states[index])

            return (np.array(final_fitnesses[:len(phenotypes)]), np.array(final_states[:len(phenotypes)]))



    env = gym.make(env_name)
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]

    print("Inputs: {} | Outputs: {}".format(inputs, outputs))

    def run():
        print("Creating neat object")
        obj_ranges = [(-100.0, 300.0), (-1.0, 1.0)]
        pop_config = MOConfiguration(pop_size, inputs, outputs, behavior_dimensions, obj_ranges)
        neat = NEAT(TestOrganism(), pop_config)

        highest_fitness = -1000.0

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
                run_env_once(most_fit[1].createPhenotype(), env)
                highest_fitness = most_fit[0]

            # print("Highest fitness all-time: {}".format(highest_fitness))

            print("Epoch {}/{}".format(neat.epochs, 150))
            print("Time: {}".format(time.time() - start))
            print("Highest fitness: {:1.2f}/{:1.2f}".format(most_fit[0], highest_fitness))

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

            last_best = most_fit

            # print("########## Epoch {} ##########".format(neat.epochs))




        return highest_fitness

    # runs = []
    # for _ in range(1):
        # runs.append(run())
    run()

    # print("Final fitnesses: {}".format(runs))

    env.close()





