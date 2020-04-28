from typing import List

import sys

import gym
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from prettytable import PrettyTable

import neat.hyperneat as hn
from neat.phenotypes import Phenotype, FeedforwardCUDA
from neat.mapElites import MapElitesConfiguration, MapElitesUpdate

import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

env_name = "BipedalWalker-v2"
nproc = 4

envs_size = 50
pop_size = 200
max_stagnation = 25
encoding_dim = 8
behavior_dimensions = 7
behavior_steps = 60
behavior_matrix_size = behavior_dimensions * behavior_steps

# _ALE_LOCK = threading.Lock()

def take_step(env_action):
    env, action = env_action

    state, reward, done, info = env.step(action)
    # with _ALE_LOCK:
        # env.render()

    return np.array([state, reward, done])

def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        env.seed(seed)
        return env
    return _f

def test_organism(phenotypes, envs, render=False):
    feedforward = FeedforwardCUDA(phenotypes)

    observations = envs.reset()

    obs_32 = np.float32(observations)
    actions = feedforward.update(obs_32)

    fitnesses = np.zeros(len(envs.remotes), dtype=np.float64)

    done = False
    done_tracker = np.array([False for _ in range(len(envs.remotes))])

    diff = len(phenotypes) - len(envs.remotes)
    if diff < 0:
        done_tracker[diff:] = True

    distances = np.zeros(len(envs.remotes))
    last_distances = np.zeros(len(envs.remotes))
    stagnations = np.zeros(len(envs.remotes))

    all_states = []


    max_steps = 50
    steps = max_steps
    while not done:
        actions = np.pad(actions, (0, abs(diff)), 'constant')
        states, rewards, dones, info = envs.step(actions)

        # if render:
        #     envs.remotes[0].send(('render', None))
        #     envs.remotes[0].recv()

        actions = feedforward.update(states)

        fitnesses[done_tracker == False] += np.around(rewards[done_tracker == False], decimals=4)
        # fitnesses[done_tracker == False] = np.around(rewards[done_tracker == False], decimals=2)

        envs_done = dones == True
        done_tracker[envs_done] = dones[envs_done]
        envs_running = len([d for d in done_tracker if d == False])

        # print("\r"+" "* 100, end='', flush=True)
        # print("\rEnvs running: {}/{}".format(envs_running, len(phenotypes)), end='')


        done = envs_running == 0

        distances += np.around(states.T[2], decimals=2)

        stagnations += distances == last_distances

        done_tracker[stagnations >= 100] = True

        last_distances = distances

        if steps == max_steps:
            steps = 0
            all_states.append(states[:, [0, 4, 6, 8, 9, 11, 13]])

        steps += 1

    all_states = np.array(all_states)
    flattened_states = []
    for row_i in range(all_states.shape[1]):
        flattened_states.append(all_states[:, row_i].flatten())

    flattened_states = pad_matrix(np.array(flattened_states), behavior_matrix_size)

    return (fitnesses, flattened_states)

def chunk_testing(phenotypes):
    all_states = []
    fitnesses = []
    for chunk in chunks(phenotypes, envs_size):
        fitness, states = test_organism(chunk, envs)
        all_states.extend(states)
        fitnesses.extend(fitness)

    return (np.array(all_states), np.array(fitnesses))

def run_env_once(phenotype):
    single_envs = SubprocVecEnv([make_env(env_name, envs_size)])
    test_organism([phenotype], single_envs, render=True)

    # Visualize().update(phenotype)
    feedforward_highest = FeedforwardCUDA([phenotype])
    states = env.reset()
    done = False
    distance = 0.0
    last_distance = 0.0
    distance_stagnation = 0
    while not done:
        actions = feedforward_highest.update(np.array([states]))
        states, reward, done, info = env.step(actions[0])
        distance += np.around(states[2], decimals=2)

        if distance <= last_distance:
            distance_stagnation += 1
        else:
            distance_stagnation = 0

        if distance_stagnation >= 100:
            done = True

        last_distance = distance

        env.render()
    env.close()
    # Visualize().close()

def refine_ae(autoencoder, phenotypes):
    autoencoder.load_weights('basic.h5')

    last_loss = 1000.0
    loss = 1000.0

    while loss <= last_loss:
        last_loss = loss

        states, _ = chunk_testing(phenotypes)

        ten_percent = max(1, int(states.shape[0] * 0.1))

        train = states[:-ten_percent]
        test = states[-ten_percent:]

        hist = autoencoder.fit(train, train,
                               epochs=50,
                               batch_size=256,
                               shuffle=True,
                               validation_data=(test, test),
                               verbose=0)

        loss = abs(hist.history['val_loss'][-1])

        print("Training autoencoder. Loss: {}".format(loss))

def pad_matrix(all_states, matrix_width):
    padded = []
    for row in all_states:
        # row = all_states[:, i].flatten()

        row = np.pad(row, (0, abs(matrix_width - row.shape[0])), 'constant')
        padded.append(row)

    return np.array(padded)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__ == '__main__':

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    sys.setrecursionlimit(10000)

    env = gym.make(env_name)

    inputs = env.observation_space.shape[0]
    outputs = env.action_space.shape[0]

    print("Inputs: {} | Outputs: {}".format(inputs, outputs))

    ############################################## Auto encoder ##############################################

    # this is our input placeholder
    input_img = Input(shape=(behavior_matrix_size,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(behavior_matrix_size, activation='relu')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.save_weights('basic.h5')
    ############################################################################################################

    print("Creating hyperneat object")
    # pop_config = SpeciesConfiguration(pop_size, inputs, outputs)
    pop_config = MapElitesConfiguration(4, pop_size, encoding_dim, inputs, outputs)
    # hyperneat = hn.HyperNEAT(pop_config)
    hyperneat = hn.NEAT(pop_config)

    start_fitness = np.zeros(pop_size)
    start_features = np.zeros((pop_size, encoding_dim))
    phenotypes: List[Phenotype] = hyperneat.epoch(MapElitesUpdate(start_fitness, start_features))
    # phenotypes: List[Phenotype] = hyperneat.epoch(SpeciesUpdate(start_fitness))

    highest_fitness: float = -1000.0
    highestDistance: float = 0.0

    envs = [make_env(env_name, seed) for seed in range(envs_size)]

    print("Creating envs...")
    envs = SubprocVecEnv(envs)
    print("Done.")

    loss = 1000.0
    progress_stagnation = 0
    train_ae = progress_stagnation >= max_stagnation

    refine_ae(autoencoder, phenotypes)


    epoch_num = 0
    while True:
        epoch_num += 1
        progress_stagnation += 1
        train_ae = progress_stagnation == max_stagnation

        if train_ae:
            ae_phenotypes = []
            sorted_genomes_and_fitness = []
            if len(hyperneat.population) > 0:

                sorted_genomes_and_fitness = sorted(hyperneat.population.population_and_fitnesses(), key=lambda a: a['fitness'])
                ae_phenotypes = [g['genome'].createPhenotype() for g in sorted_genomes_and_fitness]
            else:
                sorted_genomes_and_fitness = hyperneat.population.population_and_fitnesses()
                ae_phenotypes = phenotypes

            refine_ae(autoencoder, ae_phenotypes)
            top_percent = max(1, int(len(ae_phenotypes) * 0.25) + 1)
            ae_phenotypes = ae_phenotypes[:top_percent]
            # Re-evaluate all the genomes in the archives
            ae_states, ae_fitnesses = chunk_testing(ae_phenotypes)
            ae_pred = encoder.predict(ae_states)
            ae_fitnesses = ae_fitnesses[:top_percent]

            sorted_genomes = [g['genome'] for g in sorted_genomes_and_fitness]

            hyperneat.population.archive = {}
            hyperneat.population.archivedGenomes = []
            hyperneat.population.genomes = sorted_genomes[:top_percent]
            phenotypes = hyperneat.epoch(MapElitesUpdate(ae_fitnesses, ae_pred))

            progress_stagnation = 0


        print("########## Epoch {} ##########".format(epoch_num))

        # Test the phenotypes in the envs
        start = time.time()
        all_states, fitnesses = chunk_testing(phenotypes)
        end = time.time()
        print("Time:", end - start)


        pred = encoder.predict(all_states)

        print("Highest fitness this epoch:", max(fitnesses))
        max_fitness = max(zip(fitnesses, phenotypes), key=lambda e: e[0])

        # mpc = mp.get_context('spawn')
        # p = mpc.Process(target=run_env_once, args=(max_fitness[1],))
        # p.start()

        if max_fitness[0] > highest_fitness:
            highest_fitness = max_fitness[0] if max_fitness[0] > highest_fitness else highest_fitness

            best_phenotype = max_fitness[1]
            # Visualize().update(best_phenotype)
            run_env_once(best_phenotype)
            # Visualize().close()

            # progress_stagnation = 0
        # else:
        #     progress_stagnation += 1

        phenotypes = hyperneat.epoch(MapElitesUpdate(fitnesses, pred))

        print("Highest fitness all-time: {}".format(highest_fitness))
        print("Progress stagnation: {}".format(progress_stagnation))

        total = pow(hyperneat.population.configuration.mapResolution,
                    hyperneat.population.configuration.features)
        archiveFilled = len(hyperneat.population.archivedGenomes) / total
        print("Genomes in archive: {}".format(len(hyperneat.population.archive)))

        archive_size = max(1, len(hyperneat.population.archive))
        population = hyperneat.population.population()
        avg_neurons = sum([len(g.neurons) for g in population]) / archive_size - (
                    inputs + outputs)
        avg_links = sum([len(g.links) for g in population]) / archive_size
        avg_fitness = sum([g['fitness'] for g in hyperneat.population.population_and_fitnesses()]) / max(1, len(
            hyperneat.population.archive))

        table = PrettyTable(["Epoch", "fitness", "max fitness", "neurons", "links", "avg. fitness", "archive"])
        table.add_row([
            epoch_num,
            "{:1.4f}".format(max_fitness[0]),
            "{:1.4f}".format(highest_fitness),
            "{:1.4f}".format(avg_neurons),
            "{:1.4f}".format(avg_links),
            "{:1.4f}".format(avg_fitness),
            "{:1.8f}".format(archiveFilled)])
        print(table)

    env.close()
