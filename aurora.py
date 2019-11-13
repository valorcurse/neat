from icontract import require, ensure

from neat.evaluation import Evaluation
class Aurora:

    def __init__(self, encoding_dim, inputs_dim):
        import keras
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.models import Model
        import tensorflow as tf


        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.inputs_dim = inputs_dim
        self.encoding_dim = encoding_dim

        # this is our input placeholder
        input_img = Input(shape=(self.inputs_dim,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(self.encoding_dim, activation='sigmoid')(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(self.inputs_dim, activation='relu')(encoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input_img, decoded)

        # this model maps an input to its encoded representation
        self.encoder = Model(input_img, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        self.autoencoder.save_weights('basic.h5')

    def refine(self, phenotypes, eval_env: Evaluation):
        self.autoencoder.load_weights('basic.h5')

        last_loss = 1000.0
        loss = 1000.0

        while loss <= last_loss:
            last_loss = loss

            _, states = eval_env.evaluate(phenotypes)

            ten_percent = max(1, int(states.shape[0] * 0.1))

            train = states[:-ten_percent]
            test = states[-ten_percent:]

            hist = self.autoencoder.fit(train, train,
                                        epochs=50,
                                        batch_size=256,
                                        shuffle=True,
                                        validation_data=(test, test),
                                        verbose=0)

            loss = abs(hist.history['val_loss'][-1])

            print("Training autoencoder. Loss: {}".format(loss))

    def characterize(self, features):
        assert features.shape[1] == self.inputs_dim, \
            "Size of features {} was not equal to size of autoencoder {}".format(features.shape[1], self.inputs_dim)

        prediction = self.encoder.predict(features, verbose=0)

        assert prediction.shape[1] == self.encoding_dim, \
            "Size of encoded feature {} was not the correct size of {}".format(prediction.shape[1], self.encoding_dim)

        return prediction