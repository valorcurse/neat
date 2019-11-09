from neat.evaluation import Evaluation

class Aurora:

    def __init__(self, encoding_dim, inputs_dim):

        import keras
        from keras.layers import Input, Dense
        from keras.models import Model
        from keras.backend.tensorflow_backend import set_session
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

        # this is our input placeholder
        input_img = Input(shape=(inputs_dim,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(inputs_dim, activation='relu')(encoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input_img, decoded)

        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
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
        return self.autoencoder.predict(features)