from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Masking, GRU, TimeDistributed, Dropout, Concatenate, Conv1D, \
    MaxPooling1D, Flatten
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.models import load_model
from keras import layers
import numpy as np
import abc


class SentenceEncoder(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_last_layer(self, encoder_states, sentence_embedding_outputs):
        return

    @abc.abstractmethod
    def get_last_layer_inference(self, model, encoder_states, sentence_embedding_outputs):
        return


class SentenceEncoderRNN(SentenceEncoder):
    def __init__(self, cell_type=None, sentence_encoder_latent_dim=None, recurrent_dropout=0.0):
        self.cell_type = cell_type
        self.sentence_encoder_latent_dim = sentence_encoder_latent_dim
        self.recurrent_dropout = recurrent_dropout

    def get_last_layer(self, encoder_states, sentence_embedding_outputs):
        encoder_sentence_lstm_name = "sentence_encoder_"
        sentence_encoder = self.cell_type(self.sentence_encoder_latent_dim, return_state=True,
                                          recurrent_dropout=self.recurrent_dropout,
                                          name=encoder_sentence_lstm_name + str(0))

        sentence_encoder_outputs = sentence_encoder(sentence_embedding_outputs)
        sentence_encoder_states = sentence_encoder_outputs[1:]

        # Merge states
        initial_encoder_states = []
        for i in range(len(sentence_encoder_states)):
            merged_decoder_states = layers.concatenate([encoder_states[i], sentence_encoder_states[i]], axis=-1)
            initial_encoder_states.append(merged_decoder_states)
        return initial_encoder_states

    def get_last_layer_inference(self, model, encoder_states, sentence_embedding_outputs):
        encoder_sentence_lstm_name = "sentence_encoder_0"
        sentence_encoder = model.get_layer(encoder_sentence_lstm_name)

        sentence_encoder_outputs = sentence_encoder(sentence_embedding_outputs)
        sentence_encoder_states = sentence_encoder_outputs[1:]

        # Merging
        initial_encoder_states = []
        new_latent_dim = 0
        for i in range(len(sentence_encoder_states)):
            merged_decoder_states = layers.concatenate([encoder_states[i], sentence_encoder_states[i]], axis=-1)
            new_latent_dim = merged_decoder_states.shape[1]
            initial_encoder_states.append(merged_decoder_states)

        return initial_encoder_states, new_latent_dim


class SentenceEncoderCNN(SentenceEncoder):
    def __init__(self, decoder_input_shape=None):
        self.decoder_input_shape = decoder_input_shape

    def get_last_layer(self, encoder_states, sentence_embedding_outputs):
        sentence_conv1_layer = Conv1D(filters=64, kernel_size=3, strides=1, activation='tanh',
                                      input_shape=self.decoder_input_shape, name="sentence_encoder_conv")
        sentence_encoder_outputs = sentence_conv1_layer(sentence_embedding_outputs)
        max_pool_layer = MaxPooling1D(pool_size=2)
        sentence_encoder_outputs = max_pool_layer(sentence_encoder_outputs)
        sentence_encoder_outputs = Flatten()(sentence_encoder_outputs)
        sentence_encoder_dense = Dense(512, name="sentence_encoder_dense")
        sentence_encoder_outputs = sentence_encoder_dense(sentence_encoder_outputs)

        # Merge states
        initial_encoder_states = []
        merged_decoder_states = layers.concatenate([encoder_states[0], sentence_encoder_outputs], axis=-1)
        initial_encoder_states.append(merged_decoder_states)
        return initial_encoder_states

    def get_last_layer_inference(self, model, encoder_states, sentence_embedding_outputs):
        sentence_conv1_layer = model.get_layer('sentence_encoder_conv')
        sentence_encoder_outputs = sentence_conv1_layer(sentence_embedding_outputs)
        max_pool_layer = MaxPooling1D(pool_size=2)
        sentence_encoder_outputs = max_pool_layer(sentence_encoder_outputs)
        sentence_encoder_outputs = Flatten()(sentence_encoder_outputs)
        sentence_encoder_dense = model.get_layer("sentence_encoder_dense")
        sentence_encoder_outputs = sentence_encoder_dense(sentence_encoder_outputs)

        # Merging
        initial_encoder_states = []
        merged_decoder_states = layers.concatenate([encoder_states[0], sentence_encoder_outputs], axis=-1)
        initial_encoder_states.append(merged_decoder_states)
        new_latent_dim = merged_decoder_states.shape[1]

        return initial_encoder_states, new_latent_dim


class Seq2SeqBuilder:
    def __init__(self):
        return

    # TODO: we should optimize this method.
    # Best way is to load the data once in the memory and consecutive calls should just return the
    # layer instead of loading all of the data again
    def get_embedding_layer(self, words_to_idx, word_embedding_size, num_tokens, mask_zero, name):
        embeddings_index = {}
        # GLOVE word weights for 6 billion words for word_embedding_size = 300
        f = open('glove.6B.300d.txt')

        # Getting the glove coefficients for our vocabulary from the file
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        # Filling the embedding matrix with the appropriate word coefficients
        embedding_matrix = np.random.normal(size=(num_tokens, word_embedding_size))
        for word, i in words_to_idx.items():
            embedding_vector = embeddings_index.get(word.replace('[', '').replace(']', ''))
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # Embedding layer that we don't train
        embedding_layer = Embedding(num_tokens, word_embedding_size,
                                    weights=[embedding_matrix],
                                    mask_zero=mask_zero,
                                    trainable=False,
                                    name=name)

        return embedding_layer

    def build_encoder_decoder_model(self, image_encoder_latent_dim, sentence_encoder_latent_dim, words_to_idx,
                                    word_embedding_size, num_tokens, num_stacked,
                                    encoder_input_shape, decoder_input_shape, cell_type, sentence_encoder,
                                    masking=False,
                                    recurrent_dropout=0.0, input_dropout=0.0, include_sentence_encoder=False):

        if not include_sentence_encoder:
            sentence_encoder_latent_dim = 0

        # Shape (num_samples, 4096), 4096 is the image embedding length
        encoder_inputs = Input(shape=encoder_input_shape, name="encoder_input_layer")

        # Masking layer, makes the network ignore 0 vectors
        if masking:
            mask_layer = Masking(mask_value=0, name="mask_layer")
            mask_output = mask_layer(encoder_inputs)

        # Defining encoder layer
        encoder_lstm_name = "encoder_layer_"

        for i in range(0, num_stacked):
            if i == num_stacked - 1:
                encoder = cell_type(image_encoder_latent_dim, return_state=True, recurrent_dropout=recurrent_dropout,
                                    name=encoder_lstm_name + str(i))
            else:

                if i == 0:
                    input_dropout = input_dropout
                else:
                    input_dropout = 0.0

                encoder = cell_type(image_encoder_latent_dim, return_sequences=True, return_state=True,
                                    recurrent_dropout=recurrent_dropout, dropout=input_dropout,
                                    name=encoder_lstm_name + str(i))

            if i == 0:
                if masking:
                    encoder_outputs = encoder(mask_output)
                else:
                    encoder_outputs = encoder(encoder_inputs)
            else:
                encoder_outputs = encoder(encoder_outputs[0])

        encoder_states = encoder_outputs[1:]

        # Defining sentence encoder

        if include_sentence_encoder:
            # We can use the same inpt shape as the decoder
            encoder_sentence_inputs = Input(shape=decoder_input_shape, name="sentence_encoder_input_layer")
            # Embedding layer that we don't train
            sentence_encoder_embedding_layer = self.get_embedding_layer(words_to_idx, word_embedding_size, num_tokens,
                                                                        mask_zero=False,
                                                                        name='sentence_embedding_layer')
            sentence_embedding_outputs = sentence_encoder_embedding_layer(encoder_sentence_inputs)

            initial_encoder_states = sentence_encoder.get_last_layer(encoder_states, sentence_embedding_outputs)
        else:
            initial_encoder_states = encoder_states

        # Decoder input, should be shape (num_samples, 22)
        decoder_inputs = Input(shape=decoder_input_shape, name="decoder_input_layer")

        # Embedding layer that we don't train
        embedding_layer = self.get_embedding_layer(words_to_idx, word_embedding_size, num_tokens, mask_zero=True,
                                                   name='decoder_embedding_layer')
        embedding_outputs = embedding_layer(decoder_inputs)

        # Defining decoder layer
        decoder_lstm_name = "decoder_layer_"
        decoder_latent_dim = image_encoder_latent_dim + sentence_encoder_latent_dim
        for i in range(0, num_stacked):

            if i == 0:
                decoder = cell_type(decoder_latent_dim, return_sequences=True, return_state=True,
                                    name=decoder_lstm_name + str(i))
                decoder_outputs = decoder(embedding_outputs, initial_state=initial_encoder_states)
            else:
                decoder = cell_type(decoder_latent_dim, return_sequences=True, return_state=True,
                                    recurrent_dropout=recurrent_dropout,
                                    name=decoder_lstm_name + str(i))
                decoder_outputs = decoder(decoder_outputs[0])

        dropout_layer = Dropout(input_dropout)
        dropout_outputs = dropout_layer(decoder_outputs[0])

        decoder_dense = TimeDistributed(Dense(num_tokens, activation='softmax'), name="dense_layer")
        decoder_outputs = decoder_dense(dropout_outputs)

        if include_sentence_encoder:
            model = Model([encoder_inputs, encoder_sentence_inputs, decoder_inputs], decoder_outputs)
        else:
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        return model

    def get_number_of_layers(self, model, layer_prefix):

        count = 0
        for i in range(len(model.layers)):
            if layer_prefix in model.layers[i].get_config()['name']:
                count += 1
        return count

    def build_encoder_decoder_inference_from_file(self, model_path, sentence_encoder, include_sentence_encoder=True):
        model = load_model(model_path)
        return self.build_encoder_decoder_inference(model, sentence_encoder, include_sentence_encoder)

    def build_encoder_decoder_inference(self, model, sentence_encoder, include_sentence_encoder=True):

        latent_dim = 0
        initial_encoder_states = []
        initial_input = []

        encoder_inputs = Input(shape=model.get_layer("encoder_input_layer").get_config()['batch_input_shape'][1:])
        print("encoder inputs shape: ", encoder_inputs.shape)

        mask_layer = Masking(mask_value=0, name="mask_layer")
        mask_output = mask_layer(encoder_inputs)

        encoder_lstm_prefix = "encoder_layer_"
        num_encoder = self.get_number_of_layers(model, encoder_lstm_prefix)
        print("num: ", num_encoder)
        for i in range(num_encoder):
            encoder = model.get_layer(encoder_lstm_prefix + str(i))
            weights = encoder.get_weights()
            config = encoder.get_config()
            config['dropout'] = 0.0
            config['recurrent_dropout'] = 0.0
            encoder = layers.deserialize({'class_name': encoder.__class__.__name__, 'config': config})

            if i == 0:
                encoder_outputs = encoder(mask_output)
                encoder.set_weights(weights)
                latent_dim = encoder.get_config()['units']
            else:
                encoder_outputs = encoder(encoder_outputs[0])
                encoder.set_weights(weights)

        encoder_states = encoder_outputs[1:]

        if include_sentence_encoder:

            encoder_sentence_inputs = Input(shape=(22,))
            initial_input = [encoder_inputs, encoder_sentence_inputs]

            sentence_encoder_embedding_layer = model.get_layer('sentence_embedding_layer')
            sentence_embedding_outputs = sentence_encoder_embedding_layer(encoder_sentence_inputs)

            initial_encoder_states, new_latent_dim = sentence_encoder.get_last_layer_inference(model, encoder_states, sentence_embedding_outputs)
        else:
            initial_input = encoder_inputs
            initial_encoder_states = encoder_states
            new_latent_dim = latent_dim

        # Test 1 return image embeddings
        # im_model = Model(initial_input, encoder_states)
        # Test 2 return sentence embeddings
        # sent_model = Model(initial_input, sentence_encoder_states)
        encoder_model = Model(initial_input, initial_encoder_states)

        decoder_inputs = Input(shape=(None,))

        embedding_layer = model.get_layer("decoder_embedding_layer")
        embedding_outputs = embedding_layer(decoder_inputs)

        decoder_prefix = "decoder_layer_"
        num_decoder = self.get_number_of_layers(model, decoder_prefix)

        if len(encoder_states) == 1:
            # decoder_states_inputs = [decoder_state_input_h1, decoder_state_input_h2]
            decoder_states_inputs = []
            for i in range(num_decoder):
                decoder_states_inputs.append(Input(shape=(new_latent_dim,)))
        else:  # TODO : test if this works with stacked LSTM model
            # decoder_states_inputs = [decoder_state_input_h1, decoder_state_input_c]
            decoder_states_inputs = []
            for i in range(num_decoder):
                decoder_states_inputs.append(Input(shape=(new_latent_dim,)))
                decoder_states_inputs.append(Input(shape=(new_latent_dim,)))

        decoder_states = []
        for i in range(num_decoder):

            decoder = model.get_layer(decoder_prefix + str(i))
            weights = decoder.get_weights()
            config = decoder.get_config()
            config['dropout'] = 0.0
            config['recurrent_dropout'] = 0.0
            decoder = layers.deserialize({'class_name': decoder.__class__.__name__, 'config': config})

            if i == 0:
                decoder_outputs = decoder(embedding_outputs, initial_state=decoder_states_inputs[i])
                decoder.set_weights(weights)
                decoder_states = decoder_states + list(decoder_outputs[1:])
            else:
                decoder_outputs = decoder(decoder_outputs[0], initial_state=decoder_states_inputs[i])
                decoder.set_weights(weights)
                decoder_states = decoder_states + list(decoder_outputs[1:])

        decoder_dense = model.get_layer("dense_layer")
        decoder_outputs = decoder_dense(decoder_outputs[0])
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        # return im_model, sent_model
        return encoder_model, decoder_model
