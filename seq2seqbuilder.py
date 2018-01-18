from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Masking, GRU, TimeDistributed, RNN
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.models import load_model
import numpy as np

class Seq2SeqBuilder:
    def __init__(self):
        return

    def get_embedding_layer(self, words_to_idx, word_embedding_size, num_tokens, decoder_input_shape):
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
                                    mask_zero=True,
                                    trainable=False,
                                    name="embedding_layer")

        return embedding_layer

    def build_encoder_decoder_model(self, latent_dim, words_to_idx, word_embedding_size, num_tokens, num_stacked,
                    encoder_input_shape, decoder_input_shape, cell_type, masking=False):
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
                encoder = cell_type(latent_dim, return_state=True, name=encoder_lstm_name + str(i))
            else:
                encoder = cell_type(latent_dim, return_sequences=True, return_state=True, name=encoder_lstm_name + str(i))

            if i == 0:
                if masking:
                    encoder_outputs = encoder(mask_output)
                else:
                    encoder_outputs = encoder(encoder_inputs)
            else:
                encoder_outputs= encoder(encoder_outputs[0])

        encoder_states = encoder_outputs[1:]


        # Decoder input, should be shape (num_samples, 22)
        decoder_inputs = Input(shape=(None,), name="decoder_input_layer")

        # Embedding layer that we don't train
        embedding_layer = self.get_embedding_layer(words_to_idx, word_embedding_size, num_tokens, decoder_input_shape)
        embedding_outputs = embedding_layer(decoder_inputs)

        # Defining decoder layer
        decoder_lstm_name = "decoder_layer_"

        for i in range(0, num_stacked):
            decoder = cell_type(latent_dim, return_sequences=True, return_state=True, name=decoder_lstm_name + str(i))

            if i == 0:
                decoder_outputs = decoder(embedding_outputs, initial_state=encoder_states)
            else:
                decoder_outputs = decoder(decoder_outputs[0])

        decoder_dense = TimeDistributed(Dense(num_tokens, activation='softmax'), name="dense_layer")
        decoder_outputs = decoder_dense(decoder_outputs[0])

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model

    def get_number_of_layers(self, model, layer_prefix):

        count = 0
        for i in range(len(model.layers)):
            if layer_prefix in model.layers[i].get_config()['name']:
                count += 1
        return count

    def build_encoder_decoder_inference(self, model_name):

        latent_dim = 0
        model = load_model(model_name)
        #print(model.layers[3].get_config()['units'])
        #a = model.layers[2].get_config()
        #dec_inp = Embedding.from_config(a)
        #inp = Input(shape=model.layers[0].get_config()['batch_input_shape'])
        # plot_model(model, to_file='model.png' , show_shapes= True)

        encoder_inputs = Input(shape=model.get_layer("encoder_input_layer").get_config()['batch_input_shape'][1:])
        encoder_lstm_prefix = "encoder_lstm_"
        num_encoder = self.get_number_of_layers(model, encoder_lstm_prefix)

        for i in range(num_encoder):
            encoder = model.get_layer(encoder_lstm_prefix + str(0))
            if i == 0:
                encoder_outputs = encoder(encoder_inputs)
                latent_dim = encoder.get_config()['units']
            else:
                encoder_outputs = encoder(encoder_outputs[0])
        encoder_states = list(encoder_outputs[1:])

        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = Input(shape=(None,))

        embedding_layer = model.get_layer("embedding_layer")
        embedding_outputs = embedding_layer(decoder_inputs)

        decoder_prefix = "decoder_lstm_"
        num_decoder = self.get_number_of_layers(model, encoder_lstm_prefix)
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        if len(encoder_states) == 1:
            decoder_states_inputs = [decoder_state_input_h]
        else:
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        for i in range(num_decoder):
            decoder = model.get_layer(decoder_prefix + str(0))
            if i == 0:
                decoder_outputs = decoder(embedding_outputs, initial_state=decoder_states_inputs)
            else:
                decoder_outputs = decoder(decoder_outputs[0])

        decoder_states = list(decoder_outputs[1:])

        decoder_dense = model.get_layer("dense_layer")
        decoder_outputs = decoder_dense(decoder_outputs[0])
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        return encoder_model, decoder_model
