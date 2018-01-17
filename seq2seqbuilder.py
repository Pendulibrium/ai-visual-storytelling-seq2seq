from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Masking, GRU, TimeDistributed, RNN
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
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
                                    input_length=decoder_input_shape[0],
                                    mask_zero=True,
                                    trainable=False,
                                    name="embedding_layer")

        return embedding_layer

    def build_encoder_decoder_model(self, latent_dim, words_to_idx, word_embedding_size, num_tokens, num_stacked,
                    encoder_input_shape, decoder_input_shape, cell_type, masking=False):

        # Flag that tells us how many states does the RNN return (RNN, LSTM return 2 states, GRU returns 1 state)
        output_type = 0
        if LSTM == cell_type or RNN == cell_type:
            output_type = 2
        else:
            output_type = 1

        # Shape (num_samples, 4096), 4096 is the image embedding length
        encoder_inputs = Input(shape=encoder_input_shape, name="encoder_input_layer")

        # Masking layer, makes the network ignore 0 vectors
        if masking:
            mask_layer = Masking(mask_value=0, name="mask_layer")
            encoder_inputs = mask_layer(encoder_inputs)

        # Defining encoder layer
        encoder_lstm_name = "encoder_layer_"

        for i in range(0, num_stacked):
            if i == num_stacked - 1:
                encoder = cell_type(latent_dim, return_state=True, name=encoder_lstm_name + str(i))
            else:
                encoder = cell_type(latent_dim, return_sequences=True, return_state=True, name=encoder_lstm_name + str(i))

            if i == 0:
                encoder_outputs = encoder(encoder_inputs)
            else:
                encoder_outputs= encoder(encoder_outputs[0])

        encoder_states = encoder_outputs[1:]


        # Decoder input, should be shape (num_samples, 22)
        decoder_inputs = Input(shape=decoder_input_shape, name="decoder_input_layer")

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
