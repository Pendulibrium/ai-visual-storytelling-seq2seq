from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Masking, GRU, TimeDistributed
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np
import h5py
import json
import time
import datetime
from model_data_generator import ModelDataGenerator
from seq2seqbuilder import  Seq2SeqBuilder


vocab_json = json.load(open('./dataset/vist2017_vocabulary.json'))
train_dataset = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_train.hdf5', 'r')
valid_dataset = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_valid.hdf5','r')
train_generator = ModelDataGenerator(train_dataset, vocab_json, 64)
valid_generator = ModelDataGenerator(valid_dataset, vocab_json, 64)
words_to_idx = vocab_json['words_to_idx']

batch_size = 13
epochs = 100 # Number of epochs to train for.
latent_dim = 1024  # Latent dimensionality of the encoding space.
word_embedding_size = 300 # Size of the word embedding space.
num_of_stacked_rnn = 1 # Number of Stacked RNN layers


learning_rate = 0.0001
gradient_clip_value = 5.0

num_samples = train_generator.num_samples
num_decoder_tokens = train_generator.number_of_tokens
valid_steps = valid_generator.num_samples / batch_size
#
# embeddings_index = {}
# f = open('glove.6B.300d.txt')
#
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
#
# print('Found %s word vectors.' % len(embeddings_index))
# embedding_matrix = np.random.normal(size=(train_generator.number_of_tokens, word_embedding_size))
# for word, i in words_to_idx.items():
#     embedding_vector = embeddings_index.get(word.replace('[', '').replace(']',''))
#     if embedding_vector is not None:
#         embedding_matrix[i]=embedding_vector
#
#
ts = time.time()
start_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
#
# print('Vocab size: ', num_decoder_tokens)
#
# # Shape (num_samples, 4096), 4096 is the image embedding length
# encoder_inputs = Input(shape=(None, 4096), name="encoder_input_layer")
#
# encoder_lstm_name="encoder_lstm_"
#
# for i in range(0, num_of_stacked_rnn):
#     if i == num_of_stacked_rnn-1:
#         encoder = GRU(latent_dim, return_state=True, name=encoder_lstm_name + str(i))
#     else:
#         encoder = GRU(latent_dim, return_sequences=True, return_state=True, name=encoder_lstm_name + str(i))
#
#     if i == 0:
#         encoder_outputs, state_h = encoder(encoder_inputs)
#     else:
#         encoder_outputs, state_h = encoder(encoder_outputs)
#
#
# encoder_states = state_h
#
# decoder_inputs = Input(shape=(22,), name="decoder_input_layer")
#
# embedding_layer = Embedding(num_decoder_tokens, word_embedding_size,
#                             weights=[embedding_matrix],
#                             input_length=22,
#                             mask_zero=True,
#                             trainable=False,
#                             name="embedding_layer")
# embedding_outputs = embedding_layer(decoder_inputs)
#
# decoder_lstm_name="decoder_lstm_"
#
# for i in range(0,num_of_stacked_rnn):
#     decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True, name=decoder_lstm_name + str(i))
#
#     if i == 0:
#         decoder_outputs, _ = decoder_gru(embedding_outputs, initial_state=encoder_states)
#     else:
#         decoder_outputs, _ = decoder_gru(decoder_outputs)
#
# decoder_dense = TimeDistributed(Dense(num_decoder_tokens, activation='softmax'), name="dense_layer")
# decoder_outputs = decoder_dense(decoder_outputs)
#
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

builder = Seq2SeqBuilder()
model = builder.build_encoder_decoder_model(latent_dim,words_to_idx, word_embedding_size, num_decoder_tokens, num_of_stacked_rnn, (None, 4096), (22,), cell_type=GRU)

#optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0, clipvalue = gradient_clip_value)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer = optimizer, loss='categorical_crossentropy')
checkpoint_name=start_time+"checkpoit.hdf5"
checkpointer = ModelCheckpoint(filepath='./checkpoints/'+checkpoint_name, verbose=1, save_best_only=True)
csv_logger = CSVLogger("./loss_logs/"+start_time+".csv", separator=',', append=False)
#tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names="embedding_layer", embeddings_metadata=None)

model.fit_generator(train_generator.multiple_samples_per_story_generator(), steps_per_epoch = num_samples / batch_size, epochs = epochs,
                    validation_data=valid_generator.multiple_samples_per_story_generator(), validation_steps=valid_steps, callbacks=[checkpointer, csv_logger])
ts = time.time()
end_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

model.save('./trained_models/' + str(start_time)+"-"+ str(end_time)+':image_to_text_gru.h5')

