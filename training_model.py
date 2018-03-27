from keras.layers import LSTM, GRU, CuDNNGRU
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, CSVLogger
from result_visualisation import NLPScores
import numpy as np
import h5py
import json
import time
import datetime
from model_data_generator import ModelDataGenerator
from seq2seqbuilder import Seq2SeqBuilder, SentenceEncoderCNN, SentenceEncoderRNN
from report.report_writer import *
from util import util
from keras.utils import plot_model

vocab_json = json.load(open('./dataset/vist2017_vocabulary.json'))
train_dataset = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_train.hdf5', 'r')
valid_dataset = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_valid.hdf5', 'r')
train_generator = ModelDataGenerator(train_dataset, vocab_json, 64)
valid_generator = ModelDataGenerator(valid_dataset, vocab_json, 64)
words_to_idx = vocab_json['words_to_idx']

batch_size = 13
epochs = 25  # Number of epochs to train for.
image_encoder_latent_dim = 512  # Latent dimensionality of the encoding space.
sentence_encoder_latent_dim = 512

word_embedding_size = 300  # Size of the word embedding space.
num_of_stacked_rnn = 2  # Number of Stacked RNN layers
cell_type = GRU
learning_rate = 0.0001
gradient_clip_value = 5.0
reverse = False
last_k = 3

num_samples = train_generator.num_samples
num_decoder_tokens = train_generator.number_of_tokens
valid_steps = (valid_generator.num_samples // batch_size) + 1
train_steps = (num_samples // batch_size) + 1
print("num samples: ", num_samples)
print("train steps: ", train_steps)

start_time = time.time()
start_time_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H:%M:%S')

# Build model
encoder_input_shape = (None, 4096)
decoder_input_shape = (22,)
builder = Seq2SeqBuilder()
#sentence_encoder = SentenceEncoderCNN(decoder_input_shape=decoder_input_shape)
sentence_encoder = SentenceEncoderRNN(cell_type=cell_type, sentence_encoder_latent_dim=sentence_encoder_latent_dim, recurrent_dropout=0.0)
model = builder.build_encoder_decoder_model(image_encoder_latent_dim, sentence_encoder_latent_dim, words_to_idx,
                                            word_embedding_size, num_decoder_tokens,
                                            num_of_stacked_rnn, encoder_input_shape, decoder_input_shape,
                                            cell_type=cell_type, sentence_encoder=sentence_encoder, masking=True,
                                            recurrent_dropout=0.0, input_dropout=0.5, include_sentence_encoder=True)

optimizer = Adam(lr=learning_rate, clipvalue=gradient_clip_value)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

# Callbacks
checkpoint_name = start_time_string + "checkpoint.hdf5"
checkpointer = ModelCheckpoint(monitor='loss', filepath='./checkpoints/' + checkpoint_name, verbose=1,
                               save_best_only=True)

csv_logger_filename = "./loss_logs/" + start_time_string + ".csv"
csv_logger = CSVLogger(csv_logger_filename, separator=',', append=False)

nlpScores = NLPScores('valid')

# Start training

# train_generator.multiple_samples_per_story_generator(reverse=reverse, shuffle=True, last_k=last_k,
#                                                          only_one_epoch=True, sentence_embedding=True)

hist = model.fit_generator(
    train_generator.multiple_samples_per_story_generator(reverse=reverse, shuffle=True, last_k=last_k,
                                                         sentence_embedding=True),
    steps_per_epoch=train_steps, epochs=epochs, callbacks=[checkpointer, csv_logger])

end_time = time.time()
end_time_string = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d_%H:%M:%S')

model_filename = './trained_models/' + str(start_time_string) + "-" + str(end_time_string) + '.h5'
model.save(model_filename)

# Write report
duration_string = util.seconds_to_formatted_string(end_time - start_time)

print(hist.history)
history = hist.history
val_loss = -1
if 'val_loss' in history:
    val_loss = history['val_loss'][-1]

writer = ReportWriter('./results/model_results.csv')
writer.write(num_samples=5 * num_samples, duration=duration_string, num_epochs=epochs, loss=history['loss'][-1],
             val_loss=val_loss,
             num_layers=num_of_stacked_rnn,
             cell_type=str(cell_type.__name__).lower(),
             activation='tanh', hidden_dimension=image_encoder_latent_dim, learning_rate=learning_rate,
             gradient_clipping_value=gradient_clip_value,
             optimizer=type(optimizer).__name__.lower(),
             loss_history_filename=csv_logger_filename, model_filename=model_filename, reverse_sequence=reverse,
             notes='')
