from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import h5py

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.

num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.

train_file = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_train.hdf5','r')
print("keys: %s" % train_file.keys())
