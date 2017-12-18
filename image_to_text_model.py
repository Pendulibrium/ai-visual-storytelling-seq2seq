from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
import h5py
import json
import time

t=time.time()
vocab_json = json.load(open('./dataset/vist2017_vocabulary.json'))
train_file = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_train.hdf5','r')

batch_size = 64  # Batch size for training.
epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = len(train_file["story_ids"])

num_decoder_tokens = len(vocab_json['idx_to_words'])

max_encoder_seq_length = train_file['image_embeddings'].shape[1]
max_decoder_seq_length = train_file['story_sentences'].shape[1] * train_file['story_sentences'].shape[2]

print('Number of samples:', train_file['image_embeddings'].shape[0])
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
print('Vocab size: ', num_decoder_tokens)

image_embeddings = train_file["image_embeddings"]
story_sentences = train_file["story_sentences"]
image_embeddings2 = []
print(time.time()-t)
t=time.time()
for embedding_list in image_embeddings:

    for i in range(len(embedding_list)):
            zero_image = np.zeros(4096).tolist()
            if i==0:
                image_embeddings2.append([embedding_list[i], zero_image, zero_image, zero_image, zero_image])
            elif i==1:
                image_embeddings2.append([embedding_list[i-1], embedding_list[i], zero_image, zero_image, zero_image])
            elif i==2:
                image_embeddings2.append([embedding_list[i-2], embedding_list[i-1], embedding_list[i], zero_image, zero_image])
            elif i==3:
                image_embeddings2.append([embedding_list[i-3],embedding_list[i-2],embedding_list[i-1], embedding_list[i], zero_image])
            elif i==4:
                image_embeddings2.append([embedding_list[i-4], embedding_list[i-3], embedding_list[i-2], embedding_list[i-1], embedding_list[i]])

encoder_input_data = np.array(image_embeddings2)
print(encoder_input_data.shape)
print(time.time()-t)
t=time.time()
decoder_input_data = []
decoder_target_data = np.zeros((story_sentences.shape[0]*story_sentences.shape[1], story_sentences.shape[2], num_decoder_tokens), dtype='float32')

for story_index in range(len(story_sentences)):
    story = story_sentences[story_index]
    for sentence_index in range(len(story)):
        sentence = story[sentence_index]
        decoder_input_data.append(sentence)
        for word_index in range(len(sentence)):
            if word_index > 0:
                decoder_target_data[story_index*sentence_index, word_index-1, sentence[word_index]] = 1

decoder_input_data = np.array(decoder_input_data)
print(decoder_input_data.shape)
print(decoder_target_data.shape)
print(time.time()-t)

# Shape (num_samples, 4096), 4096 is the image embedding length
encoder_inputs = Input(shape = (None, 4096))
print(encoder_inputs.shape)
encoder = LSTM(latent_dim, return_state = True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape = (22,))
print(decoder_inputs.shape)
embedding_layer = Embedding(num_decoder_tokens, 300, mask_zero=True)
decoder_inputs = embedding_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model ([encoder_inputs, decoder_inputs], decoder_outputs)
print("se e ok do tuka")
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
print("sledi treniranje")
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
print("zavrshi treniranjeto")
