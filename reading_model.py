from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Masking
from keras.optimizers import *
from keras.models import load_model
import numpy as np
import h5py
import json
# from nltk.translate.bleu_score import  sentence_bleu
from story_visualization import StoryPlot

latent_dim = 256

model = load_model("trained_models/2017-12-20_16:13:22-2017-12-21_08:08:17:image_to_text.h5")
print(model.layers)

encoder_inputs = model.inputs[0]
mask_layer = model.layers[2]
mask_tensor = mask_layer(encoder_inputs)
encoder_lstm = model.layers[4]
encoder_outputs, state_h, state_c = encoder_lstm(mask_tensor)
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = Input(shape=(1,), name="input_2")

embedding_layer = model.layers[3]
embedding_outputs = embedding_layer(decoder_inputs)
print("Embed", embedding_outputs.shape)
decoder_lstm = model.layers[5]
decoder_state_input_h = Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = Input(shape=(latent_dim,), name="input_4")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(embedding_outputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_dense = model.layers[6]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

vocab_json = json.load(open('./dataset/vist2017_vocabulary.json'))
num_decoder_tokens = len(vocab_json['idx_to_words'])
words_to_idx = vocab_json["words_to_idx"]
idx_to_words = vocab_json["idx_to_words"]

max_decoder_seq_length = 22

def decode_sequence(input_seq):
    decoded_sentences = []

    for images in input_seq:

        images = images.reshape((1, 5, 4096))
        decoded_sentence = ''
        states_value = encoder_model.predict(images)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = words_to_idx["<START>"]

        stop_condition = False
        i = 0

        while not stop_condition:
            i += 1

            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            sampled_word_index = np.argmax(output_tokens[0, -1, :])

            sampled_word = idx_to_words[sampled_word_index]

            if i > max_decoder_seq_length or sampled_word == "<END>":
                break
            decoded_sentence += sampled_word + " "
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_word_index
            states_value = [h, c]
        decoded_sentences.append(decoded_sentence)

    return decoded_sentences


train_file = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_valid.hdf5', 'r')
story_ids = train_file["story_ids"]
image_embeddings = train_file["image_embeddings"]
story_sentences = train_file["story_sentences"]

random_sample_index = np.random.randint(0, 4900)
input_id = story_ids[random_sample_index]
input_images = image_embeddings[random_sample_index]

input_senteces = story_sentences[random_sample_index]
print(input_id)

encoder_batch_input_data = np.zeros((5, 5, 4096))

print("input_images shape: ", input_images.shape)
for j in range(5):
    encoder_batch_input_data[j:5, j] = input_images[j]

original_sentences = []

for story in input_senteces:
    st = ''
    for word in story:
        if not (idx_to_words[word] == "<START>" or idx_to_words[word] == "<END>" or idx_to_words[word] == "<NULL>"):
            st += idx_to_words[word] + " "

    original_sentences.append(st)

decoded = decode_sequence(encoder_batch_input_data)

for i in range(5):
    # score = sentence_bleu([original_sentences[i]],decoded[i])
    print("Original", original_sentences[i])
    print("Decoded", decoded[i])
    # print(score)

story_plot = StoryPlot()
story_plot.visualize_story(str(input_id), decoded)