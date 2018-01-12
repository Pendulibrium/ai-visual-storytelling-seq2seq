from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Masking
from keras.optimizers import *
from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import h5py
import json

# from nltk.translate.bleu_score import  sentence_bleu

latent_dim = 512
num_of_stacked_rnn = 2

model = load_model("trained_models/2018-01-12_10:43:10-2018-01-12_10:47:39:image_to_text_gru.h5")
# print(model.layers)
# plot_model(model, to_file='model.png' , show_shapes= True)
encoder_inputs = Input(shape=(None, 4096), name="encoder_input_layer")
encoder_lstm_name = "encoder_lstm_"
encoder_lstm = model.get_layer(encoder_lstm_name + str(0))
encoder_outputs, state_h = encoder_lstm(encoder_inputs)

encoder_states = [state_h]

encoder_model = Model(encoder_inputs, encoder_states)
# plot_model(encoder_model, to_file='encoder_model.png' , show_shapes= True)

decoder_inputs = Input(shape=(22,), name="input_2")

embedding_layer = model.get_layer("embedding_layer")
embedding_outputs = embedding_layer(decoder_inputs)
print("Embed", embedding_outputs.shape)

decoder_lstm_name = "decoder_lstm_"
decoder_state_input_h = Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = Input(shape=(latent_dim,), name="input_4")
decoder_states_inputs = [decoder_state_input_h]

decoder_lstm = model.get_layer(decoder_lstm_name + str(0))
decoder_outputs, state_h = decoder_lstm(embedding_outputs, initial_state=decoder_states_inputs)
decoder_states = [state_h]

decoder_dense = model.get_layer("dense_layer")
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

vocab_json = json.load(open('./dataset/vist2017_vocabulary.json'))
num_decoder_tokens = len(vocab_json['idx_to_words'])
words_to_idx = vocab_json["words_to_idx"]
idx_to_words = vocab_json["idx_to_words"]

max_decoder_seq_length = 22


# plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True)

def decode_sequence(input_seq):
    decoded_sentences = []

    for images in input_seq:

        images = images.reshape((1, 1, 4096))

        decoded_sentence = ''
        states_value = encoder_model.predict(images)
        print(states_value[0][0:20])
        states_value = [states_value]

        target_seq = np.zeros((1, 22))
        target_seq[0, 0] = words_to_idx["<START>"]

        stop_condition = False
        i = 0
        #temperature = 0.5

        while not stop_condition:
            i += 1

            output_tokens, h = decoder_model.predict([target_seq] + states_value)
            # preds = np.asarray(output_tokens[0, -1, :]).astype('float64')
            # preds = np.log(preds) / temperature
            # exp_preds = np.exp(preds)
            # preds = exp_preds / np.sum(exp_preds)
            # probas = np.random.multinomial(1, preds, 1)
            # sampled_word_index = np.argmax(probas)
            sampled_word_index = np.argmax(output_tokens[0, -1, :])
            # print(sorted(output_tokens[0,0,:])[0:10])
            sampled_word = idx_to_words[sampled_word_index]

            if i >= max_decoder_seq_length or sampled_word == "<END>":
                break
            decoded_sentence += sampled_word + " "
            target_seq = np.zeros((1, 22))
            target_seq[0, 0] = sampled_word_index
            states_value = [h]
        decoded_sentences.append(decoded_sentence)

    return decoded_sentences


train_file = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_train.hdf5', 'r')
story_ids = train_file["story_ids"]
image_embeddings = train_file["image_embeddings"]
story_sentences = train_file["story_sentences"]

random_sample_index = np.random.randint(100)
# random_sample_index = 9
input_id = story_ids[random_sample_index]
input_images = image_embeddings[random_sample_index]

input_senteces = story_sentences[random_sample_index]
print(input_id)

encoder_batch_input_data = np.zeros((5, 1, 4096))

print("input_images shape: ", input_images.shape)
for j in range(5):
    encoder_batch_input_data[j, 0] = input_images[j]

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

