from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Masking
from keras.optimizers import *
from keras.models import load_model
from keras.preprocessing import sequence
import numpy as np
import h5py
import json
# from nltk.translate.bleu_score import  sentence_bleu
# from story_visualization import StoryPlot

latent_dim = 256
max_length = 22

model = load_model("trained_models/2017-12-20_16:13:22-2017-12-21_08:08:17:image_to_text.h5")
print(model.layers)

encoder_inputs = model.inputs[0]
mask_layer = model.layers[2]
mask_tensor = mask_layer(encoder_inputs)
encoder_lstm = model.layers[4]
encoder_outputs, state_h, state_c = encoder_lstm(mask_tensor)
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = Input(shape=(None,), name="input_2")

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


def predict(live_sentence, state_values):
    probs = []
    new_states = []

    for i in range(len(live_sentence)):
        target_seq=np.zeros((1,1))
        last_word = live_sentence[i][len(live_sentence[i])-1]
        target_seq[0,0] = last_word
        prob, h, c = decoder_model.predict([target_seq]+state_values[i])

        reshaped_prob = np.reshape(prob, (10004))
        probs.append(reshaped_prob)
        new_states.append([h,c])

    return probs, new_states


def decode_sequence(input_seq, beam_size=1 ,vocab_size=1):
    decoded_sentences = []
    scores = []
    m = 1
    for images in input_seq:

        images = images.reshape((1, 5, 4096))
        decoded_sentence = ''
        state_value = encoder_model.predict(images)
        live_beam = 1
        live_sentence = [[words_to_idx["<START>"]]]
        live_score = [0]
        dead_beam = 0
        dead_sentence = []
        dead_scores = []
        state_values = []

        for i in range(beam_size):
            state_values.append(state_value)

        j=0
        while live_beam and dead_beam < beam_size:

            probs, new_states = predict(live_sentence, state_values)
            if j == 0:
                state_values = []
                for _ in range(beam_size):
                   state_values.append(new_states[0])

            else:
                state_values = new_states
            j+=1
            cand_scores = np.array(live_score)[:,None] - np.log(probs)
            cand_flat = cand_scores.flatten()

            ranks_flat = cand_flat.argsort()[:(beam_size-dead_beam)]
            live_score = cand_flat[ranks_flat]

            live_sentence= [live_sentence[r//vocab_size]+[r%vocab_size] for r in ranks_flat]

            zombie = [s[-1] == 2 or len(s) >= max_length for s in live_sentence]


            # add zombies to the dead
            dead_sentence += [s for s, z in zip(live_sentence, zombie) if z]  # remove first label == empty
            dead_scores += [s for s, z in zip(live_score, zombie) if z]
            dead_beam = len(dead_sentence)
            # remove zombies from the living
            live_sentence= [s for s, z in zip(live_sentence, zombie) if not z]
            live_score = [s for s, z in zip(live_score, zombie) if not z]
            live_beam = len(live_sentence)

        decoded_sentences.append(dead_sentence+live_sentence)
        scores.append(dead_scores+live_score)

    return decoded_sentences, scores



def integers_to_sentence(sentence, idx_to_word):
    result = ''
    for id in sentence:
        result = result + idx_to_word[id]+" "

    return result
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

decoded,scores = decode_sequence(encoder_batch_input_data, beam_size=3, vocab_size=len(words_to_idx))
#print decoded
for i in range(5):
    # score = sentence_bleu([original_sentences[i]],decoded[i])
    print("Original", original_sentences[i])

    print("Decoded", integers_to_sentence(decoded[i][0],idx_to_words))
    print("Decoded", integers_to_sentence(decoded[i][1], idx_to_words))
    print("Decoded", integers_to_sentence(decoded[i][2], idx_to_words))
    # print("Decoded", integers_to_sentence(decoded[i][3], idx_to_words))
    # print("Decoded", integers_to_sentence(decoded[i][4], idx_to_words))
    # print("Decoded", integers_to_sentence(decoded[i][5], idx_to_words))
    # print("Decoded", integers_to_sentence(decoded[i][6], idx_to_words))
    print(scores[i])
    # print(score)


#story_plot = StoryPlot()
#story_plot.visualize_story(str(input_id), decoded)