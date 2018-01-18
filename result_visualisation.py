from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Masking
from keras.optimizers import *
from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import h5py
import json
from nltk.translate.bleu_score import sentence_bleu


class Inference:
    def __init__(self, dataset_file_path, vocabulary_file_path, encoder_model, decoder_model):
        self.vocab_json = json.load(open(vocabulary_file_path))
        self.num_decoder_tokens = len(self.vocab_json['idx_to_words'])
        self.words_to_idx = self.vocab_json["words_to_idx"]
        self.idx_to_words = self.vocab_json["idx_to_words"]
        self.dataset_file = h5py.File(dataset_file_path, 'r')
        self.story_ids = self.dataset_file["story_ids"]
        self.image_embeddings = self.dataset_file["image_embeddings"]
        self.story_sentences = self.dataset_file["story_sentences"]
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.vocab_size = len(self.words_to_idx)

        return

    def beam_search_predict_helper(self, live_sentence, state_values):
        probs = []
        new_states = []

        for i in range(len(live_sentence)):
            target_seq = np.zeros((1, 1))
            last_word = live_sentence[i][-1]
            target_seq[0, 0] = last_word
            output = self.decoder_model.predict([target_seq] + state_values[i])

            prob = output[0]
            states = output[1]
            reshaped_prob = np.reshape(prob, (self.vocab_size))
            probs.append(reshaped_prob)
            new_states.append(list(states))

        return probs, new_states

    def predict_story_beam_search(self, input_sequence, max_decoder_seq_length=22, images_per_story=5,
                                  image_embed_size=4096, beam_size=3):
        decoded_sentences = []
        scores = []
        m = 1

        for images in input_sequence:

            images = images.reshape((1, 5, 4096))
            state_value = self.encoder_model.predict(images)
            live_beam = 1
            live_sentence = [[self.words_to_idx["<START>"]]]
            live_score = [0]
            dead_beam = 0
            dead_sentence = []
            dead_scores = []
            state_values = []

            for i in range(beam_size):
                state_values.append(state_value)

            j = 0

            while live_beam and dead_beam < beam_size:

                probs, new_states = self.beam_search_predict_helper(live_sentence, state_values)

                if j == 0:
                    state_values = []
                    for _ in range(beam_size):
                        state_values.append(new_states[0])

                else:
                    state_values = new_states

                j += 1
                cand_scores = np.array(live_score)[:, None] - np.log(probs)
                cand_flat = cand_scores.flatten()

                ranks_flat = cand_flat.argsort()[:(beam_size - dead_beam)]
                live_score = cand_flat[ranks_flat]

                live_sentence = [live_sentence[r // self.vocab_size] + [r % self.vocab_size] for r in ranks_flat]

                zombie = [s[-1] == 2 or len(s) >= max_decoder_seq_length for s in live_sentence]

                # add zombies to the dead
                dead_sentence += [s for s, z in zip(live_sentence, zombie) if z]  # remove first label == empty
                dead_scores += [s for s, z in zip(live_score, zombie) if z]
                dead_beam = len(dead_sentence)
                # remove zombies from the living
                live_sentence = [s for s, z in zip(live_sentence, zombie) if not z]
                live_score = [s for s, z in zip(live_score, zombie) if not z]
                state_values = [s for s, z in zip(state_values, zombie) if not z]

                live_beam = len(live_sentence)

            decoded_sentences.append(dead_sentence + live_sentence)
            scores.append(dead_scores + live_score)

        # returns sentence in integer list, should fix this
        return decoded_sentences, scores

    def predict_batch(self, input_sequence, sentence_length):

        num_stories = input_sequence.shape[0]

        decoded_sentences = np.zeros((num_stories, sentence_length), dtype='int32')
        states_value = self.encoder_model.predict(input_sequence)
        states_value = [states_value]

        target_seq = np.zeros((num_stories, 1))
        target_seq[0:num_stories, 0] = self.words_to_idx["<START>"]

        stop_condition = False
        i = 0

        while not stop_condition:
             output_tokens, h = self.decoder_model.predict([target_seq] + states_value)

             sampled_word_index = np.argmax(output_tokens[:, 0, :], axis=1)
             print(sampled_word_index)

             if i >= sentence_length:
                 break

             decoded_sentences[:, i] = sampled_word_index.astype(dtype='int32')
             target_seq = np.zeros((num_stories, 1))
             target_seq[0:num_stories, 0] = sampled_word_index.astype(dtype='int32')
             states_value = [h]
             i += 1

        return decoded_sentences

    def predict_all(self, start_index, end_index, number_of_sentences=5, sentence_length=22, number_of_images=5, img_embedding_length=4096):
        num_stories = end_index - start_index
        input_id = self.story_ids[start_index:end_index]
        input_images = self.image_embeddings[start_index:end_index]
        input_sentences = self.story_sentences[start_index:end_index]
        print(input_id)

        encoder_batch_input_data = np.zeros((number_of_sentences * num_stories, number_of_images, img_embedding_length))
        print(encoder_batch_input_data.shape)
        print("input_images shape: ", input_images.shape)
        for k in range(num_stories):
            for j in range(number_of_images):
                encoder_batch_input_data[(k*j)+j:(k*j)+5, j] = input_images[k][j]


        original_sentences = []

        for j in range(num_stories):
            for story in input_sentences[j]:
                st=''
                for word in story:
                    if not (self.idx_to_words[word] == "<START>" or self.idx_to_words[word] == "<END>" or self.idx_to_words[word]=="<NULL>"):
                        st += self.idx_to_words[word] + " "

                original_sentences.append(st)


        decoded = self.predict_batch(encoder_batch_input_data, sentence_length)
        # #print(len(decoded))
        for i in range(number_of_sentences * num_stories):
            #score = sentence_bleu([original_sentences[i]],decoded[i])
            #print(i)
             print("Original", original_sentences[i])
             print("Decoded", self.vec_to_sentence(decoded[i], self.idx_to_words))
             #print(score)
        return

    def vec_to_sentence(self, sentence_vec, idx_to_word):
        """ Return human readable sentence of given sentence vector

        Parameters

        """
        words = []
        for word_idx in sentence_vec:
            word = idx_to_word[word_idx]
            if word == "<END>":
                break
            words.append(word)

        return " ".join(words)