from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Masking
from keras.optimizers import *
from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import h5py
import json
from nltk.translate.bleu_score import sentence_bleu
from model_data_generator import ModelDataGenerator
from nlp import nlp
from seq2seqbuilder import Seq2SeqBuilder
from nlp.scores import Scores, Score_Method


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
        self.num_stacked_layers = Seq2SeqBuilder().get_number_of_layers(encoder_model,layer_prefix="encoder_layer_")

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
            states = output[1:]
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
            images = images.reshape(1, 5, 4096)
            state_value = self.encoder_model.predict(images)
            state_value_shape = state_value.shape
            live_beam = 1
            live_sentence = [[self.words_to_idx["<START>"]]]
            live_score = [0]
            dead_beam = 0
            dead_sentence = []
            dead_scores = []
            state_values = []

            for i in range(beam_size):

                state_value_tmp = [state_value]
                for i in range(self.num_stacked_layers - 1):
                    state_value_tmp.append(np.zeros(state_value_shape))

                state_values.append(state_value_tmp)

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

    def predict_batch(self, input_sequence, sentence_length, ):

        num_stories = input_sequence.shape[0]

        decoded_sentences = np.zeros((num_stories, sentence_length), dtype='int32')
        states_value = self.encoder_model.predict(input_sequence)
        states_value_shape = states_value.shape
        states_value = [states_value]
        for i in range(self.num_stacked_layers-1):
            states_value.append(np.zeros(states_value_shape))
        # TODO: check if the type is list

        target_seq = np.zeros((num_stories, 1), dtype='int32')
        target_seq[0:num_stories, 0] = self.words_to_idx["<START>"]

        stop_condition = False
        i = 0

        while not stop_condition:
            output = self.decoder_model.predict([target_seq] + states_value)
            output_tokens = output[0]

            sampled_word_index = np.argmax(output_tokens[:, 0, :], axis=1).astype(dtype='int32')

            if i >= sentence_length:
                break

            decoded_sentences[:, i] = sampled_word_index
            target_seq[0:num_stories, 0] = sampled_word_index

            states_value = output[1:]
            i += 1

        return decoded_sentences

    # "Why we don't use the generator"?
    # TODO: If we send start_index = 0 and end index is the last story then we will have memory overflow
    def predict_all(self, batch_size, sentence_length=22):

        data_generator = ModelDataGenerator(self.dataset_file, self.vocab_json, batch_size)
        bleu_score = 0.0
        meteor_score = 0.0
        count = 0
        for batch in data_generator.multiple_samples_per_story_generator(reverse=False, only_one_epoch=True):
            count+=1
            encoder_batch_input_data = batch[0][0]
            original_sentences_input = batch[0][1]

            references = []
            hypotheses = []



            decoded = self.predict_batch(encoder_batch_input_data, sentence_length)
            # encoder_batch_input_data = encoder_batch_input_data[0:1,]
            for i in range(encoder_batch_input_data.shape[0]):
                original = nlp.vec_to_sentence(original_sentences_input[i], self.idx_to_words)
                result = nlp.vec_to_sentence(decoded[i], self.idx_to_words)
                hypotheses.append(original)
                references.append(result)
                #print("Original", original)
                #print("Decoded", result)

            meteor_score += Scores().calculate_scores(Score_Method.METEOR, references,hypotheses)[1]
            #bleu_score += Scores().calculate_scores(Score_Method.BLEU, references, hypotheses)[1]
            print(str(count) + ":" + str(meteor_score))

        print("Total", meteor_score / (count * batch_size))
        #print("Total", bleu_score/(count*batch_size))
        #print(Scores().calculate_scores(Score_Method.BLEU, references,hypotheses))
        #print(Scores().calculate_scores(Score_Method.METEOR, references, hypotheses))

    def predict_all_beam_search(self, batch_size, beam_size=3, sentence_length=22):
        data_generator = ModelDataGenerator(self.dataset_file, self.vocab_json, batch_size)
        count = 0
        for batch in data_generator.multiple_samples_per_story_generator(reverse=False, only_one_epoch=True):

            encoder_batch_input_data = batch[0][0]
            original_sentences_input = batch[0][1]
            print(encoder_batch_input_data.shape)
            # encoder_batch_input_data = encoder_batch_input_data[0:1,]
            decoded = self.predict_story_beam_search(encoder_batch_input_data, beam_size=beam_size)
            for i in range(len(decoded[0])):

                original = nlp.vec_to_sentence(original_sentences_input[i], self.idx_to_words)
                print("Original", original)
                for j in range(beam_size):
                    result = nlp.vec_to_sentence(decoded[0][i][j], self.idx_to_words)
                    print("Decoded", result)
                print(decoded[1][i])
                max_score_index = np.argmin(decoded[1][i])
                print("Decoded", decoded[0][i][max_score_index])
            break
    # print(Scores().calculate_scores(Score_Method.BLEU, references,hypotheses))
    # print(Scores().calculate_scores(Score_Method.METEOR, references, hypotheses))