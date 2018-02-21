from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Masking
from keras.optimizers import *
from keras.models import load_model
from keras.utils import plot_model
from keras.callbacks import Callback
import numpy as np
import h5py
import json
from nltk.translate.bleu_score import sentence_bleu
from model_data_generator import ModelDataGenerator
from nlp import nlp
from seq2seqbuilder import Seq2SeqBuilder
from nlp.scores import Scores, Score_Method
import time
import commands


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

        self.num_stacked_layers = Seq2SeqBuilder().get_number_of_layers(encoder_model, layer_prefix="encoder_layer_")

        return

    def beam_search_predict_helper(self, live_sentence, state_values):

        target_seq = np.zeros((self.get_number_of_sentences(live_sentence), 1))
        last_words = np.zeros(self.get_number_of_sentences(live_sentence))
        i = 0
        for sentences in live_sentence:
            for sentence in sentences:
                last_words[i] = (sentence[-1])
                i += 1
        target_seq[:, 0] = last_words
        # print(target_seq.shape)
        # print(np.array(state_values).shape)
        output = self.decoder_model.predict([target_seq] + state_values)
        prob = output[0]
        states = output[1:]

        return prob, states

    def predict_story_beam_search(self, input_sequence, max_decoder_seq_length=22, batch_size=5,
                                  image_embed_size=4096, beam_size=3):

        encoder_states_value = self.encoder_model.predict(input_sequence)
        decoded_sentences = []
        scores = []
        batch_size = encoder_states_value.shape[0]
        states_value_shape = encoder_states_value.shape
        new_encoder_states = [encoder_states_value]

        for _ in range(self.num_stacked_layers - 1):
            new_encoder_states.append(np.zeros(states_value_shape))

        live_beam = np.ones(batch_size)
        live_sentences = []
        for i in range(batch_size):
            live_sentences.append([[self.words_to_idx["<START>"]]])
        live_score = []
        for i in range(batch_size):
            live_score.append([0])

        dead_beam = np.zeros(batch_size)
        dead_sentences = []
        for i in range(batch_size):
            dead_sentences.append([])
        dead_scores = []
        for i in range(batch_size):
            dead_scores.append([])
        state_values = []
        for i in range(self.num_stacked_layers):
            state_values.append([])
        j = 0

        while self.check_live_beams(live_beam) and self.check_dead_beams(dead_beam, beam_size):
            probs, new_states = self.beam_search_predict_helper(live_sentences, new_encoder_states)
            probs = np.reshape(probs, (probs.shape[0], probs.shape[2]))
            start_index = 0

            for i in range(len(live_sentences)):

                live_sentence_tmp = live_sentences[i]
                if len(live_sentence_tmp) == 0:
                    # print("prazna: ", i)
                    continue
                live_beam_tmp = int(live_beam[i])
                live_score_tmp = live_score[i]
                dead_beam_tmp = int(dead_beam[i])
                dead_sentence_tmp = dead_sentences[i]
                dead_score_tmp = dead_scores[i]
                state_values_tmp = []
                for _ in range(self.num_stacked_layers):
                    state_values_tmp.append([])

                if j == 0:
                    probs_tmp = probs[i]
                    for k in range(self.num_stacked_layers):
                        for _ in range(beam_size):
                            state_values_tmp[k].append(new_states[k][i, :])
                else:
                    end_index_tmp = start_index + live_beam_tmp
                    probs_tmp = probs[start_index: end_index_tmp]

                cand_scores = np.array(live_score_tmp)[:, None] - np.log(probs_tmp)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(beam_size - dead_beam_tmp)]
                live_score_tmp = cand_flat[ranks_flat]
                live_sentence_tmp = [live_sentence_tmp[r // self.vocab_size] + [r % self.vocab_size] for r in
                                     ranks_flat]

                if j != 0:
                    end_index_tmp = start_index + live_beam_tmp
                    # get the states from the prediction
                    for k in range(self.num_stacked_layers):
                        for v in range(start_index, end_index_tmp):
                            state_values_tmp[k].append(new_states[k][v, :])
                    start_index = end_index_tmp
                    # assign which state comes from what sentence
                    state_values_helper = []
                    for _ in range(self.num_stacked_layers):
                        state_values_helper.append([])
                    # print("new sentence")
                    for k in range(self.num_stacked_layers):
                        for r in ranks_flat:
                            r_sentence = r // self.vocab_size
                            # print("r:", r_sentence)
                            state_values_helper[k].append(state_values_tmp[k][r_sentence])
                    state_values_tmp = state_values_helper

                zombie = [s[-1] == 2 or len(s) >= max_decoder_seq_length for s in live_sentence_tmp]
                # print(zombie)
                # add zombies to the dead
                dead_sentence_tmp += [s for s, z in zip(live_sentence_tmp, zombie) if z]  # remove first label == empty
                dead_score_tmp += [s for s, z in zip(live_score_tmp, zombie) if z]
                dead_beam_tmp = len(dead_sentence_tmp)
                # remove zombies from the living
                live_sentence_tmp = [s for s, z in zip(live_sentence_tmp, zombie) if not z]
                live_score_tmp = [s for s, z in zip(live_score_tmp, zombie) if not z]

                for k in range(self.num_stacked_layers):
                    state_values_tmp[k] = [s for s, z in zip(state_values_tmp[k], zombie) if not z]

                # state_values = [s for s, z in zip(state_values, zombie) if not z]
                live_beam_tmp = len(live_sentence_tmp)
                # update parameters
                live_sentences[i] = live_sentence_tmp
                live_beam[i] = live_beam_tmp
                live_score[i] = live_score_tmp
                dead_beam[i] = dead_beam_tmp
                dead_sentences[i] = dead_sentence_tmp
                dead_scores[i] = dead_score_tmp
                for k in range(self.num_stacked_layers):
                    tmp = state_values_tmp[k]
                    for v in range(len(tmp)):
                        state_values[k].append(tmp[v])

            # print(live_sentences)
            # print(np.array(state_values).shape)
            # Update state_values so they can be compatible with the upper predict call
            new_encoder_states = []
            for k in range(len(state_values)):
                new_encoder_states.append(np.array(state_values[k]))
            state_values = []
            for k in range(self.num_stacked_layers):
                state_values.append([])
            j += 1
            decoded_sentences = dead_sentences + live_sentences
            scores = dead_scores + live_score

        decoded_sentences = [x for x in decoded_sentences if x != []]
        scores = [x for x in scores if x != []]
        return decoded_sentences, scores

    def predict_batch(self, input_sequence, sentence_length):

        num_stories = input_sequence.shape[0]

        decoded_sentences = np.zeros((num_stories, sentence_length), dtype='int32')
        states_value = self.encoder_model.predict(input_sequence)
        states_value_shape = states_value.shape
        states_value = [states_value]
        for i in range(self.num_stacked_layers - 1):
            states_value.append(np.zeros(states_value_shape))

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

    # TODO: we should send the reverse params
    def predict_all(self, batch_size, sentence_length=22, references_file_name='', hypotheses_file_name='',
                    sentence_embedding=True, no_duplicates=False, beam_search=False, beam_size=3):

        data_generator = ModelDataGenerator(self.dataset_file, self.vocab_json, batch_size)
        count = 0

        references = []
        hypotheses = []

        for batch in data_generator.multiple_samples_per_story_generator(reverse=False, only_one_epoch=True, last_k=3):
            count += 1

            print("batch_number: ", count)
            if sentence_embedding:
                encoder_batch_input_data = batch[0][0]
                encoder_batch_sentence_input_data = batch[0][1]
                original_sentences_input = batch[0][2]

                print(len(original_sentences_input))

            else:
                encoder_batch_input_data = batch[0][0]
                original_sentences_input = batch[0][1]
                # fast prediction with duplicates
                decoded = self.predict_batch(encoder_batch_input_data, sentence_length)

            words = [self.words_to_idx["<UNK>"]]
            for i in range(encoder_batch_input_data.shape[0]):

                if sentence_embedding:
                    if i % 5 == 0:
                        encoder_sentence = encoder_batch_sentence_input_data[i]
                        words = [self.words_to_idx["<UNK>"]]
                    if no_duplicates:
                        decoded = self.predict_batch_with_sentence_embed(encoder_batch_input_data[i],
                                                                         encoder_sentence=encoder_sentence,
                                                                         sentence_embed_boolean=True,
                                                                         sentence_length=sentence_length, words=words,
                                                                         no_duplicates=True)
                    else:
                        if beam_search:

                            decoded = self.slow_beam_search(encoder_batch_input_data[i], encoder_sentence)
                            decoded = np.array(decoded)
                            print("Hola")
                        else:
                            decoded = self.predict_batch_with_sentence_embed(encoder_batch_input_data[i],
                                                                         encoder_sentence=encoder_sentence,
                                                                         sentence_embed_boolean=True,
                                                                         sentence_length=sentence_length, words=words,
                                                                         no_duplicates=False)

                    encoder_sentence = decoded
                    original = nlp.vec_to_sentence(original_sentences_input[i], self.idx_to_words)
                    if no_duplicates:
                        words = words + decoded.tolist()
                    result = nlp.vec_to_sentence(decoded, self.idx_to_words)
                else:
                    original = nlp.vec_to_sentence(original_sentences_input[i], self.idx_to_words)

                    if no_duplicates:
                        if i % 5 == 0:
                            words = [self.words_to_idx["<UNK>"]]
                        # reusing same function
                        decoded = self.predict_batch_with_sentence_embed(encoder_batch_input_data[i],
                                                                         encoder_sentence=None,
                                                                         sentence_embed_boolean=False,
                                                                         sentence_length=sentence_length, words=words)
                        words = words + decoded.tolist()
                        result = nlp.vec_to_sentence(decoded, self.idx_to_words)
                    else:
                        result = nlp.vec_to_sentence(decoded[i], self.idx_to_words)
                hypotheses.append(result)
                references.append(original)

        if references_file_name:
            original_file = open("./results/" + references_file_name, "w")
            original_sentences_with_new_line = map(lambda x: x + "\n", references)
            original_file.writelines(original_sentences_with_new_line)
            original_file.close()

        hypotheses_file = open(hypotheses_file_name, "w")
        hypotheses_sentences_with_new_line = map(lambda x: x + "\n", hypotheses)
        hypotheses_file.writelines(hypotheses_sentences_with_new_line)
        hypotheses_file.close()

    def predict_all_beam_search(self, batch_size, beam_size=3, sentence_length=22, references_file_name='',
                                hypotheses_file_name=''):

        data_generator = ModelDataGenerator(self.dataset_file, self.vocab_json, batch_size)
        references = []
        hypotheses = []

        for batch in data_generator.multiple_samples_per_story_generator(reverse=False, only_one_epoch=True, last_k=3):
            encoder_batch_input_data = batch[0][0]
            original_sentences_input = batch[0][1]
            print(encoder_batch_input_data.shape)

            decoded = self.predict_story_beam_search(encoder_batch_input_data, beam_size=beam_size)
            for i in range(len(decoded[0])):
                max_score_index = np.argmin(decoded[1][i])
                # print("Decoded", nlp.vec_to_sentence(decoded[0][i][max_score_index], self.idx_to_words))
                hypotheses.append(nlp.vec_to_sentence(decoded[0][i][max_score_index], self.idx_to_words))

        if references_file_name:
            original_file = open("./results/" + references_file_name, "w")
            original_sentences_with_new_line = map(lambda x: x + "\n", references)
            original_file.writelines(original_sentences_with_new_line)
            original_file.close()

        hypotheses_file = open(hypotheses_file_name, "w")
        hypotheses_sentences_with_new_line = map(lambda x: x + "\n", hypotheses)
        hypotheses_file.writelines(hypotheses_sentences_with_new_line)
        hypotheses_file.close()

    def predict_batch_with_sentence_embed(self, input_sequence, encoder_sentence, sentence_embed_boolean,
                                          sentence_length, words, no_duplicates):

        input_sequence = input_sequence.reshape((1, input_sequence.shape[0], input_sequence.shape[1]))
        num_stories = input_sequence.shape[0]
        decoded_sentences = np.zeros((sentence_length), dtype='int32')

        if sentence_embed_boolean:
            states_value = self.encoder_model.predict([input_sequence, encoder_sentence])
        else:
            states_value = self.encoder_model.predict(input_sequence)

        states_value_shape = states_value.shape
        states_value = [states_value]
        for i in range(self.num_stacked_layers - 1):
            states_value.append(np.zeros(states_value_shape))

        target_seq = np.zeros((num_stories, 1), dtype='int32')
        target_seq[0:num_stories, 0] = self.words_to_idx["<START>"]

        stop_condition = False
        i = 0

        while not stop_condition:
            output = self.decoder_model.predict([target_seq] + states_value)
            output_tokens = output[0]

            sampled_word_index = np.argmax(output_tokens[:, 0, :], axis=1).astype(dtype='int32')
            if no_duplicates:
                if i > 0:
                    j = 1
                    while sampled_word_index in decoded_sentences or sampled_word_index in words:
                        sampled_word_index = np.argsort(output_tokens[0, 0, :])[-j]
                        j += 1

            if i >= sentence_length or sampled_word_index == 2:
                break

            decoded_sentences[i] = sampled_word_index
            target_seq[0:num_stories, 0] = sampled_word_index

            states_value = output[1:]
            i += 1

        return decoded_sentences

    def slow_beam_search_helper(self, live_sentence, state_values):
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

    def slow_beam_search(self, images, encoder_sentence, max_decoder_seq_length=22, images_per_story=5,
                                  image_embed_size=4096, beam_size=3):
        decoded_sentences = []
        scores = []
        m = 1

        images = images.reshape(1, 5, 4096)
        state_value = self.encoder_model.predict([images, encoder_sentence])
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
            for k in range(self.num_stacked_layers - 1):
                state_value_tmp.append(np.zeros(state_value_shape))

            state_values.append(state_value_tmp)

        j = 0

        while live_beam and dead_beam < beam_size:

            probs, new_states = self.slow_beam_search_helper(live_sentence, state_values)

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

        decoded_sentences = dead_sentence + live_sentence
        scores = dead_scores + live_score

        best_idx = np.argmin(scores)
        return decoded_sentences[best_idx]




    def get_number_of_sentences(self, live_sentences):
        num_sentences = 0
        for sentences in live_sentences:
            num_sentences += len(sentences)

        return num_sentences

    def check_dead_beams(self, dead_beam, beam_size):
        for i in range(len(dead_beam)):
            if dead_beam[i] < beam_size:
                return True
        return False

    def check_live_beams(self, live_beam):
        for i in range(len(live_beam)):
            if live_beam[i] > 0:
                return True
        return False


class NLPScores(Callback):
    def __init__(self, dataset_type):
        super(NLPScores, self).__init__()
        self.dataset_type = dataset_type

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        print("On Epoch End: ")
        print("epoch: ", epoch)

        t = time.time()
        builder = Seq2SeqBuilder()
        encoder_model, decoder_model = builder.build_encoder_decoder_inference(self.model)
        inference = Inference('./dataset/image_embeddings_to_sentence/stories_to_index_' + self.dataset_type + '.hdf5',
                              './dataset/vist2017_vocabulary.json', encoder_model, decoder_model)

        hypotheses_filename = "./results/temp/hypotheses_" + self.dataset_type + "_epoch_" + str(
            epoch) + ".txt"
        references_filename = "./results/original_" + self.dataset_type + ".txt"

        inference.predict_all(batch_size=64, references_file_name='',
                              hypotheses_file_name=hypotheses_filename)

        # calculating Meteor
        status, output_meteor = commands.getstatusoutput(
            "java -Xmx2G -jar nlp/meteor-1.5.jar " + hypotheses_filename + " " + references_filename + " -t hter -l en -norm")

        # Calculating BLEU score
        status, output_bleu = commands.getstatusoutput(
            "perl ./nlp/multi-bleu.perl " + references_filename + " < " + hypotheses_filename)

        text_file = open('./results/temp/' + "bleu_" + self.dataset_type + "_epoch_" + str(epoch), "w")
        text_file.write(output_bleu)
        text_file.close()

        text_file = open('./results/temp/' + "meteor_" + self.dataset_type + "_epoch_" + str(epoch), "w")
        text_file.write(output_meteor)
        text_file.close()

        print("Meteor/Bleu time(minutes) : ", (time.time() - t) / 60.0)
        print("end calculating")
