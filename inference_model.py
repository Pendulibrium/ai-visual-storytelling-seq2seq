from keras.optimizers import *
import numpy as np
import h5py
import json
from nlp.scores import Scores, Score_Method
import nlp.nlp as nlp
from seq2seqbuilder import Seq2SeqBuilder
from result_visualisation import Inference
import time as time

latent_dim = 1024
num_of_stacked_rnn = 2

builder = Seq2SeqBuilder()
encoder_model, decoder_model = builder.build_encoder_decoder_inference(
    "trained_models/2018-01-18_17:39:24-2018-01-20_18:50:39:image_to_text_gru.h5")

inference = Inference('./dataset/image_embeddings_to_sentence/stories_to_index_valid.hdf5',
                       './dataset/vist2017_vocabulary.json', encoder_model, decoder_model)
t=time.time()
inference.predict_all_beam_search(batch_size=5, beam_size=10)
print((time.time()-t)/60.0)
# vocab_json = json.load(open('./dataset/vist2017_vocabulary.json'))
# num_decoder_tokens = len(vocab_json['idx_to_words'])
# words_to_idx = vocab_json["words_to_idx"]
# idx_to_words = vocab_json["idx_to_words"]
#
# max_decoder_seq_length = 22

# def decode_sequence(input_seq, num_stories):
#     decoded_sentences = np.zeros((num_stories,22), dtype='int32')
#
#
#     #decoded_sentence = ''
#     states_value = encoder_model.predict(input_seq)
#     print("", states_value.shape)
#     # # print(states_value[0][0:20])
#     states_value = [states_value]
#
#     target_seq = np.zeros((num_stories, 22))
#     target_seq[0:num_stories, 0] = words_to_idx["<START>"]
#
#     stop_condition = False
#     i = 0
#
#     while not stop_condition:
#
#
#         output_tokens, h = decoder_model.predict([target_seq] + states_value)
#          #print(output_tokens.shape)
#         sampled_word_index = np.argmax(output_tokens[:,-1,:], axis=1)
#         print(sampled_word_index)
#     #     sampled_word = idx_to_words[sampled_word_index]
#     #
#     #    if i >= max_decoder_seq_length or sampled_word == "<END>":
#         if i >= max_decoder_seq_length:
#              break
#         decoded_sentences[:,i] = sampled_word_index.astype(dtype='int32')
#         target_seq = np.zeros((num_stories, 22))
#         target_seq[0:num_stories, 0] = sampled_word_index.astype(dtype='int32')
#         states_value = [h]
#         i += 1
#
#     return decoded_sentences
#
#
# def calc_total_score(file_name):
#     test_file = h5py.File(file_name, 'r')
#     image_embeddings = test_file["image_embeddings"]
#     story_sentences = test_file["story_sentences"]
#     bleu_total = 0.0
#     meteor_total = 0.0
#     for i in range(len(story_sentences)):
#         print(str(i) + ":" + str(bleu_total))
#         input_images = image_embeddings[i]
#         input_senteces = story_sentences[i]
#         encoder_batch_input_data = np.zeros((5, 1, 4096))
#
#         for j in range(5):
#             encoder_batch_input_data[j, 0] = input_images[j]
#
#         original_sentences = []
#
#         for story in input_senteces:
#             st = ''
#             for word in story:
#                 if not (idx_to_words[word] == "<START>" or idx_to_words[word] == "<END>" or idx_to_words[
#                     word] == "<NULL>"):
#                     st += idx_to_words[word] + " "
#
#             original_sentences.append(st)
#
#         print("before")
#         decoded = decode_sequence(encoder_batch_input_data)
#         print("after")
#
#         sentence_scores = Scores()
#
#         _, bleu_score = sentence_scores.calculate_scores(Score_Method.BLEU, decoded, original_sentences)
#
#         # print("Bleu", bleu_scores[1])
#         bleu_total += bleu_score
#         # _, meteor_score = sentence_scores.calculate_scores(Score_Method.METEOR, decoded, original_sentences)
#         # meteor_total += meteor_score
#         # print("Meteor", meteor_scores[1])
#
#     print('Bleu: ', bleu_total)
#     print("Meteor: ", meteor_total)
#
#
# train_file = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_valid.hdf5', 'r')
# story_ids = train_file["story_ids"]
# image_embeddings = train_file["image_embeddings"]
# story_sentences = train_file["story_sentences"]
#
# #calc_total_score('./dataset/image_embeddings_to_sentence/stories_to_index_train.hdf5')
#
# t = time.time()
# num_stories = 10
# random_sample_index = 0
# input_id = story_ids[random_sample_index:random_sample_index+num_stories]
# input_images = image_embeddings[random_sample_index:random_sample_index+num_stories]
#
# input_senteces = story_sentences[random_sample_index:random_sample_index+num_stories]
# print(input_id)
#
# encoder_batch_input_data = np.zeros((5*len(input_id), 5, 4096))
#
# print("input_images shape: ", input_images.shape)
# for k in range(len(input_id)):
#     for j in range(5):
#         print("k:",k)
#         print("j:",j)
#         encoder_batch_input_data[(k*j)+j:(k*j)+5, j] = input_images[k][j]
#
#
# original_sentences = []
#
# for j in range(len(input_id)):
#     for story in input_senteces[j]:
#         st=''
#         for word in story:
#             if not (idx_to_words[word] == "<START>" or idx_to_words[word] == "<END>" or idx_to_words[word]=="<NULL>"):
#                 st += idx_to_words[word] + " "
#
#         original_sentences.append(st)
#
#
# decoded = decode_sequence(encoder_batch_input_data, 5*num_stories)
# #print(len(decoded))
# for i in range(5*len(input_id)):
#     #score = sentence_bleu([original_sentences[i]],decoded[i])
#     #print(i)
#     print("Original", original_sentences[i])
#     print("Decoded", nlp.vec_to_sentence(decoded[i],idx_to_words))
#     #print(score)

# print((time.time()-t)/60.0)

# def decode_sequence(input_seq):
#     decoded_sentences = []
#
#     for images in input_seq:
#
#         images = images.reshape((1, 5, 4096))
#
#         decoded_sentence = ''
#         states_value = encoder_model.predict(images)
#         states_value = [states_value]
#
#         target_seq = np.zeros((1, 22))
#         target_seq[0, 0] = words_to_idx["<START>"]
#
#         stop_condition = False
#         i = 0
#
#         while not stop_condition:
#             i += 1
#
#             output_tokens, h = decoder_model.predict([target_seq] + states_value)
#             sampled_word_index = np.argmax(output_tokens[0, -1, :])
#             #print(sorted(output_tokens[0,0,:])[0:10])
#             sampled_word = idx_to_words[sampled_word_index]
#
#             if i >= max_decoder_seq_length or sampled_word == "<END>":
#                 break
#             decoded_sentence += sampled_word + " "
#             target_seq = np.zeros((1, 22))
#             target_seq[0, 0] = sampled_word_index
#             states_value = [h]
#         decoded_sentences.append(decoded_sentence)
#
#     return decoded_sentences
#
#
# train_file = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_train.hdf5', 'r')
# story_ids = train_file["story_ids"]
# image_embeddings = train_file["image_embeddings"]
# story_sentences = train_file["story_sentences"]
#
# t = time.time()
# for i in range(0,10):
#     random_sample_index=i
#     input_id = story_ids[random_sample_index]
#     input_images = image_embeddings[random_sample_index]
#
#     input_senteces = story_sentences[random_sample_index]
#     print(input_id)
#
#     encoder_batch_input_data = np.zeros((5, 5, 4096))
#
#     print("input_images shape: ", input_images.shape)
#     for j in range(5):
#         encoder_batch_input_data[j:5, j] = input_images[j]
#
#
#     original_sentences = []
#
#     for story in input_senteces:
#         st=''
#         for word in story:
#             if not (idx_to_words[word] == "<START>" or idx_to_words[word] == "<END>" or idx_to_words[word]=="<NULL>"):
#                 st += idx_to_words[word] + " "
#
#         original_sentences.append(st)
#
#
#     decoded = decode_sequence(encoder_batch_input_data)
#     for i in range(5):
#         print("Original", original_sentences[i])
#         print("Decoded", decoded[i])
#
# print((time.time()-t)/60)
