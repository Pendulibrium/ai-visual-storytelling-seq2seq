import numpy as np
import json
import h5py
from nlp import nlp


class ModelDataGenerator:
    def __init__(self, dataset, vocab_json, batch_size, num_samples_per_epoch=None):

        self.dataset = dataset
        self.vocab_json = vocab_json
        self.batch_size = batch_size

        # Shape: (num_samples, number_of_images_in_sample, image_embedding_length)
        self.image_embeddings = self.dataset["image_embeddings"]

        # Shape: (num_samples, number_of_sentences_in_sample, number_of_words_in_sentence)
        self.story_sentences = self.dataset["story_sentences"]

        self.story_length = self.story_sentences.shape[1]
        self.image_embeddings_size = self.image_embeddings.shape[2]
        self.sentences_length = self.story_sentences.shape[2]

        self.num_samples = self.image_embeddings.shape[0]

        if num_samples_per_epoch is not None:
            self.num_samples = num_samples_per_epoch

        # Number of unique words in the vocabulary
        self.number_of_tokens = len(self.vocab_json["idx_to_words"])

    '''
        Generate multiple samples from one story. If the story has 5 images then we will generate 5 samples e.g
        1 sample only with one image and the first sentence, 2 sample with 2 images and the second sentence, etc.
        It will approximate the batch size, due to the fact that we are generating 5 samples from one story, e.g
        if you sent batch_size 64 it will generate the batch size of 65.
    '''

    def generate_story_samples_from_index(self, story_index, reverse=False, last_k=5, sentence_embedding=True):

        encoder_batch_input_data = np.zeros(
            (self.story_length, self.story_length, self.image_embeddings_size))
        text_encoder_batch_input_data = np.zeros((self.story_length, self.sentences_length), dtype=np.int32)
        decoder_batch_input_data = np.zeros((self.story_length, self.sentences_length), dtype=np.int32)
        decoder_batch_target_data = np.zeros(
            (self.story_length, self.sentences_length, self.number_of_tokens),
            dtype=np.int32)

        for j in range(self.story_length):

            encoder_batch_input_data[j:min(j + last_k, self.story_length), j] = self.image_embeddings[story_index][j]

            if reverse:
                encoder_batch_input_data[j] = np.flip(encoder_batch_input_data[j], axis=0)

            # TODO: this should be optimized in the database instead of in the generating process

            if j == 0:
                empty_sentece = np.zeros((self.sentences_length))
                empty_sentece[0] = 1
                text_encoder_batch_input_data[j] = empty_sentece
            else:
                text_encoder_batch_input_data[j] = self.story_sentences[story_index][j - 1]

            temp_story = self.story_sentences[story_index][j].tolist()
            end_index = temp_story.index(2)
            temp_story[end_index] = 0
            decoder_batch_input_data[j] = np.array(temp_story)

            sentence = self.story_sentences[story_index][j]
            for word_index in range(len(sentence)):
                if word_index > 0:
                    decoder_batch_target_data[j, word_index - 1, sentence[word_index]] = 1

        if sentence_embedding:
            return encoder_batch_input_data, text_encoder_batch_input_data, decoder_batch_input_data, decoder_batch_target_data
        else:
            return encoder_batch_input_data, decoder_batch_input_data, decoder_batch_target_data

    def multiple_samples_per_story_generator(self, reverse=False, only_one_epoch=False, shuffle=False, last_k=5,
                                             sentence_embedding=True):

        story_batch_size = int(np.round(self.batch_size / float(self.story_length)))  # Number of stories

        while 1:
            if shuffle:
                permutation = np.random.permutation(self.num_samples)
            else:
                permutation = range(self.num_samples)

            for i in range(0, self.num_samples, story_batch_size):

                batch_stories_indicies = permutation[i:i + story_batch_size]
                number_of_stories_in_batch = len(batch_stories_indicies)
                approximate_batch_size = number_of_stories_in_batch * self.story_length  # Actual batch size

                encoder_batch_input_data = np.zeros(
                    (approximate_batch_size, self.story_length, self.image_embeddings_size))
                text_encoder_batch_input_data = np.zeros((approximate_batch_size, self.sentences_length),
                                                         dtype=np.int32)
                decoder_batch_input_data = np.zeros((approximate_batch_size, self.sentences_length), dtype=np.int32)
                decoder_batch_target_data = np.zeros(
                    (approximate_batch_size, self.sentences_length, self.number_of_tokens),
                    dtype=np.int32)

                for idx, story_index in enumerate(batch_stories_indicies):
                    story_samples = self.generate_story_samples_from_index(story_index, reverse, last_k, sentence_embedding)
                    start = idx * self.story_length
                    end = start + self.story_length
                    encoder_batch_input_data[start: end] = story_samples[0]
                    text_encoder_batch_input_data[start: end] = story_samples[1]
                    decoder_batch_input_data[start: end] = story_samples[2]
                    decoder_batch_target_data[start: end] = story_samples[3]

                if sentence_embedding:
                    # print(text_encoder_batch_input_data.shape)
                    # print(text_encoder_batch_input_data[0])
                    # print(decoder_batch_input_data[0])
                    # print(text_encoder_batch_input_data[1])
                    yield ([encoder_batch_input_data, text_encoder_batch_input_data, decoder_batch_input_data], decoder_batch_target_data)
                else:
                    yield ([encoder_batch_input_data, decoder_batch_input_data], decoder_batch_target_data)

            if only_one_epoch:
                raise StopIteration()

    '''
        Generate only one sample per story, all images included in the sample.
    '''

    def one_sample_from_story_generator(self, reverse=False, concatenate_all_sentences=False):
        print("one sample from story")
        while 1:

            encoder_batch_input_data = np.zeros((self.batch_size, self.story_length, self.image_embeddings_size))
            decoder_batch_input_data = np.zeros((self.batch_size, self.sentences_length), dtype=np.int32)
            decoder_batch_target_data = np.zeros((self.batch_size, self.sentences_length, self.number_of_tokens),
                                                 dtype=np.int32)

            for i in range(self.num_samples):
                if reverse:
                    encoder_batch_input_data[i % self.batch_size] = np.flip(self.image_embeddings[i], axis=0)
                else:
                    encoder_batch_input_data[i % self.batch_size] = self.image_embeddings[i]

                # TODO: we are getting only the first sentence for now, we should concatenate all the sentence if
                # concatenate_all_sentences is True
                temp_story = self.story_sentences[i][0].tolist()
                end_index = temp_story.index(2)
                temp_story[end_index] = 0
                decoder_batch_input_data[i % self.batch_size] = np.array(temp_story)

                sentence = self.story_sentences[i][0]

                for word_index in range(len(sentence)):
                    if word_index > 0:
                        decoder_batch_target_data[i % self.batch_size, word_index - 1, sentence[word_index]] = 1

                if (i + 1) % self.batch_size == 0:
                    yield ([encoder_batch_input_data, decoder_batch_input_data], decoder_batch_target_data)

                    encoder_batch_input_data = np.zeros(
                        (self.batch_size, self.story_length, self.image_embeddings_size))
                    decoder_batch_input_data = np.zeros((self.batch_size, self.sentences_length), dtype=np.int32)
                    decoder_batch_target_data = np.zeros(
                        (self.batch_size, self.sentences_length, self.number_of_tokens),
                        dtype=np.int32)

    '''
        Generate multiple samples from one story with only one image, mapping the image to the descirption of the image
    '''

    def image_caption_generator(self):
        while 1:

            story_batch_size = int(np.round(self.batch_size / float(self.story_length)))  # Number of stories
            approximate_batch_size = story_batch_size * self.story_length  # Actual batch size

            encoder_batch_input_data = np.zeros((approximate_batch_size, 1, self.image_embeddings_size))
            decoder_batch_input_data = np.zeros((approximate_batch_size, self.sentences_length), dtype=np.int32)
            decoder_batch_target_data = np.zeros((approximate_batch_size, self.sentences_length, self.number_of_tokens),
                                                 dtype=np.int32)
            for i in range(self.num_samples):
                for j in range(self.story_length):
                    encoder_row_start_range = ((i % story_batch_size) * self.story_length) + j

                    encoder_batch_input_data[encoder_row_start_range, 0] = self.image_embeddings[i][j]

                    decoder_row = (i % story_batch_size) * self.story_length + j

                    temp_story = self.story_sentences[i][j].tolist()
                    end_index = temp_story.index(2)
                    temp_story[end_index] = 0
                    decoder_batch_input_data[decoder_row] = np.array(temp_story)

                story = self.story_sentences[i]
                for sentence_index in range(len(story)):
                    sentence = story[sentence_index]
                    for word_index in range(len(sentence)):
                        if word_index > 0:
                            decoder_row = ((i % story_batch_size) * self.story_length) + sentence_index
                            decoder_batch_target_data[decoder_row, word_index - 1, sentence[word_index]] = 1

                if (i + 1) % story_batch_size == 0:
                    yield ([encoder_batch_input_data, decoder_batch_input_data], decoder_batch_target_data)

                    encoder_batch_input_data = np.zeros((approximate_batch_size, 1, self.image_embeddings_size))
                    decoder_batch_input_data = np.zeros((approximate_batch_size, self.sentences_length), dtype=np.int32)
                    decoder_batch_target_data = np.zeros(
                        (approximate_batch_size, self.sentences_length, self.number_of_tokens),
                        dtype=np.int32)
