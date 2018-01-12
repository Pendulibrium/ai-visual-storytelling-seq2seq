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

    def multiple_samples_per_story_generator(self, reverse=False):
        print("generate image sequences from story")

        story_batch_size = int(np.round(self.batch_size / float(self.story_length)))  # Number of stories
        approximate_batch_size = story_batch_size * self.story_length  # Actual batch size

        while 1:
            encoder_batch_input_data = np.zeros((approximate_batch_size, self.story_length, self.image_embeddings_size))
            decoder_batch_input_data = np.zeros((approximate_batch_size, self.sentences_length), dtype=np.int32)
            decoder_batch_target_data = np.zeros((approximate_batch_size, self.sentences_length, self.number_of_tokens),
                                                 dtype=np.int32)

            for i in range(self.num_samples):
                for j in range(self.story_length):
                    encoder_row_start_range = ((i % story_batch_size) * self.story_length) + j
                    encoder_row_end_range = ((i % story_batch_size) * self.story_length) + self.story_length
                    encoder_batch_input_data[encoder_row_start_range: encoder_row_end_range, j] = \
                        self.image_embeddings[i][j]

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

                if ((i + 1) % story_batch_size) == 0 and i != 0:
                    yield ([encoder_batch_input_data, decoder_batch_input_data], decoder_batch_target_data)

                    encoder_batch_input_data.fill(0.0)
                    decoder_batch_input_data.fill(0)
                    decoder_batch_target_data.fill(0)

    '''
        Generate only one sample per story, all images included in the sample.
    '''

    def one_sample_from_story_generator(self, reverse=False, concatenate_all_sentences=False):
        print("one sample from story")

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

                if ((i + 1) % story_batch_size) == 0 and i != 0:
                    yield ([encoder_batch_input_data, decoder_batch_input_data], decoder_batch_target_data)

                    encoder_batch_input_data.fill(0.0)
                    decoder_batch_input_data.fill(0)
                    decoder_batch_target_data.fill(0)

# vocab_json = json.load(open('./dataset/vist2017_vocabulary.json'))
# train_dataset = h5py.File('./dataset/image_embeddings_to_sentence/stories_to_index_train.hdf5', 'r')
# model_generator = ModelDataGenerator(train_dataset, vocab_json, 64)
# generator = model_generator.image_caption_generator()

# batch_data = generator.next()



# image_embeddings = batch_data[0][0]
# decoder_input = batch_data[0][1]
# decoder_output = batch_data[1]
#
# image_embeddings_dataset = train_dataset['image_embeddings']
#
# for i in range(len(image_embeddings)):
#     print(np.array_equal(image_embeddings[i][0], image_embeddings_dataset[i / 5 + 14][i % 5]))
