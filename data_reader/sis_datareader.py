import json
import numpy as np
import operator
from unidecode import unidecode
import h5py
import os
import glob


class SIS_DataReader:

    def __init__(self, path_to_file='../dataset/vist_dataset/training_data/sis/train.story-in-sequence.json'):
        self.path_to_file = path_to_file

    def create_word_frequency_document(self, path_to_json_file='../dataset/word_frequencies.json'):

        data = json.load(open(self.path_to_file))
        annotations = data['annotations']

        frequency = {}
        for annotation in annotations:
            sentence = annotation[0]['text'].split()
            for word in sentence:
                # proverka za brishenje na greski so zborovi vo unicode format(latinski zborovi)
                if any(x.isupper() for x in unidecode(word)) == False:
                    count = frequency.get(word, 0)
                    frequency[word] = count + 1

        sorted_frequency = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)

        with open(path_to_json_file, 'w') as fp:
            json.dump(sorted_frequency, fp)

    def get_n_most_frequent_words(self, word_frequency_file='../dataset/word_frequencies.json', vocabulary_size=10000):

        data = json.load(open(word_frequency_file))
        return data[0:vocabulary_size]

    def generate_vocabulary(self, vocabulary_file='../dataset/vist2017_vocabulary.json',
                            word_frequency_file='../dataset/word_frequencies.json', vocabulary_size=10000):

        data = self.get_n_most_frequent_words(word_frequency_file, vocabulary_size)

        idx_to_words = []
        idx_to_words.append("<NULL>")
        idx_to_words.append("<START>")
        idx_to_words.append("<END>")
        idx_to_words.append("<UNK>")

        for element in data:
            idx_to_words.append(element[0])

        words_to_idx = {}
        for i in range(len(idx_to_words)):
            words_to_idx[idx_to_words[i]] = i

        vocabulary = {}
        vocabulary["idx_to_words"] = idx_to_words
        vocabulary["words_to_idx"] = words_to_idx

        with open(vocabulary_file, 'w') as fp:
            json.dump(vocabulary, fp)

    def get_max_sentence_length(self):

        data = json.load(open(self.path_to_file))
        annotations = data['annotations']
        save_sent = []
        max_sentence_length = 0

        for annotation in annotations:
            sentence = annotation[0]['text'].split()
            length = len(sentence)
            if (length > max_sentence_length):
                max_sentence_length = length
                save_sent = sentence

        print(save_sent)
        return max_sentence_length

    def get_min_sentence_length(self):

        data = json.load(open(self.path_to_file))
        annotations = data['annotations']
        save_sent = []
        min_sentence_length = 100

        for annotation in annotations:
            sentence = annotation[0]['text'].split()
            length = len(sentence)
            if (length < min_sentence_length):
                min_sentence_length = length
                save_sent = sentence

        print(save_sent)
        return min_sentence_length

    def sentences_to_index(self, vocabulary_file='../dataset/vist2017_vocabulary.json',
                           image_embedding_file="../dataset/models/alexnet/alexnet_image_train_features.hdf5",
                           save_file_path='../dataset/image_embeddings_to_sentence/stories_to_index_train.hdf5',
                           images_directory='../dataset/vist_dataset/training_data/train_img',
                           max_length=20):

        vocabulary = json.load(open(vocabulary_file))
        words_to_idx = vocabulary["words_to_idx"]

        data = json.load(open(self.path_to_file))
        annotations = data["annotations"]

        img_hash = self.get_image_features_hash(image_embedding_file)

        images_path_names = [y for x in os.walk(images_directory) for y in glob(os.path.join(x[0], "*.jpg"))]
        images_path_names_png = [y for x in os.walk(images_directory) for y in glob(os.path.join(x[0], "*.png"))]
        images_path_names = np.append(images_path_names, images_path_names_png)

        story_ids = []
        story_sentences = []
        story_images = []
        story_images_ids = []
        story_images_paths = []

        for i in range(0, len(annotations), 5):

            story_id = annotations[i][0]["story_id"]

            img_id1, order1 = int(annotations[i][0]["photo_flickr_id"]), annotations[i][0][
                "worker_arranged_photo_order"]
            img_id2, order2 = int(annotations[i + 1][0]["photo_flickr_id"]), annotations[i + 1][0][
                "worker_arranged_photo_order"]
            img_id3, order3 = int(annotations[i + 2][0]["photo_flickr_id"]), annotations[i + 2][0][
                "worker_arranged_photo_order"]
            img_id4, order4 = int(annotations[i + 3][0]["photo_flickr_id"]), annotations[i + 3][0][
                "worker_arranged_photo_order"]
            img_id5, order5 = int(annotations[i + 4][0]["photo_flickr_id"]), annotations[i + 4][0][
                "worker_arranged_photo_order"]

            if not (img_hash.has_key(str(img_id1))):
                image1=np.zeros(4096)
            else:
                image1=img_hash[str(img_id1)]

            if not (img_hash.has_key(str(img_id2))):
                image2 = np.zeros(4096)
            else:
                image2 = img_hash[str(img_id2)]

            if not (img_hash.has_key(str(img_id3))):
                image3 = np.zeros(4096)
            else:
                image3 = img_hash[str(img_id3)]

            if not (img_hash.has_key(str(img_id4))):
                image4 = np.zeros(4096)
            else:
                image4 = img_hash[str(img_id4)]

            if not (img_hash.has_key(str(img_id5))):
                image5 = np.zeros(4096)
            else:
                image5 = img_hash[str(img_id5)]

            story1 = self.sentences_to_index_helper(annotations[i][0]["text"], words_to_idx, max_length)
            story2 = self.sentences_to_index_helper(annotations[i + 1][0]["text"], words_to_idx, max_length)
            story3 = self.sentences_to_index_helper(annotations[i + 2][0]["text"], words_to_idx, max_length)
            story4 = self.sentences_to_index_helper(annotations[i + 3][0]["text"], words_to_idx, max_length)
            story5 = self.sentences_to_index_helper(annotations[i + 4][0]["text"], words_to_idx, max_length)

            order1 = annotations[i][0]["worker_arranged_photo_order"]
            order2 = annotations[i + 1][0]["worker_arranged_photo_order"]
            order3 = annotations[i + 2][0]["worker_arranged_photo_order"]
            order4 = annotations[i + 3][0]["worker_arranged_photo_order"]
            order5 = annotations[i + 4][0]["worker_arranged_photo_order"]

            story_list = [(story1, order1), (story2, order2), (story3, order3), (story4, order4), (story5, order5)]
            story_list = sorted(story_list, key=operator.itemgetter(1))

            image_list = [(image1, order1), (image2, order2), (image3, order3),(image4, order4), (image5, order5)]
            image_list = sorted(image_list, key=operator.itemgetter(1))

            ordered_stories = [story_list[0][0], story_list[1][0], story_list[2][0], story_list[3][0], story_list[4][0]]
            ordered_images = [image_list[0][0], image_list[1][0], image_list[2][0], image_list[3][0], image_list[4][0]]
            ordered_image_ids = [img_id1, img_id2, img_id3, img_id4, img_id5]

            ordered_image_path_names = []
            for file_idx in range(len(images_path_names)):
                if images_path_names[file_idx].find(str(img_id1)):
                    ordered_image_path_names.append(images_path_names[file_idx])
                elif images_path_names[file_idx].find(str(img_id2)):
                    ordered_image_path_names.append(images_path_names[file_idx])
                elif images_path_names[file_idx].find(str(img_id3)):
                    ordered_image_path_names.append(images_path_names[file_idx])
                elif images_path_names[file_idx].find(str(img_id4)):
                    ordered_image_path_names.append(images_path_names[file_idx])
                elif images_path_names[file_idx].find(str(img_id5)):
                    ordered_image_path_names.append(images_path_names[file_idx])
                else:
                    ordered_image_path_names.append("None")

            story_ids.append(int(story_id))
            story_sentences.append(ordered_stories)
            story_images.append(ordered_images)
            story_images_ids.append(ordered_image_ids)
            story_images_paths.append(ordered_image_path_names)

        data_file = h5py.File(save_file_path, 'w')
        data_file.create_dataset("story_ids", data = story_ids)
        data_file.create_dataset("story_sentences", data = story_sentences)
        data_file.create_dataset("image_embeddings", data = story_images)
        data_file.create_dataset("image_ids", data = story_images_ids)
        data_file.create_dataset("image_paths", data = story_images_paths)

    def sentences_to_index_helper(self, sentence, word_to_idx, max_length):
        words = sentence.split()
        result_sentence = []

        for word in words:
            if len(result_sentence) == max_length:
                break
            else:
                if (word_to_idx.has_key(word)):
                    result_sentence.append(word_to_idx[word])
                else:
                    result_sentence.append(word_to_idx["<UNK>"])

        result_sentence.insert(0, word_to_idx["<START>"])
        result_sentence.append(word_to_idx["<END>"])

        while len(result_sentence) < max_length + 2:
            result_sentence.append(word_to_idx["<NULL>"])

        return result_sentence

    def indecies_to_sentence(self, sentence, idx_to_word):

        result_sentence = ""
        for word in sentence:
            if word == 0 :
                result_sentence = result_sentence + " " + idx_to_word[word]

        print(result_sentence)
        return result_sentence

    def get_image_features_hash(self,file_name):

        image_features_file = h5py.File(file_name, 'r')
        image_features_ids = image_features_file["image_ids"]
        image_embeddings = image_features_file["embeddings"]
        dictionary = {}

        for id, em in zip(image_features_ids,image_embeddings):
            dictionary[str(id)] = em

        return dictionary


object = SIS_DataReader()
object.sentences_to_index()
# #
# data_file=h5py.File('../dataset/stories_to_index.hdf5','r')
# story_id=data_file["story_ids"]
# sentence=data_file['story_sentences']
# embed=data_file['image_embeddings']
