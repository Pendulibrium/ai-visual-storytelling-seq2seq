import json
import numpy as np
import operator
from unidecode import unidecode

class SIS_DataReader:

    def __init__(self, path_to_file='../dataset/vist_sis/train.story-in-sequence.json'):
        self.path_to_file = path_to_file

    def create_word_frequency_document(self, path_to_json_file='../dataset/word_frequencies.json'):

        data = json.load(open(self.path_to_file))
        annotations = data['annotations']

        frequency = {}
        for annotation in annotations:
            sentence = annotation[0]['text'].split()
            for word in sentence:
                #proverka za brishenje na greski so zborovi vo unicode format(latinski zborovi)
                    if any(x.isupper() for x in unidecode(word)) == False:
                        count = frequency.get(word,0)
                        frequency[word] = count + 1

        sorted_frequency = sorted(frequency.items(),key=operator.itemgetter(1),reverse=True)

        with open(path_to_json_file, 'w') as fp:
            json.dump(sorted_frequency, fp)

    def get_n_most_frequent_words(self, word_frequency_file='../dataset/word_frequencies.json', vocabulary_size=10000):
        data = json.load(open(word_frequency_file))
        return data[0:vocabulary_size]

    def generate_vocabulary(self,vocabulary_file='../dataset/vist2017_vocabulary.json', word_frequency_file='../dataset/word_frequencies.json', vocabulary_size=10000):
        data = self.get_n_most_frequent_words(word_frequency_file,vocabulary_size)
        idx_to_words = []
        idx_to_words.append("<NULL>")
        idx_to_words.append("<START>")
        idx_to_words.append("<END>")
        idx_to_words.append("<UNK>")

        for element in data:

            idx_to_words.append(element[0])

        words_to_idx={}
        for i in range(len(idx_to_words)):
            words_to_idx[idx_to_words[i]]=i

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
            if(length > max_sentence_length):
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
            if(length < min_sentence_length):
                min_sentence_length = length
                save_sent = sentence

        print(save_sent)
        return min_sentence_length

    def sentences_to_index(self,vocabulary_file='../dataset/vist2017_vocabulary.json'):
        #treba da se dopolni koga kje znaeme sto da pravime so maksimumot
        vocabulary=json.load(open(vocabulary_file))
        words_to_idx=vocabulary["words_to_idx"]

        data=json.load(open(self.path_to_file))
        annotations=data["annotations"]

        for i in range(len(annotations)):
            print(annotations[i][0]["story_id"])




object=SIS_DataReader()
object.sentences_to_index()

# data = json.load(open('../dataset/word_frequencies.json'))
# print(type(data))
# print(len(data))
# print(type(data[0]))
# for i in range(1000,2000):
#     print(data[i])