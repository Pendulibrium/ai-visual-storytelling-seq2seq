import json
import numpy as np
import keras


def vec_to_sentence(sentence_vec, idx_to_word):
    """ Return human readable sentence of given sentence vector

    Parameters

    """
    words = []
    for word_idx in sentence_vec:
        word = idx_to_word[word_idx]

        if word == "<START>":
            continue
        if word == "<END>" or word == "<NULL>":
            break

        words.append(word)

    return " ".join(words)


def one_hot_vec_to_sentence(sentence_vec, idx_to_word):
    """ Return human readable sentence of given sentence one hot vector

    Parameters

    """

    words = []

    for one_hot_word_vec in sentence_vec:

        one_hot_where = np.where(one_hot_word_vec == 1)[0]
        if len(one_hot_where) > 0:
            word_idx = one_hot_where[0]
            words.append(idx_to_word[word_idx])

    return " ".join(words)


class Bleu_Score_Callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        return
    def on_train_end(self, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
     #self.losses.append(logs.get('loss'))
     validation_data = self.model.validation_data[0]
     print(validation_data.shape)
     #print("Hi")

     return