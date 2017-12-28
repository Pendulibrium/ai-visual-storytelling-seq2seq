import json
import numpy as np


def vec_to_sentence(sentence_vec, idx_to_word):
    """ Return human readable sentence of given sentence vector

    Parameters

    """
    words = []
    for word_idx in sentence_vec:
        words.append(idx_to_word[word_idx])

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
