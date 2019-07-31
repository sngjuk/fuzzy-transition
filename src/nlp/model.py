import re

import fasttext
import numpy as np
from scipy.spatial import distance


class EmbedModel:
    def __init__(self, model_path=None, ft_matrix=None):
        self.model = None
        self.load_model(model_path)

        # below lines are merely for nearest_words()
        self.ft_words = self.get_words()
        self.word_frequencies = dict(zip(*self.get_words(include_freq=True)))
        self.ft_matrix = ft_matrix
        if self.ft_matrix is None:
            self.ft_matrix = np.empty((len(self.ft_words), self.get_dimension()))
            for i, word in enumerate(self.ft_words):
                self.ft_matrix[i, :] = self.get_word_vector(word)

    # >> need to overwrite <<
    def load_model(self, model_path):
        self.model = fasttext.load_model(model_path)

    # >> need to overwrite <<
    # // return type: List : e.g.) [ embedding_vector, ... ], sentence is also supported
    def get_word_vector(self, word):
        # sentence
        if len(word.split()) > 1:
            return self.model.get_sentence_vector(word)
        # word
        else:
            return self.model.get_word_vector(word)

    # >> need to overwrite <<
    # // return type: List of Tuples: e.g.) [('word', similarity), ... ]
    def nearest_words(self, word, n=30, word_freq=None):
        result = self.find_nearest_neighbor(self.model.get_word_vector(word), self.ft_matrix, n=n)
        if word_freq:
            return [(self.ft_words[r[0]], r[1]) for r in result if self.word_frequencies[self.ft_words[r[0]]] >= word_freq]
        else:
            return [(self.ft_words[r[0]], r[1]) for r in result]

    # skip name contained vectors
    def filtered_nearest_neighbor(self, name, num=30, sim_threshold=0.39):
        result = self.nearest_words(name, num)
        res = []

        for i in result:
            my_regex = r".*" + re.escape(name) + r".*"
            if re.search(my_regex, i[0]):
                continue
            elif i[1] > sim_threshold:
                res.append(i)
        return res

    def get_similarity(self, word1, word2):
        return 1 - distance.cosine(self.get_word_vector(word1), self.get_word_vector(word2))

    # // return type: Int : e.g.) 100
    def get_dimension(self):
        return self.model.get_dimension()

    # // return type: List : e.g.) ['word', ... ]
    # (include_freq=True):
    # Tuple of List and numpy.ndarray : (['word', 'word2', ... ], array([9886, 5953, 4888, ..., 5, 5, 5])))
    def get_words(self, include_freq=False):
        return self.model.get_words(include_freq=include_freq)

    @staticmethod
    def find_nearest_neighbor(query, vectors, n=10,  cossims=None):
        if cossims is None:
            cossims = np.matmul(vectors, query, out=cossims)

        norms = np.sqrt((query**2).sum() * (vectors**2).sum(axis=1))
        cossims = cossims/norms
        result_i = np.argpartition(-cossims, range(n+1))[1:n+1]
        return list(zip(result_i, cossims[result_i]))
