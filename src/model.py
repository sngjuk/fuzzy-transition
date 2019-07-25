import fasttext
import numpy as np


class Model:
    def __init__(self, model_path=None, ft_matrix=None):
        if not model_path:
            model_path = "/Users/user/Desktop/fuzzy-transition/nlp/fastText-0.9.1/model/namu_mecab_400.bin"
        model = fasttext.load_model(model_path)

        self.model = model
        self.ft_words = model.get_words()
        self.word_frequencies = dict(zip(*model.get_words(include_freq=True)))
        self.ft_matrix = ft_matrix
        if self.ft_matrix is None:
            self.ft_matrix = np.empty((len(self.ft_words), model.get_dimension()))
            for i, word in enumerate(self.ft_words):
                self.ft_matrix[i, :] = model.get_word_vector(word)

    # return type: [embed_vector]
    def get_word_vector(self, word):
        return self.model.get_word_vector(word)

    @staticmethod
    def find_nearest_neighbor(query, vectors, n=10,  cossims=None):

        if cossims is None:
            cossims = np.matmul(vectors, query, out=cossims)

        norms = np.sqrt((query**2).sum() * (vectors**2).sum(axis=1))
        cossims = cossims/norms
        result_i = np.argpartition(-cossims, range(n+1))[1:n+1]
        return list(zip(result_i, cossims[result_i]))

    # return type: [('name', similarity), ... ]
    def nearest_words(self, word, n=30, word_freq=None):
        result = self.find_nearest_neighbor(self.model.get_word_vector(word), self.ft_matrix, n=n)
        if word_freq:
            return [(self.ft_words[r[0]], r[1]) for r in result if self.word_frequencies[self.ft_words[r[0]]] >= word_freq]
        else:
            return [(self.ft_words[r[0]], r[1]) for r in result]
