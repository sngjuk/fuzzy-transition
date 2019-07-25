from collections import OrderedDict


class Node:
    name = None
    _vector = None
    _core_belief = None
    _implication = None

    def __init__(self, name, vector):
        self.name = name
        self.vector = vector
        self.core_belief = OrderedDict()
        self.implication = OrderedDict()

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, value):
        self._vector = value

    @property
    def core_belief(self):
        return self._core_belief

    @core_belief.setter
    def core_belief(self, value):
        self._core_belief = value

    # implication; OrderedDict() of list, { name : [vector, probability, count] }
    @property
    def implication(self):
        return self._implication

    @implication.setter
    def implication(self, value):
        self._implication = value

    # source, target, target_vector, probability, core
    def add_implication(self, target, vector, probability, core=None):
        if core:
            self._implication[target] = [vector, probability, 1, core]
            self._core_belief[target] = [vector, probability, 1, core]
            return self._implication[target][1], self._implication[target][2]

        if target not in self._implication:
            self._implication[target] = [vector, probability, 1, core]

        else:
            target_implication = self._implication[target]
            adjusted_probability = \
                (target_implication[1] * target_implication[2] + probability) / (target_implication[2] + 1)

            self._implication[target][1] = adjusted_probability
            self._implication[target][2] += 1

        return self._implication[target][1], self._implication[target][2]

    def sort_implication(self):
        self._implication = OrderedDict(sorted(self._implication.items(), key=lambda x: x[1][1], reverse=True))
