from collections import OrderedDict


class Node:
    name = None
    vector = None
    _membership = None
    _implication = None
    _belief = None
    _reason = None

    def __init__(self, name, vector):
        self.name = name
        self.vector = vector
        self.membership = OrderedDict()
        self.implication = OrderedDict()
        self.belief = OrderedDict()
        self._reason = OrderedDict()

    # membership; OrderedDict() { name; [probability, count], ... }
    @property
    def membership(self):
        return self._membership

    @membership.setter
    def membership(self, value):
        self._membership = value

    def add_membership(self, target, vector, probability, belief=None):
        probability, count = self.add_reason(target, vector, probability, belief)
        self._membership[target] = [probability, count]
        return probability, count

    # implication; OrderedDict() { name; [probability, count], ... }
    @property
    def implication(self):
        return self._implication

    @implication.setter
    def implication(self, value):
        self._implication = value

    def add_implication(self, target, vector, probability):
        probability, count = self.add_reason(target, vector, probability)
        self._implication[target] = [probability, count]
        return probability, count

    # belief; OrderedDict() { name; [probability, count], ... }
    @property
    def belief(self):
        return self._belief

    @belief.setter
    def belief(self, value):
        self._belief = value

    def add_belief(self, target, vector, probability, belief=True):
        probability, count = self.add_reason(target, vector, probability, belief)
        self._belief[target] = [probability, count]
        return probability, count

    # reason; OrderedDict() of list, { name : [vector, probability, count], ... }
    @property
    def reason(self):
        return self._reason

    @reason.setter
    def reason(self, value):
        # reason can not be set from the outside
        self._reason = self._reason

    # source, target, target_vector, probability, belief
    def add_reason(self, target, vector, probability, belief=None):
        # first learn
        if target not in self._reason:
            self._reason[target] = [vector, probability, 1]
            return self._reason[target][1], self._reason[target][2]

        if belief:
            self._reason[target][0] = vector
            self._reason[target][1] = probability
            self._reason[target][2] += 1
            return self._reason[target][1], self._reason[target][2]

        # default add-reason procedure
        target_reason = self._reason[target]
        adjusted_probability = \
            (target_reason[1] * target_reason[2] + probability) / (target_reason[2] + 1)

        self._reason[target][1] = adjusted_probability
        self._reason[target][2] += 1

        return self._reason[target][1], self._reason[target][2]

    def sort_reason(self):
        self._reason = OrderedDict(sorted(self._reason.items(), key=lambda x: x[1][1], reverse=True))
