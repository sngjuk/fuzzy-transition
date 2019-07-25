import os
import pickle
import re
from collections import OrderedDict
from functools import reduce

import numpy as np
from scipy.spatial import distance

from model import Model
from node import Node
from traverse import search_path, across_vector_space

glossary = OrderedDict()
glossary_vector = []
model = Model()


def most_sim_name(target):
    target_vector = model.get_word_vector(target)

    distances = distance.cdist(np.array([target_vector]), np.array(glossary_vector), "cosine")[0]
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    return list(glossary.items())[int(min_index)][0], 1-min_distance


def add_noema(name):
    if name in glossary:
        return False

    neo = Node(name, model.get_word_vector(name))
    glossary[name] = neo
    glossary_vector.append(neo.vector)

    print(f'\n//{name} added')

    return True


def add_implication(source_name, target_name, probability, core=None):

    add_noema(source_name)
    add_noema(target_name)

    res_prob = glossary[source_name].add_implication(target_name, model.get_word_vector(target_name), probability, core)
    glossary[source_name].sort_implication()

    print(f'\n//{source_name} -> {target_name}; {res_prob[0]}, count: {res_prob[1]} core: {core}')


def find_path(source, dest):
    usr_input = source, dest

    source_sim = None
    dest_sim = None

    if source not in glossary:
        source, source_sim = most_sim_name(source)

    if dest not in glossary:
        dest, dest_sim = most_sim_name(dest)

    # path_prob: [[path_list, prob_list], ...]
    path_prob = search_path(glossary, source, dest)

    for idx, i in enumerate(path_prob):
        if not source == usr_input[0]:
            path_prob[idx][0].insert(0, usr_input[0])
            path_prob[idx][1].insert(0, source_sim)

    for idx, i in enumerate(path_prob):
        if not dest == usr_input[1]:
            path_prob[idx][0].append(usr_input[1])
            path_prob[idx][1].append(dest_sim)

    prob_sum = 0
    for i in path_prob:
        print('path;', i[0])
        print('    ', end='')
        for j in i[1]:
            print('   %2.2f' % j, end='')
        prob = reduce((lambda x, y: x * y), i[1])
        print(' prob; ', '%.2f' % prob)
        prob_sum += prob

    print("\n total prob; %.2f percent" % prob_sum)


def show_implication(name):
    for i in glossary[name].implication:
        print(i)


def across_space(source, dest):
    path_prob = across_vector_space(glossary, model, source, dest)
    print(path_prob)


def list_glossary():
    for i in glossary:
        print(glossary[i].name)


def load_glossary():
    if os.path.isfile('save.p'):
        global glossary
        global glossary_vector
        glossary, glossary_vector = pickle.load(open('save.p', 'rb'))
        print('glossary loaded')


def save_glossary():
    pickle.dump((glossary, glossary_vector), open("save.p", "wb"))


def show_nearest_neighbor(name, sim_threshold=0.39):
    result = model.nearest_words(name, 30)
    for i in result:
        my_regex = r".*" + re.escape(name) + r".*"
        if re.search(my_regex, i[0]):
            continue
        elif i[1] > sim_threshold:
            print(i)


def cli():
    load_glossary()
    print(glossary.keys(), '\n')

    while True:
        print('===== Select =====\nl; list glossary\na; add name\nai; add implication\
              \nac: add core belief \ndi: delete implication \nac: add core belief\
              \ncr: cross vector space \nsi: show implications\
              \nsn: show nearest neighbor\nfp; find path\nx; exit')
        sel = input()

        if sel == 'l':
            # list glossary
            list_glossary()

        elif sel == 'a':
            print('input; name')
            name = input()
            # add noema
            res = add_noema(name)
            if not res:
                print('already exist')

        elif sel == 'ai':
            print('input; source_name target_name probability')
            usr_input = input().split()
            source_name = usr_input[0]
            target_name = usr_input[1]
            probability = float(usr_input[2])
            # add implication
            add_implication(source_name, target_name, probability)

        elif sel == 'ac':
            print('input; source_name target_name probability')
            usr_input = input().split()
            source_name = usr_input[0]
            target_name = usr_input[1]
            probability = float(usr_input[2])
            # add core belief
            add_implication(source_name, target_name, probability, 'core')

        elif sel == 'sn':
            print('input; name')
            name = input()
            # show nearest neighbor
            show_nearest_neighbor(name)

        elif sel == 'fp':
            print('input; source dest')
            usr_input = input().split()
            source = usr_input[0]
            dest = usr_input[1]
            # find path
            find_path(source, dest)

        elif sel == 'cr':
            print('input; source dest')
            usr_input = input().split()
            source = usr_input[0]
            dest = usr_input[1]
            # find path
            across_space(source, dest)

        elif sel == 'si':
            print('input; name')
            name = input()
            # show implications
            show_implication(name)

        elif sel == 'x':
            print('save & exit')
            # save
            save_glossary()
            break

        print('\nok\n')


def main():
    cli()


if __name__ == '__main__':
    main()
