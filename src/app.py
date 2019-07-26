import os
import pickle
import re
from collections import OrderedDict
from functools import reduce

from model import Model
from node import Node
from traverse import search_path, across_vector_space, search_hidden_path_with_length

glossary = OrderedDict()
glossary_vector = []
model = Model()


def preprocess_input():
    in_text = input()
    in_text = re.sub(r'[ \t]+$', '', in_text)
    re_res = re.search(r'\s*(.*)', in_text)
    if re_res:
        in_text = re_res.group(1)

    return in_text


def load_glossary(load_file):
    if os.path.isfile(load_file):
        global glossary
        global glossary_vector
        glossary, glossary_vector = pickle.load(open(load_file, 'rb'))
        print('glossary loaded')


def save_glossary():
    pickle.dump((glossary, glossary_vector), open("save.p", "wb"))


def list_glossary():
    for i in glossary:
        print(glossary[i].name)


def add_noema(name):
    if name in glossary:
        return False

    neo = Node(name, model.get_word_vector(name))
    glossary[name] = neo
    glossary_vector.append(neo.vector)

    print(f'\n//{name} added')

    return True


def add_implication(source_name, target_name, probability):
    add_noema(source_name)
    add_noema(target_name)

    res_prob = glossary[source_name].add_implication(target_name, model.get_word_vector(target_name), probability)
    glossary[source_name].sort_reason()

    print(f'\n//{source_name} -> {target_name}; {res_prob[0]}, count: {res_prob[1]}')


def add_belief(source_name, target_name, probability):
    add_noema(source_name)
    add_noema(target_name)

    res_prob = glossary[source_name].add_belief(target_name, model.get_word_vector(target_name), probability)
    glossary[source_name].sort_reason()

    print(f'\n//{source_name} -> {target_name}; {res_prob[0]}, count: {res_prob[1]}')


def add_membership(source_name, target_name, target_prob, source_prob):
    add_noema(source_name)
    add_noema(target_name)

    res_prob = glossary[source_name].add_membership(target_name, model.get_word_vector(target_name), target_prob)
    glossary[source_name].sort_reason()

    print(f'\n//{source_name} -> {target_name}; {res_prob[0]}, count: {res_prob[1]}')

    res_prob = glossary[target_name].add_membership(source_name, model.get_word_vector(source_name), source_prob)
    glossary[target_name].sort_reason()

    print(f'\n//{target_name} -> {source_name}; {res_prob[0]}, count: {res_prob[1]}')


def show_reasons(name):
    if not len(glossary) or name not in glossary:
        print('empty glossary or not exist noema')
        return

    for i in glossary[name].reason:
        print(i)


def find_path(source, dest):
    if not len(glossary):
        print('empty glossary')
        return

    # path_prob: [[path_list, prob_list], ...]
    path_prob = search_path(glossary, glossary_vector, model, source, dest)

    if not len(path_prob):
        across_space(source, dest)
        print('\nlogical path not found in system!! trying to hop!-\n')

    prob_sum = 0
    for i in path_prob:
        print('path;', i[0])
        print('        ', end='')
        for j in i[1]:
            print('   %2.2f' % j, end='')
        prob = reduce((lambda x, y: x * y), i[1])
        print('\nprob; ', '%.2f' % prob)
        prob_sum += prob

    print("\n total prob; %.2f percent" % prob_sum)


def across_space(source, dest):
    path_prob = across_vector_space(glossary, model, source, dest)
    print(path_prob)


def search_hidden_path(source, length):
    if not len(glossary):
        print('empty glossary')
        return

    path_prob = search_hidden_path_with_length(glossary, glossary_vector, model, source, length)

    if not len(path_prob):
        print(f'not found hidden path for {source} with length {length}!, try smaller length')

    for i in path_prob:
        print('path;', i[0])
        print('        ', end='')
        for j in i[1]:
            print('   %2.2f' % j, end='')
        prob = reduce((lambda x, y: x * y), i[1])
        print('\nprob; ', '%.2f' % prob)


def show_nearest_neighbor(name, num=30, sim_threshold=0.39):
    result = model.filtered_nearest_neighbor(name, num, sim_threshold)
    for i in result:
        print(i)


def show_distance(text1, text2):
    print('dist; %.4f' % model.get_distance(text1, text2))


def cli(load_file):
    load_glossary(load_file)
    print(glossary.keys(), '\n')

    while True:
        print('===== Select =====\nl; list glossary\na; add name\nai; add implication\
              \nab: add belief \nam: add membership\
              \ncr: cross vector space \nsr: show reasons \nsd: show distance\
              \nsh: search hidden path \nsn: show nearest neighbor\nfp; find path \
              \ndi: delete implication \nx; exit')
        sel = preprocess_input()

        if sel == 'l':
            # list glossary
            list_glossary()

        elif sel == 'a':
            print('input; name')
            name = preprocess_input()
            # add noema
            res = add_noema(name)
            if not res:
                print('already exist')

        elif sel == 'ai':
            print('input; source_name ')
            source_name = preprocess_input()
            print('input; target_name ')
            target_name = preprocess_input()
            print('input; probability')
            probability = float(preprocess_input())
            # add implication
            add_implication(source_name, target_name, probability)

        elif sel == 'ab':
            print('input; source_name ')
            source_name = preprocess_input()
            print('input; target_name ')
            target_name = preprocess_input()
            print('input; probability')
            probability = float(preprocess_input())
            # add belief
            add_belief(source_name, target_name, probability)

        elif sel == 'am':
            print('input; source_name ')
            source_name = preprocess_input()
            print('input; target_name ')
            target_name = preprocess_input()
            print('input; source->target probability')
            target_prob = float(preprocess_input())
            print('input; target->source probability')
            source_prob = float(preprocess_input())
            # add membership
            add_membership(source_name, target_name, target_prob, source_prob)

        elif sel == 'sr':
            print('input; name')
            name = preprocess_input()
            # show reasons
            show_reasons(name)

        elif sel == 'fp':
            print('input; source ')
            source = preprocess_input()
            print('input; dest ')
            dest = preprocess_input()
            # find path
            find_path(source, dest)

        elif sel == 'cr':
            print('input; source ')
            source = preprocess_input()
            print('input; dest ')
            dest = preprocess_input()
            # find path
            across_space(source, dest)

        elif sel == 'sh':
            print('input; source')
            source = preprocess_input()
            print('input; length')
            length = int(preprocess_input())
            # search hidden paths with length
            search_hidden_path(source, length)

        elif sel == 'sn':
            print('input; name')
            name = preprocess_input()
            # show nearest neighbor
            show_nearest_neighbor(name)

        elif sel == 'sd':
            print('input; word1')
            word1 = preprocess_input()
            print('input; word2')
            word2 = preprocess_input()
            # show word distance
            show_distance(word1, word2)

        elif sel == 'x':
            print('save & exit')
            # save
            save_glossary()
            break

        print('\nok\n')


def main():
    load_file = 'save.p'
    cli(load_file)


if __name__ == '__main__':
    main()
