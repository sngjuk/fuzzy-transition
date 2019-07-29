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
setting = {'depth_limit': 9, 'jump_limit': 1}


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


def delete_noema(name):
    if name not in glossary:
        return False

    rm_idx = list(glossary.keys()).index(name)
    del glossary[name]
    del glossary_vector[rm_idx]
    print(f'{name} deleted')


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


def show_noema(name):
    if len(glossary) == 0 or name not in glossary:
        print('empty glossary or not exist noema')
        return

    print(f'// {name}')
    print('=== membership ===')
    for i in glossary[name].membership:
        print(f'->{i}; prob {glossary[name].membership[i][0]}, count {glossary[name].membership[i][1]}')

    print('\n=== implication ===')
    for i in glossary[name].implication:
        print(f'->{i}; prob {glossary[name].implication[i][0]}, count {glossary[name].implication[i][1]}')

    print('\n=== belief ===')
    for i in glossary[name].belief:
        print(f'->{i}; prob {glossary[name].belief[i][0]}, count {glossary[name].belief[i][1]}')


def find_path(source, dest):
    if not len(glossary):
        print('empty glossary')
        return

    # path_prob: [[path_list, prob_list], ...]
    path_prob = search_path(glossary, glossary_vector, model, source, dest, setting['depth_limit'], setting['jump_limit'])

    if not len(path_prob):
        across_space(source, dest)
        print('\nlogical path not found!! this is hop result!-\n')

    prob_sum = 0
    for i in path_prob:
        print('path;', i[0])
        print('        ', end='')
        for j in i[1]:
            print('   %2.2f' % j, end='')
        prob = reduce((lambda x, y: x * y), i[1])
        print('\n~ %.2f' % prob)
        prob_sum += prob

    print("\n//total ~ %.2f " % prob_sum)


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
        print('\n~ %.2f' % prob)


def show_nearest_neighbor(name, num=30, sim_threshold=0.39):
    result = model.filtered_nearest_neighbor(name, num, sim_threshold)
    for i in result:
        print(i)


def show_similarity(text1, text2):
    print('sim; %.4f' % model.get_similarity(text1, text2))


def cli(load_file):
    load_glossary(load_file)
    print(glossary.keys(), '\n')

    while True:
        print('===== select =====\nl; list glossary\na; add name\ndn: delete name\nai; add implication\
              \nab: add belief \nam: add membership \nnn: show nearest neighbor \
              \nsn: show noema \nss: show similarity \n\n=== paths ===\
              \nfp; find path  \ncr: cross vector space \
              \nsh: search hidden path \n\
              \nx; exit')
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

        elif sel == 'dn':
            print('input; name to delete')
            name = preprocess_input()
            # delete noema
            delete_noema(name)

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

        elif sel == 'sn':
            print('input; name')
            name = preprocess_input()
            # show relations
            show_noema(name)

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

        elif sel == 'nn':
            print('input; name')
            name = preprocess_input()
            # show nearest neighbor
            show_nearest_neighbor(name)

        elif sel == 'ss':
            print('input; word1')
            word1 = preprocess_input()
            print('input; word2')
            word2 = preprocess_input()
            # show word distance
            show_similarity(word1, word2)

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
