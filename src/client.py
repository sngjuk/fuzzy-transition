#!/usr/bin/env python3
import pickle
import re
from collections import OrderedDict
from time import sleep

import zmq

import node


class FuzzyClient:
    def __init__(self, ip='localhost', port=5555):
        self.ip = ip
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect('tcp://%s:%d' % (self.ip, self.port))

        self.glossary = OrderedDict()
        self.glossary_vector = []
        self.setting = {'depth_limit': 9, 'jump_limit': 1, 'num': 30, 'sim_th': 0.39}

    @staticmethod
    def req(rep_name, glossary=None, glossary_vector=None, name1=None, name2=None, setting=None):
        req_json = {
            'req': rep_name,
            'glossary': glossary,
            'glossary_vector': glossary_vector,
            'name1': name1,
            'name2': name2,
            'setting': setting
        }
        return req_json

    @staticmethod
    def preprocess_input():
        in_text = input()
        in_text = re.sub(r'[ \t]+$', '', in_text)
        re_res = re.search(r'\s*(.*)', in_text)
        if re_res:
            in_text = re_res.group(1)

        return in_text

    def list_names(self):
        print('// list names')
        for i in self.glossary:
            print(self.glossary[i].name)

    def add_name(self, name):
        print('// add name')
        if name in self.glossary:
            return False

        neo = node.Node(name, self.get_word_vector(name))
        self.glossary[name] = neo
        self.glossary_vector.append(neo.vector)

        print(f'\n//{name} added')
        return True

    def delete_name(self, name):
        print('// delete name')
        if name not in self.glossary:
            return False

        rm_idx = list(self.glossary.keys()).index(name)
        del self.glossary[name]
        del self.glossary_vector[rm_idx]
        print(f'{name} deleted')

    def add_implication(self, source_name, target_name, probability):
        print('// add implication')
        self.add_name(source_name)
        self.add_name(target_name)

        res_prob = self.glossary[source_name].add_implication(target_name, self.get_word_vector(target_name), probability)
        self.glossary[source_name].sort_reason()
        print(f'\n//{source_name} -> {target_name}; {res_prob[0]}, count: {res_prob[1]}')

    def add_belief(self, source_name, target_name, probability):
        print('// add belief')
        self.add_name(source_name)
        self.add_name(target_name)

        res_prob = self.glossary[source_name].add_belief(target_name, self.get_word_vector(target_name), probability)
        self.glossary[source_name].sort_reason()

        print(f'\n//{source_name} -> {target_name}; {res_prob[0]}, count: {res_prob[1]}')

    def add_membership(self, source_name, target_name, target_prob, source_prob):
        print('// add membership')
        self.add_name(source_name)
        self.add_name(target_name)

        res_prob = self.glossary[source_name].add_membership(target_name, self.get_word_vector(target_name), target_prob)
        self.glossary[source_name].sort_reason()

        print(f'\n//{source_name} -> {target_name}; {res_prob[0]}, count: {res_prob[1]}')

        res_prob = self.glossary[target_name].add_membership(source_name, self.get_word_vector(source_name), source_prob)
        self.glossary[target_name].sort_reason()

        print(f'\n//{target_name} -> {source_name}; {res_prob[0]}, count: {res_prob[1]}')

    def show_name(self, name):
        if len(self.glossary) == 0 or name not in self.glossary:
            print('empty glossary or not exist name')
            return

        print(f'//// {name}')
        print('=== membership ===')
        for i in self.glossary[name].membership:
            print(f'->{i}; prob {self.glossary[name].membership[i][0]}, count {self.glossary[name].membership[i][1]}')

        print('\n=== implication ===')
        for i in self.glossary[name].implication:
            print(f'->{i}; prob {self.glossary[name].implication[i][0]}, count {self.glossary[name].implication[i][1]}')

        print('\n=== belief ===')
        for i in self.glossary[name].belief:
            print(f'->{i}; prob {self.glossary[name].belief[i][0]}, count {self.glossary[name].belief[i][1]}')

    def get_word_vector(self, name):
        req_x = self.req('gw', name1=name)
        self.socket.send(pickle.dumps(req_x))

        res = self.socket.recv()
        loaded_res = pickle.loads(res)
        return loaded_res['res_data']

    def get_glossary_list(self):
        req_x = self.req('sl')
        self.socket.send(pickle.dumps(req_x))

        res = self.socket.recv()
        loaded_res = pickle.loads(res)
        print(loaded_res['res_data'])

    def load_glossary(self, file_name):
        req_x = self.req('lg', name1=file_name)
        self.socket.send(pickle.dumps(req_x))

        res = self.socket.recv()
        loaded_res = pickle.loads(res)
        if loaded_res['res_data']:
            self.glossary, self.glossary_vector = loaded_res['res_data']
            print(f'{file_name} loaded!')
        else:
            print(f'\'{file_name}\' file not found in \'save\' folder ;)\n')

    def save_glossary(self, file_name):
        req_x = self.req('x', glossary=self.glossary, glossary_vector=self.glossary_vector, name1=file_name)
        self.socket.send(pickle.dumps(req_x))

        res = self.socket.recv()
        loaded_res = pickle.loads(res)
        if loaded_res['res_data']:
            print(loaded_res['res_data'])

    def find_path(self, source, dest):
        if not len(self.glossary):
            print('empty glossary')
            return

        req_x = self.req('fp', glossary=self.glossary, glossary_vector=self.glossary_vector,
                         name1=source, name2=dest, setting=self.setting)
        self.socket.send(pickle.dumps(req_x))

        res = self.socket.recv()
        loaded_res = pickle.loads(res)
        print(loaded_res['res_data'])

    def across_space(self, source, dest):
        req_x = self.req('cr', name1=source, name2=dest)
        self.socket.send(pickle.dumps(req_x))

        res = self.socket.recv()
        loaded_res = pickle.loads(res)
        print(loaded_res['res_data'])

    def search_possible_path(self, source, length):
        if not len(self.glossary):
            print('empty glossary')
            return

        setting = {'depth_limit': length, 'jump_limit': self.setting['jump_limit'], 'sim_th': self.setting['sim_th']}
        req_x = self.req('sp', glossary=self.glossary, glossary_vector=self.glossary_vector,
                         name1=source, name2=None, setting=setting)
        self.socket.send(pickle.dumps(req_x))

        res = self.socket.recv()
        loaded_res = pickle.loads(res)
        print(loaded_res['res_data'])

    def show_nearest_neighbor(self, name, num=30, sim_th=0.39):
        setting = {'num': num, 'sim_th': sim_th}
        req_x = self.req('nn', name1=name, name2=None, setting=setting)
        self.socket.send(pickle.dumps(req_x))

        res = self.socket.recv()
        loaded_res = pickle.loads(res)
        print(loaded_res['res_data'])

    def show_similarity(self, text1, text2):
        req_x = self.req('ss', name1=text1, name2=text2)
        self.socket.send(pickle.dumps(req_x))

        res = self.socket.recv()
        loaded_res = pickle.loads(res)
        print(loaded_res['res_data'])

    def show_all_names(self):
        for name in self.glossary:
            print('\n')
            self.show_name(name)

    def clear_glossary(self):
        self.glossary = OrderedDict()
        self.glossary_vector = []

    def user_select(self):
        print('~ welcome ~')
        print('load file name? (example \'bug.p\')')
        save_filename = self.preprocess_input()
        self.load_glossary(save_filename)
        self.list_names()

        while True:
            print('===== select ===== \nsl: server glossaries list \nlg: load server glossary\
                  \nln; list names\na; add name\ndn: delete name\nsa: show all names\nai; add implication\
                  \nab: add belief \nam: add membership \nnn: show nearest neighbor \nst: find path setting\
                  \nsn: show name \nss: show similarity \n\n-=-=- paths -=-=-\
                  \nfp; find path  \ncr: cross vector space \ncg: clear current glossary\
                  \nsp: search possible path \n \
                  \n----- exit -----\
                  \nx; save &exit \nxx; exit without save')

            try:
                sel = self.preprocess_input()

                if sel == 'sl':
                    self.get_glossary_list()

                elif sel == 'lg':
                    self.get_glossary_list()
                    print('input; load file name')
                    name = self.preprocess_input()
                    # add name
                    self.load_glossary(name)
                    self.list_names()

                elif sel == 'ln':
                    # list glossary
                    self.list_names()

                elif sel == 'a':
                    print('input; name')
                    name = self.preprocess_input()
                    # add name
                    res = self.add_name(name)
                    if not res:
                        print('already exist')

                elif sel == 'dn':
                    print('input; name to delete')
                    name = self.preprocess_input()
                    # delete name
                    self.delete_name(name)

                elif sel == 'sa':
                    # show all names
                    self.show_all_names()

                elif sel == 'ai':
                    print('input; source_name ')
                    source_name = self.preprocess_input()
                    print('input; target_name ')
                    target_name = self.preprocess_input()
                    print('input; probability')
                    probability = float(self.preprocess_input())
                    # add implication
                    self.add_implication(source_name, target_name, probability)

                elif sel == 'ab':
                    print('input; source_name ')
                    source_name = self.preprocess_input()
                    print('input; target_name ')
                    target_name = self.preprocess_input()
                    print('input; probability')
                    probability = float(self.preprocess_input())
                    # add belief
                    self.add_belief(source_name, target_name, probability)

                elif sel == 'am':
                    print('input; source_name ')
                    source_name = self.preprocess_input()
                    print('input; target_name ')
                    target_name = self.preprocess_input()
                    print(f'input; {source_name}->{target_name} similarity')
                    target_prob = float(self.preprocess_input())
                    print(f'input; {target_name}->{source_name} similarity')
                    source_prob = float(self.preprocess_input())
                    # add membership
                    self.add_membership(source_name, target_name, target_prob, source_prob)

                elif sel == 'sn':
                    print('input; name')
                    name = self.preprocess_input()
                    # show relations
                    self.show_name(name)

                elif sel == 'st':
                    print('input; depth limit')
                    self.setting['depth_limit'] = int(self.preprocess_input())
                    print('input; jump limit')
                    self.setting['jump_limit'] = int(self.preprocess_input())

                elif sel == 'fp':
                    print('input; source ')
                    source = self.preprocess_input()
                    print('input; dest ')
                    dest = self.preprocess_input()
                    # find path
                    self.find_path(source, dest)

                elif sel == 'cr':
                    print('input; source ')
                    source = self.preprocess_input()
                    print('input; dest ')
                    dest = self.preprocess_input()
                    # find path
                    self.across_space(source, dest)

                elif sel == 'sp':
                    print('input; source')
                    source = self.preprocess_input()
                    print('input; length')
                    length = int(self.preprocess_input())
                    # search possible paths with length
                    self.search_possible_path(source, length)

                elif sel == 'nn':
                    print('input; name')
                    name = self.preprocess_input()
                    # show nearest neighbor
                    self.show_nearest_neighbor(name)

                elif sel == 'ss':
                    print('input; word1')
                    word1 = self.preprocess_input()
                    print('input; word2')
                    word2 = self.preprocess_input()
                    # show word distance
                    self.show_similarity(word1, word2)

                elif sel == 'cg':
                    # clear current glossary
                    self.clear_glossary()

                elif sel == 'x':
                    print('save file name?')
                    save_filename = self.preprocess_input()
                    # save
                    self.save_glossary(save_filename)
                    break

                elif sel == 'xx':
                    print('exit without save')
                    print('see ya')
                    break

                print('\nok\n')

            except KeyboardInterrupt:
                print('  \n\n### Plz Enter \'x\' or \'xx\' to exit ###\n')
                sleep(0.33)


def main():
    fc = FuzzyClient(ip='35.200.11.163', port=8888)
    fc.user_select()


if __name__ == '__main__':
    main()
