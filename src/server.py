"""
Server for FuzzyClient
"""
import glob
import os
import pickle
import random
import threading
from os import path

import portalocker
import zmq
from scipy._lib.six import reduce

from helper import set_logger, get_args_parser
from nlp.model import EmbedModel
from traverse import search_path, search_possible_path_with_length, across_vector_space


class FuzzyServer(threading.Thread):
    def __init__(self, args):
        super().__init__()
        """Server routine"""
        self.logger = set_logger('VENTILATOR')
        self.model_path = os.path.abspath(args.model_path)

        self.port = args.port
        self.thread_num = args.thread_num

        self.url_worker = "inproc://workers"
        self.url_client = "tcp://*:" + self.port
        self.logger.info('opened server : %s' % self.url_client)
        self.logger.info('num of threads : %s' % self.thread_num)

        # Load model
        self.logger.info('loading model...')
        self.model = EmbedModel(self.model_path)
        self.logger.info('model load done.')
        random.seed()

        # Prepare Context and Sockets
        self.logger.info('Prepare Context and Sockets...')
        self.context = zmq.Context.instance()

        # Socket to talk to clients
        self.logger.info('opening client socket...')
        self.clients = self.context.socket(zmq.ROUTER)
        self.clients.bind(self.url_client)

        # Socket to talk to workers
        self.logger.info('opening worker socket...')
        self.workers = self.context.socket(zmq.DEALER)
        self.workers.bind(self.url_worker)
        self.threads = []

    def run(self):
        # Launch pool of worker threads
        self.logger.info('starting workers...')

        for i in range(self.thread_num):
            #thread = threading.Thread(target=worker_routine, args=(url_worker,i,))
            thread = FuzzyServer.MgWorker(worker_url=self.url_worker, worker_id=i, model=self.model)
            thread.start()
            self.threads.append(thread)

        zmq.proxy(self.clients, self.workers)

    def close(self):
        self.logger.info('shutting down...')
        for p in self.threads:
            p.close()
            print(p, ': thread down.')
        self.join()

    def __exit__(self):
        self.close()

    class MgWorker(threading.Thread):
        def __init__(self, worker_url, worker_id, model, context=None):
            super().__init__()
            self.logger = set_logger('WORKER-%d ' % worker_id)
            self.worker_url = worker_url
            self.worker_id = worker_id
            self.model = model
            self.context = context
            self.socket = None
            self.save_path = path.join('save')

        @staticmethod
        def rep(rep_name, res_data=None):
            rep_json = {
                'rep': rep_name,
                'res_data': res_data
            }
            return rep_json

        def find_path(self, glossary, glossary_vector, source, dest, setting):
            ret_str = ''
            if not len(glossary):
                ret_str += 'empty glossary\n'
                return

            ret_str += '\n// find path\n'

            # path_prob: [[path_list, prob_list], ...]
            path_prob = search_path(glossary, glossary_vector, self.model, source, dest,
                                    setting['depth_limit'], setting['jump_limit'])

            if not len(path_prob):
                self.across_space(source, dest)
                ret_str += '\nlogical path not found!! this is hop result!-\n\n'

            prob_sum = 0
            for i in path_prob:
                ret_str += 'path; ' + str(i[0]) + '\n'
                ret_str += '        '
                for j in i[1]:
                    ret_str += '   %2.2f' % j
                prob = reduce((lambda x, y: x * y), i[1])
                ret_str += '\n~ %.2f\n' % prob
                prob_sum += prob

            ret_str += '\n//total ~ %.2f ' % prob_sum
            return ret_str

        def search_possible_path(self, glossary, glossary_vector, source, setting):
            ret_str = '// show possible paths\n'
            if not len(glossary):
                ret_str += 'empty glossary\n'
                return
            ret_str += '\n// possible paths\n'

            path_prob = search_possible_path_with_length(glossary, glossary_vector,
                                                         self.model, source, setting['depth_limit'])

            if not len(path_prob):
                ret_str += f'not found hidden path for {source} with length: \
                            {setting["depth_limit"]}! try smaller length\n'

            for i in path_prob:
                ret_str += 'path;' + str(i[0]) + '\n'
                ret_str += '        '
                for j in i[1]:
                    ret_str += '   %2.2f' % j
                prob = reduce((lambda x, y: x * y), i[1])
                ret_str += '\n~ %.2f\n' % prob
            return ret_str

        def across_space(self, source, dest):
            ret_str = ''
            ret_str += '\n// cross vector space\n'
            path_prob = across_vector_space(self.model, source, dest)
            ret_str += str(path_prob)
            return ret_str

        def show_nearest_neighbor(self, name, num=30, sim_threshold=0.39):
            ret_str = '// show nearest neighbor\n'
            result = self.model.filtered_nearest_neighbor(name, num, sim_threshold)
            for i in result:
                ret_str += str(i) + '\n'
            return ret_str

        def show_similarity(self, text1, text2):
            ret_str = 'sim; %.4f' % self.model.get_similarity(text1, text2)
            return ret_str

        def load_glossary(self, file_name):
            file_path = path.join(self.save_path, file_name)

            if os.path.isfile(file_path):
                with portalocker.Lock(file_path, 'rb+', timeout=3) as fh:
                    glossary, glossary_vector = pickle.load(fh)
                    fh.flush()
                    os.fsync(fh.fileno())
                    return glossary, glossary_vector
            else:
                return None

        def save_glossary(self, glossary, glossary_vector, file_name):
            res_str = '//save glossary\n'
            file_path = path.join(self.save_path, file_name)
            if os.path.isfile(file_path):
                res_str += f'{file_name} updated!'
            else:
                res_str += f'{file_name} written'

            with portalocker.Lock(file_path, 'wb+', timeout=3) as fh:
                pickle.dump((glossary, glossary_vector), fh)
                fh.flush()
                os.fsync(fh.fileno())
            return res_str

        def get_word_vector(self, name):
            return self.model.get_word_vector(name)

        def get_glossary_list(self):
            res_str = '//server save file list\n'
            file_list = glob.glob(path.join(self.save_path, '*'))
            for f in file_list:
                res_str += str(path.basename(f)) + '\n'
            return res_str

        def run(self):
            """Worker routine"""
            self.context = self.context or zmq.Context.instance()
            # Socket to talk to dispatcher
            self.socket = self.context.socket(zmq.REP)
            self.socket.connect(self.worker_url)

            while True:
                self.logger.info('waiting for query worker id %d: ' % (int(self.worker_id)))
                # query  = self.socket.recv().decode("utf-8")
                request = self.socket.recv()
                request = pickle.loads(request)

                self.logger.info('request\treq worker id %d: %s' % (int(self.worker_id), str(request['req'])))

                rq_res = None
                if request['req'] == 'fp':
                    # find path
                    rq_res = self.find_path(request['glossary'], request['glossary_vector'],
                                            request['name1'], request['name2'], request['setting'])

                elif request['req'] == 'sp':
                    # search possible paths with length
                    rq_res = self.search_possible_path(request['glossary'], request['glossary_vector'],
                                                       request['name1'], request['setting'])

                elif request['req'] == 'cr':
                    # across space
                    rq_res = self.across_space(request['name1'], request['name2'])

                elif request['req'] == 'nn':
                    # show nearest neighbor
                    rq_res = self.show_nearest_neighbor(request['name1'])

                elif request['req'] == 'ss':
                    # show word distance
                    rq_res = self.show_similarity(request['name1'], request['name2'])

                elif request['req'] == 'gw':
                    # get word vector
                    rq_res = self.get_word_vector(request['name1'])

                elif request['req'] == 'sl':
                    # get word vector
                    rq_res = self.get_glossary_list()

                elif request['req'] == 'lg':
                    # get word vector
                    rq_res = self.load_glossary(request['name1'])

                elif request['req'] == 'x':
                    # get word vector
                    rq_res = self.save_glossary(request['glossary'], request['glossary_vector'], request['name1'])

                rq_msg = pickle.dumps(self.rep(request['req'], rq_res))

                # response.
                self.socket.send(rq_msg)

        def close(self):
            self.logger.info('shutting %d worker down ...' % self.worker_id)
            self.join()
            self.logger.info('%d worker terminated!' % self.worker_id)


def main():
    args = get_args_parser()
    fs = FuzzyServer(args)
    fs.start()
    fs.join()


if __name__ == '__main__':
    main()
