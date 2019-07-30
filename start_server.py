#!/bin/env python3
from helper import get_args_parser
from server import FuzzyServer


def main():
    args = get_args_parser()
    args.model_path = '/Users/user/Desktop/fuzzy-flow/server/nlp/namu_mecab_400.bin'
    fs = FuzzyServer(args)
    fs.start()
    fs.join()


if __name__ == '__main__':
    main()
