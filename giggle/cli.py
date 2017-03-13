import argparse
import json
import pdb
import pickle
import os

from typing import (
    Any,
    Dict,
    Callable,
)


def train(args):
    # Trains recommender system
    pass

def evaluate(args):
    # Evaluates recommender system
    pass


def web(args):
    # Starts web-server
    from .web_service import app
    app.run('0.0.0.0', port=args.port)


TODO = {
    'train': train,
    'evaluate': evaluate,
    'web': web,
}


def main():

    parser = argparse.ArgumentParser(
        description='Command-line interface for the recommender system',
    )

    subparsers = parser.add_subparsers(
        dest='command',
    )

    # Sub-parser for training
    parser_1 = subparsers.add_parser(
        'train',
        help='Trains a pipeline',
    )
    parser_1.add_argument(
        '-d', '--dataset',
        required=True,
        help='which dataset to use.',
    )
    parser_1.add_argument(
        '-v', '--verbose',
        default=0,
        action='count',
        help='show more output.',
    )

    # Sub-parser for evaluate
    parser_2 = subparsers.add_parser(
        'evaluate',
        help='Benchmarks the current pipeline',
    )
    parser_2.add_argument(
        '-d', '--dataset',
        required=True,
        help='which dataset to use.',
    )
    parser_2.add_argument(
        '--plot',
        required=False,
        default=False,
        action='store_true',
        help='plot accuracy vs. throughput (only for the fields to be extracted).',
    )
    parser_2.add_argument(
        '-v', '--verbose',
        default=0,
        action='count',
        help='show more output.',
    )

    # Sub-parser for web-service
    parser_3 = subparsers.add_parser(
        'web',
        help='Starts a web-service',
    )
    parser_3.add_argument(
        '--port',
        required=False,
        default=6667,
        type=int,
        help='port number.',
    )
    parser_3.add_argument(
        '-v', '--verbose',
        default=0,
        action='count',
        help='show more output.',
    )

    args = parser.parse_args()
    TODO[args.command](args)


if __name__ == '__main__':
    main()