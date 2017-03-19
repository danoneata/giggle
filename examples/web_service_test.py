import argparse
import json
import pdb
import requests


def add_data(args):
    URL = 'http://localhost:6667/addData/'
    data = {
        "user": args.user,
        "joke": args.joke,
        "rating": args.rating,
    }
    response = requests.post(URL, json=data)
    print(json.dumps(json.loads(response.text), indent=4))


def predict_interests(args):
    URL = 'http://localhost:6667/predictInterests/{:d}'
    response = requests.get(URL.format(args.user))
    print(json.dumps(json.loads(response.text), indent=4))


def similar_items(args):
    URL = 'http://localhost:6667/similarItems/{:d}'
    response = requests.get(URL.format(args.joke))
    print(json.dumps(json.loads(response.text), indent=4))


TODO = {
    'add': add_data,
    'predict': predict_interests,
    'sims': similar_items,
}


def main():

    parser = argparse.ArgumentParser(
        description='Sends requests to the running web-service.',
    )
    subparsers = parser.add_subparsers(dest='command')

    parser_1 = subparsers.add_parser(
        'add',
        help='Adds a new rating in the database',
    )
    parser_1.add_argument(
        '-u', '--user',
        type=int,
        required=True,
        help='user ID',
    )
    parser_1.add_argument(
        '-j', '--joke',
        type=int,
        help='joke ID',
    )
    parser_1.add_argument(
        '-r', '--rating',
        type=float,
        help='rating',
    )

    parser_2 = subparsers.add_parser(
        'predict',
        help='Predicts interests for the given user',
    )
    parser_2.add_argument(
        '-u', '--user',
        type=int,
        required=True,
        help='user ID',
    )

    parser_3 = subparsers.add_parser(
        'sims',
        help='Finds similar jokes to given one',
    )
    parser_3.add_argument(
        '-j', '--joke',
        type=int,
        required=True,
        help='joke ID',
    )

    args = parser.parse_args()
    TODO[args.command](args)


if __name__ == '__main__':
    main()
