import argparse
import pdb

from flask import Flask

from flask_sqlalchemy import SQLAlchemy

from functools import partial

import names

import numpy as np

from sqlalchemy.orm import validates

from .config import Config

from .utils import grouper


db = SQLAlchemy()


class User(db.Model):

    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80))

    def __repr__(self):
        return '<User {:d}: {:s}>'.format(self.id, self.name)


class Joke(db.Model):

    __tablename__ = 'jokes'

    id = db.Column(db.Integer, primary_key=True)

    def __repr__(self):
        return '<Joke {:d}>'.format(self.id)


class Rating(db.Model):

    __tablename__ = 'ratings'

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    user = db.relationship('User')

    joke_id = db.Column(db.Integer, db.ForeignKey('jokes.id'), nullable=False)
    joke = db.relationship('Joke')

    rating = db.Column(db.Numeric(precision=3), nullable=False)

    def __repr__(self):
        return '<Rating {:s} for joke {:d} from user {:d}>'.format(
            self.rating,
            self.joke_id,
            self.user_id,
        )

    @validates('rating')
    def validate_rating(self, key, value):
        if not -10 <= value <= 10:
            message = "Rating should be between -10 and 10, but it is {:.3f}"
            raise ValueError(message.format(value))
        else:
            return value


def get_users(nr_users):
    return (User(name=names.get_full_name()) for _ in range(nr_users))


def get_jokes(nr_jokes):
    return (Joke() for _ in range(nr_jokes))


def get_ratings(data):
    return (Rating(user_id=int(user_id), joke_id=int(joke_id), rating=rating)
            for user_id, joke_id, rating in data)


def main():

    parser = argparse.ArgumentParser(description='Script to initialize the database.')
    parser.add_argument(
        '-t', '--todo',
        default=[],
        nargs='+',
        choices=('init', 'drop'),
        help="what operation to perform.")
    args = parser.parse_args()

    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)

    if 'init' in args.todo:

        data = np.loadtxt('data/jester/jester_ratings.dat')
        nr_users = int(np.max(data[:, 0]))
        nr_jokes = int(np.max(data[:, 1]))

        def get_objects(table):
            if table == 'users':
                return get_users(nr_users)
            elif table == 'jokes':
                return get_jokes(nr_jokes)
            elif table == 'ratings':
                return get_ratings(data)
            else:
                assert False, "Unknown table {:s}".format(table)

        my_grouper = partial(grouper, n=10000)

        with app.app_context():
            db.create_all()

            for table in ('users', 'jokes', 'ratings'):
                for objects in my_grouper(get_objects(table)):
                    objects = [obj for obj in objects if obj is not None]
                    db.session.bulk_save_objects(objects)
                    db.session.commit()

    if 'drop' in args.todo:

        with app.app_context():
            db.drop_all()


if __name__ == '__main__':
    main()
