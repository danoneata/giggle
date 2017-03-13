import numpy as np

from sklearn.model_selection import KFold


def count_ratings():
    with open('data/jester/jester_ratings.dat', 'r') as f:
        return len(f.readlines())


def save(i, split, index):
    path = 'data/folds/{:s}_{:02d}.txt'
    with open(path.format(split, i), 'w') as f:
        f.write('\n'.join(map(str, index)))


def main():
    kf = KFold(n_splits=10, shuffle=True, random_state=1337)
    index = np.arange(count_ratings())
    for i, (train_index, test_index) in enumerate(kf.split(index), start=1):
        save(i, 'train', train_index)
        save(i, 'test', test_index)


if __name__ == '__main__':
    main()
