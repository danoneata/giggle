import pdb

jokes = {}

with open('data/jester/jester_items.dat', 'r') as f:
    text = f.read()
    for i, joke in enumerate(text.strip().split('\n\n\n'), 1):
        lines = joke.split('\n')
        assert lines[0] == '{:d}:'.format(i)
        jokes[i] = '\n'.join(lines[1:]).strip()

for i, joke in jokes.items():
    with open('data/jokes/{:d}.txt'.format(i), 'w') as f:
        f.write(joke)
