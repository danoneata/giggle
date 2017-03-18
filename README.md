A recommender system for jokes based on the Jester dataset.

# Setup

Run the following commands to create a virtual environment, install the requirements and generate a command line interface, `giggle`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py install
```

## Populate database

Download the data:

```bash
mkdir -p data/jester && cd data/jester
for f in jester_ratings jester_items; do
    wget http://www.seas.harvard.edu/courses/cs281/data/${f}.tar.gz
    tar xvzf ${f}.tar.gz
done
cd -
```

Create the database:

```bash
# Create a new user
sudo -u postgres createuser -s jester
# Create a new database
createdb -U jester jester_db
```

Set up environment variables:

```
# Set the URL to the database as a system variable
export DATABASE_URL="postgresql://jester@localhost/jester_db"
export SECRET_KEY=???
```

Run the script to populate the database:

```bash
python -m giggle.models --todo init
```

# Usage

The command line interface, `giggle`, exposes three sub-commands (see the [next section](#details-and-examples) for more details and examples):

* `train`: Trains a predictive model
* `evaluate`: Creates a report with the performance of the current model
* `web`: Starts an web service that can be used for prediction

You can get more information about what arguments each sub-command accepts by running the help command:

```bash
giggle --help
giggle predict --help
```

## Details and examples

# Development

In order to keep code-base standardized, I have tried:

* to keep the code [PEP8](ihttps://www.python.org/dev/peps/pep-0008/) compliant:

```bash
find examples giggle scripts -name '*py' | xargs pep8 --ignore E501
```

* to add [type annotations](https://docs.python.org/3/library/typing.html) and make sure `mypy` accepts it:

```bash
mypy --fast-parser --incremental -m giggle
```

In order to keep the project organized, I am keeping:

* [a list of things to do](../blob/master/TODO.md)
* [a list of ideas and resources](../blob/master/IDEAS.md)
