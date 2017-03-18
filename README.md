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

Create a new user and the database:

```bash
sudo -u postgres createuser -s jester
createdb -U jester jester_db
```

Set up environment variables, the URL to the database and a secret key required by Flask:

```
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
giggle train --help
giggle evaluate --help
giggle web --help
```

## Details and examples

Below are some examples for the three sub-commands mentioned above.

* Evaluates a neighbourhood-based recommender algorithm, `neigh`, on the `large` setting of the dataset using K-fold cross-validation:

```bash
giggle evaluate -d large -r neigh
```

The command will print at the standard output a report consisting of the metric (root mean squared error) for the three folds and its mean and standard error values. Here is the ouptput of running the previous command:

```
 0 4.4660
 1 4.4695
 2 4.4740
--------------
4.4699 ± 0.002
```

* Trains a neighbourhood-based recommender algorithm, `neigh`, on the entire `large` dataset:

```bash
giggle train -d large -r neigh -v
```

* Starts a web server using the neighbourhood-based recommender algorithm, `neigh`:

```bash
RECOMMENDER=neigh giggle web -v
```

In order to check that the web-service is running properly, you can use the [`web_service_test.py`](examples/web_service_test.py) script. Here are some examples:

```bash
python examples/web_service_test.py predict -u 21
python examples/web_service_test.py add -u 21 -j 17 -r 7.3
```

# Development

In order to have the code-base standardized and project standardized, I have tried:

* to keep the code [PEP8](ihttps://www.python.org/dev/peps/pep-0008/) compliant:

```bash
find examples giggle scripts -name '*py' | xargs pep8 --ignore E501
```

* to add [type annotations](https://docs.python.org/3/library/typing.html) and make sure `mypy` accepts it:

```bash
mypy --fast-parser --incremental -m giggle
```

* to write tests using [py.test](http://doc.pytest.org/en/latest/):

```bash
pytest tests -v
```

* to keep [a list of things to do](../blob/master/TODO.md)
* to keep [a list of ideas and resources](../blob/master/IDEAS.md)
