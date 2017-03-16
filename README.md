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

```bash
python scripts/create_folds.py
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

* to keep the code [PEP8](ihttps://www.python.org/dev/peps/pep-0008/) compliant.
* to add [type annotations](https://docs.python.org/3/library/typing.html) and make sure `mypy` accepts it:

```bash
mypy --fast-parser --incremental -m giggle
```

## TODO

- [x] Build high-level API
- [x] Decide upon a dataset
- [x] Get dataset into a database
- [x] Prepare the experimental setup: create folds, write metric
- [x] Exploratory data analysis
- [x] Implement "addData" REST point
- [x] Create a subset of the data
- [x] Implement and evaluate some baselines
- [ ] Implement and evaluate an item-based prediction method
- [ ] Implement and evaluate a matrix factorization method
- [ ] Implement more entry points: find similar jokes to a given joke
- [ ] Extra: Predict ratings from items' features
- [ ] Check code to be PEP8-compliant
- [ ] Add type annotations
- [ ] Add tests
- [ ] Regenerate database with not-nullable constraint
- [ ] Clean up `README.md`: split in `TODO.md` and `IDEAS.md`

## Ideas

From [CS281, Assignment 3](http://www.seas.harvard.edu/courses/cs281/files/assignment-3.pdf):

> We might imagine that some jokes are just better or worse than others. We might also imagine that some users tend to have higher or lower means in their ratings. Introduce such biases into the model and fit it again, learning these new biases as well. Explain how you did this. One side-effect is that you should be able to rank the jokes from best to worst. What are the best and worst jokes?

From Programming Collective Intelligence, chapter 2, exercise 5:

> Audioscrobbler. Take a look at http://www.audioscrobbler.net, a dataset containing music preferences for a large set of users. Use their web services API to get a set of data for making and building a music recommendation system.

## Resources

Below there are some resources that discuss collaborative filtering: the courses have assignments on the topic, while the books present a more theoretical discussion. I've studied these resources to get some inspiration.

- Coursera Stanford: [Machine Learning](http://www.holehouse.org/mlclass/)
- Harvard: [CS109 Data Science](http://cs109.github.io/2015/)
- Washington: [CSE446 Machine Learning](https://courses.cs.washington.edu/courses/cse446/15sp/)
- Brown: [CS195 Introduction to Machine Learning](http://cs.brown.edu/courses/cs195-5/spring2012/)
- Harvard: [CS281 Advanced Machine Learning](http://www.seas.harvard.edu/courses/cs281/)
- Toby Segaran: [Programming Collective Intelligence](https://www.amazon.com/Programming-Collective-Intelligence-Building-Applications/dp/0596529325/ref=sr_1_1?ie=UTF8&qid=1488878344&sr=8-1&keywords=programming+collective)
- David Barber: [Bayesian Reasoning and Machine Learning](https://www.amazon.com/Bayesian-Reasoning-Machine-Learning-Barber/dp/0521518148/ref=sr_1_1?ie=UTF8&qid=1488878372&sr=8-1&keywords=bayesian+reasoning+and+machine+learning)
- Kevin Murphy: [Machine Learning: A Probabilistic Approach](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020/ref=sr_1_1?ie=UTF8&qid=1488878389&sr=8-1&keywords=machine+learning+a+probabilistic+perspective)

### Coursera Stanford: Machine Learning

- Introduction into collaborative filtering
- Presents the most common way to formulate the task, as a low-rank matrix factorization problem
- **NICE** Practical tip: mean normalization
- The [assignment](https://github.com/gopaczewski/coursera-ml/blob/master/mlclass-ex8-005/ex8.pdf) uses a movie dataset and asks to implement the objective function and its gradients
- **NICE** A straightforward and solid baseline

### Harvard: CS109 Data Science

- Two, very practical assignments: [HW3](https://github.com/cs109/content/blob/master/HW3.ipynb) and [HW4](https://github.com/cs109/content/blob/master/HW4.ipynb)
- HW3 tackles a binary prediction task: for a user predict whether we will enjoy or not a given movie based on its review
- **NICE** HW3: Work with the API from Rotten Tomatoes to build a dataset
- HW4 tackles a collaborative filtering task: build a recommendation system for restaurants
- **VERY NICE** HW4: Very thorough assignment: data exploration and visualization, wide diverse of methods and practical suggestions on scaling up the system

### Washington: CSE446 Machine Learning

- The [assignment](https://courses.cs.washington.edu/courses/cse446/15sp/assignments/2/) is concerned with the well-known Netflix dataset.
- **NICE** The assignment works through the methods presented in the paper [Empirical Analysis of Predictive Algorithms for Collaborative Filtering](https://courses.cs.washington.edu/courses/cse446/15sp/assignments/2/algsweb.pdf).

### Brown: CS195 Introduction to Machine Learning

- The [assignment](http://cs.brown.edu/courses/cs195-5/spring2012/homework/hw9.pdf) is on the MovieLens data
- Uses Factor Analysis and Bayesian Linear Regression to solve the task
- The approaches presented in this assignment might be a bit to technical for the task at hand, but they certainly suits my interests and I'm going to revisit them later

### Harvard: CS281 Advanced Machine Learning

- There are two assignments; both of them use a cleaned up version of the Jester dataset.
- **NICE** The [first assignment](http://www.seas.harvard.edu/courses/cs281/files/assignment-1.pdf) presents a more exploratory task: modeling the ratings using various distributions without knowledge of the users or movies.
- The [second assignment](http://www.seas.harvard.edu/courses/cs281/files/assignment-3.pdf) has three problems:
  1. Clustering Jokes and Ratings with Expectation Maximization
  2. GLM Regression of Ratings with Text Features
  3. Modeling Users and Jokes with a Latent Linear Model
- They are all very interesting, but they also seem quite hard: not sure if I have enough time to go through them; definitely try them later.

### Toby Segaran: Programming Collective Intelligence

- **NICE** Chapter 2 is an introduction on collaborative filtering with lots of coding examples and applications (movie and link recommendations)
- Presents straightforward ways of computing the item-item and user-user similarities based on Euclidean distance or Pearson correlation
- Two methods for predicting the rating `r[i, j]` of user `i` for item `j`:

```
r_user_1[i, j] = sum(user_sim[i, :] * r[:, j]) / sum(user_sim[i, :])
r_item_2[i, j] = sum(item_sim[:, j] * r[i, :]) / sum(item_sim[:, j])
```

- The item-based prediction is usually more practical: item-item similarities change less frequently than user-user similarities, hence we can cache them and only recompute them at low-traffic times.

### David Barber: Bayesian Reasoning and Machine Learning

- Section 15.5.2 presents a collaborative filtering approach using PCA with missing data
- I think this is similar to the methods presented by Andrew Ng in his Coursera course

### Kevin Murphy: Machine Learning: A Probabilistic Approach

- Section 27.6.2 presents probabilistic matrix factorization for collaborative filtering
