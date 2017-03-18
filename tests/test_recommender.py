import numpy as np

from giggle.recommender import (
    RECOMMENDERS,
)

from giggle.data import (
    DATASETS,
)


dataset = DATASETS['small']()


class TestBaseline:

    recommender = RECOMMENDERS['baseline']

    def test_error_improves(self):
        TOL = 0.1
        NR_ITERS = 20
        reco = TestBaseline.recommender
        data = dataset.get_data()
        reco.mu = data.data_frame.rating.mean()
        prev_rmse = np.inf
        for i, _ in enumerate(reco._update_params(data)):
            if i == 20:
                break
            curr_rmse = reco._compute_rmse(data.data_frame)
            assert curr_rmse - prev_rmse < TOL
            prev_rmse = curr_rmse


class TestNeighbourhood:

    recommender = RECOMMENDERS['neigh']
    recommender.fit(dataset.get_data(), verbose=0)

    def test_sims(self):
        sims = TestNeighbourhood.recommender.sims
        assert sims.shape[0] == sims.shape[1]
        assert np.all(np.logical_and(-1 <= sims, sims <= 1))
        assert np.allclose(np.diag(sims), 1)

    def test_user_joke_matrix(self):
        mat = TestNeighbourhood.recommender.user_joke_matrix
        nr_users = len(TestNeighbourhood.recommender.data.users)
        nr_jokes = len(TestNeighbourhood.recommender.data.jokes)
        assert mat.shape == (nr_users, nr_jokes)
