import heapq
import pickle
import time
from operator import itemgetter

import numpy as np

import movie_lens
import svd


class Recommender(object):
    def __init__(self, udata):
        self.udata = udata
        self.users = None
        self.movies = None
        self.reviews = None

        # Descriptive properties
        self.build_start = None
        self.build_finish = None
        self.description = None

        # Model properties
        self.model = None
        self.features = 2
        self.steps = 5000
        self.alpha = 0.0002
        self.beta = 0.02

        self.load_dataset()

    def load_dataset(self):
        """
        Load an index of users & movies as a heap and reviews table as an N * M array where N is
        the number of users and M is the number of movies.
        """
        self.users = set([])
        self.movies = set([])
        for review in movie_lens.load_reviews(self.udata):
            self.users.add(review['user_id'])
            self.movies.add(review['movie_id'])

        self.users = sorted(self.users)
        self.movies = sorted(self.movies)

        self.reviews = np.zeros(shape=(len(self.users), len(self.movies)))
        for review in movie_lens.load_reviews(self.udata):
            uid = self.users.index(review['user_id'])
            mid = self.movies.index(review['movie_id'])
            self.reviews[uid, mid] = review['rating']

    def sparsity(self):
        """
        Return the percentage of elements that are zero in the array
        """
        return 1 - self.density()

    def density(self):
        """
        Return the percentage of elements that are nonzero in the array
        """
        nonzero = float(np.count_nonzero(self.reviews))
        return nonzero / self.reviews.size

    @classmethod
    def load(cls, pickle_path):
        """
        Instantiate the class by deserializing the pickle
        """
        with open(pickle_path, 'rb') as pkl:
            return pickle.load(pkl)

    def dump(self, pickle_path):
        """
        Dump the object into a serialized file using the pickle module
        """
        with open(pickle_path, 'wb') as pkl:
            pickle.dump(self, pkl)

    def build(self, output=None):
        """
        Train the model by employing matrix factorization on training data set
        """
        options = {
            'K': self.features,
            'steps': self.steps,
            'alpha': self.alpha,
            'beta': self.beta
        }

        self.build_start = time.time()
        P, Q = svd.factor(self.reviews, **options)
        self.model = np.dot(P, Q.T)
        self.build_finish = time.time()

        if output:
            self.dump(output)

    def predict_ranking(self, user, movie):
        uid = self.users.index(user)
        mid = self.movies.index(movie)
        if self.reviews[uid, mid] > 0:
            return None
        return self.model[uid, mid]

    def top_rated(self, user, n=12):
        movies = [(mid, self.predict_ranking(user, mid)) for mid in self.movies]
        return heapq.nlargest(n, movies, key=itemgetter(1))


if __name__ == "__main__":
    # data_path = "ml-100k/u.data"
    # model = Recommender(data_path)
    # model.build('record.pickle')
    rec = Recommender.load('record.pickle')
    for item in rec.top_rated(234):
        print("%i: %0.3f" % item)
