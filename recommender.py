import numpy as np
import movie_lens
import pickle


class Recommender(object):
    def __init__(self, udata):
        self.udata = udata
        self.users = None
        self.movies = None
        self.reviews = None

        # Descriptive properties
        self.build_path = None
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


if __name__ == "__main__":
    data_path = "ml-100k/u.data"
    model = Recommender(data_path)
    print(model.sparsity())
    print(model.density())
